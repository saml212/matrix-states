"""h2h_cell_train_rd.py -- DEPLOY-STAGE WIRING for HEAD_TO_HEAD_DEMO_DESIGN.md
sec 1.19 items (3)-(6): the REAL train/eval callables the build stage
deliberately dependency-injected (h2h_calibration_wrappers_rd.py /
h2h_sweep_runner_rd.py / h2h_msweep_fanout_rd.py all take `train_fn`/
`eval_fn` params; this module supplies them and the per-cell CLI the chain
script drives).

WHAT THIS FILE OWNS (and the build-stage files deliberately do NOT):
  - build_arm_model: the three rung-1 arms at their PINNED configs
    (contender = DeltaNetLM frozen-bias per_token lambda=0.58, sec 1.2;
    ablation = AblationLM d_state-pinned, sec 1.3(a); transformer =
    TransformerLM n_layers=2 R3-F3 pin, sec 1.3(b)).
  - train_grammar_cell: tasks 1/2 (grammar_rd episodes) joint training,
    sec 1.3.1.3's exact loss (loss_CE + aux_weight * mean(1 - cos(pred,
    target)); aux_weight REVISED 0.1 -> 2.0 at calibration, see AUX_WEIGHT),
    probe head trained jointly from step 0, recovered_frac@0.9 learning
    curves logged every EVAL_EVERY steps (axis 1, sec 1.4.1) alongside
    probe_cos_mean (pre-threshold diagnosability addition), held-out-hop
    eval for task 2, aux-vs-CE tap-point gradient-norm ratio measured at
    step 500 (sec 1.3.1.3's aux_weight revision trigger -- MEASURED AND
    REPORTED here, never auto-revised; freezing is the coordinator's
    margin-freeze decision) plus an aux-swamps-CE overshoot guard.
  - train_lm_cell: task 3 (static corpora) arch-generic LM training,
    mirroring lm_pretrain_rd.train()'s own regime (fp32 + set_and_log_tf32,
    AdamW, cosine schedule, grad-clip 1.0, non-finite-grad skip, FIX-1
    fixed-seed eval windows) without that function's DeltaNetLM-only
    diagnostics.
  - Timing pilots (sec 1.7 gate 2): --pilot-pair measures REAL s/step per
    arch x task via the SAME code path as training; --msweep-pilot is
    R3-F4's checkpoints-resident 2-M-value pilot, driven through the
    AUDITED run_msweep_timing_pilot wrapper.
  - The axis-2 capped-inference eval pass (M-sweep fan-out + contender
    horizon references).
  - sec 1.7 gate 5 launch-token enforcement: every GPU mode REFUSES to run
    unless BOTH env tokens (HEADTOHEAD_PI_SIGNOFF=1,
    HEADTOHEAD_MATCH_GATE_SIGNOFF=1) are set AND the gates 6+7 box token
    files exist. --token-probe exposes this check for the chain's
    negative-test triple.

DEPLOY-STAGE PINS (flagged, not silently improvised -- each carried into
the CALIBRATION_COMPLETE report for coordinator review at margin freeze):
  DEPLOY-PIN-1: n_query_train=8 queries/episode during TRAINING (symmetric
    across all three arms; full-K queries at EVAL). The design pins the
    joint-loss formula but not the training query count; the Transformer's
    tap replicates context per query (sec 1.3.1.2's own disclosed cost), so
    full-K training queries would triple that arm's step cost asymmetrically.
  DEPLOY-PIN-2: task3 trains on openr1-mix-ext (frozen_bias_lm_sweep.py's
    own calibration_cell precedent), with BOTH corpora's val losses logged
    (lm_pretrain_rd's own OTHER_CORPUS pairing convention).
  DEPLOY-PIN-3: tasks 1/2 model vocab = pools.vocab_size_total (50259: GPT-2
    base + BUFFER + <Q>), per sec 1.3.1.1's T_val over vocab_size_total.
    The two reserved rows are ordinary learned embeddings in ALL THREE arms
    symmetrically (no per-arm asymmetry; model_rd's buffer-row zero-pinning
    is that architecture's own convention, not part of the three arms' pin).
  DEPLOY-PIN-4: axis-2 long-horizon contexts = the task's bind sequence
    right-padded with BUFFER filler tokens to the absolute horizon length
    (H2=454/H4=902/H8=1798, sec 1.4.2) -- bind->query distance therefore
    equals the horizon. The design's "filler/distractor tokens" language is
    satisfied with the task's own neutral BUFFER token; distractor-CONTENT
    filler is a coordinator-reviewable variant, not built here.
  DEPLOY-PIN-5: task2 config K=32, H_train=(1,2), H_test=(3,4) (sec 1.4's
    task-2 pin), n_query per DEPLOY-PIN-1.

Run the CPU selftest: REASONING_LINK_FORCE_CPU_STUB=1 python h2h_cell_train_rd.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_fla_stub_rd import ensure_fla_stub

_STUB_INSTALLED = ensure_fla_stub()
from lm_pretrain_rd import (DeltaNetLM, load_corpus, get_batch, get_lr,          # noqa: E402
                            corpus_fixed_seed, set_and_log_tf32,
                            DEFAULT_DATA_DIR, OTHER_CORPUS, _MIN_KERNEL_T)
from ablation_mixer_rd import AblationLM                                          # noqa: E402
from transformer_baseline_rd import (TransformerLM, sink_fifo_mask,               # noqa: E402
                                     cap_length_tokens)
import probe_head_rd as ph                                                        # noqa: E402
from grammar_rd import (DeltaNetRDTaskConfig, build_entity_pools,                 # noqa: E402
                        sample_batch_rd)
from h2h_calibration_wrappers_rd import (build_full_calibration_manifest,         # noqa: E402
                                         run_msweep_timing_pilot,
                                         project_and_gate_msweep_fanout,
                                         check_gate1_full_cell_band)
from h2h_sweep_runner_rd import (build_27_cell_manifest, REQUIRED_RESULT_KEYS,    # noqa: E402
                                 is_valid_result)
from h2h_msweep_fanout_rd import build_90_pass_manifest, run_msweep_fanout        # noqa: E402

# ---------------------------------------------------------------------------
# Pinned configs (sec 1.2 / 1.3 / 1.4 / 1.6)
# ---------------------------------------------------------------------------

VOCAB_BASE = 50257
CONTENDER_KW = dict(d_model=256, d_state=64, n_layers=2, conv_size=4, num_heads=1, ffn_mult=4,
                    frozen_bias_arm="per_token", frozen_bias_lambda=0.58)   # sec 1.2 pin
ABLATION_KW = dict(d_model=256, d_state=64, n_layers=2, conv_size=4, ffn_mult=4,
                   gate_extra_width=0)                                       # sec 1.3(a), d_state pinned
TRANSFORMER_KW = dict(d_model=256, n_layers=2, n_heads=4, ffn_mult=4)        # sec 1.3(b), R3-F3 pin
TAP_DIM = {"contender": 64, "ablation": 64, "transformer": 256}
VALUE_DIM = 64                                    # sec 1.3.1.1: value_dim = contender d_state at rung

FULL_STEPS = 20_000                               # sec 1.6's 20,000-step cell (0.2524 GPU-h realized rate)
WARMUP_STEPS = 100                                # M3 pin (h2h_calibration_wrappers_rd.WARMUP_STEPS)
WEIGHT_DECAY = 0.01                               # lm_pretrain_rd default, reused for every arm
GRAMMAR_BATCH = 32
GRAMMAR_EVAL_EVERY = 500                          # axis-1 learning-curve resolution (sec 1.4.1)
LM_SEQ_LEN, LM_BATCH = 512, 32                    # frozen_bias_lm_sweep.py's own rung-1 regime
LM_EVAL_EVERY, LM_EVAL_BATCHES, LM_EVAL_BATCH_SIZE = 1000, 8, 16
N_QUERY_TRAIN = 8                                 # DEPLOY-PIN-1
EVAL_EPISODE_BATCHES = 4                          # 4 x 32 = 128 fixed eval episodes per hop-set
EVAL_SEED = 20260710                              # fixed eval-episode seed, IDENTICAL across arms/seeds
                                                  # (AUDIT FIX-1 discipline applied to episodes); distinct
                                                  # from PROBE_TARGET_SEED/ANCHOR_INIT_SEED by construction
TASK3_CORPUS = "openr1-mix-ext"                   # DEPLOY-PIN-2
HORIZON_TOKENS = {"H2": 454, "H4": 902, "H8": 1798}   # sec 1.4.2 absolute-token horizons
K_SINK = 4                                        # F-NEW-2 sink+FIFO pin
GRAD_RATIO_STEP = 500                             # sec 1.3.1.3's aux_weight revision-trigger step
# sec 1.3.1.3 CALIBRATION REVISION (2026-07-09, the pre-registered trigger FIRED):
# the aux_weight=0.1 calibration run measured ce/aux backbone grad-norm ratio ~20.9x
# at step 500 (ce 0.1375 vs aux 0.0066; exceeds_10x_trigger=True on all nine task-1/2
# cells, all three arms; recovered_frac@0.9=0.0 probe-wide). The design pinned the
# TRIGGER (>10x) but left the revision TARGET unpinned; we target GRADIENT PARITY at
# the tap: 0.1 * 20.93 ~= 2.09, pinned at the round value 2.0 (rounding disclosed).
# Frozen before the 27-cell sweep. The aux_weight=0.1 run's artifacts stay archived
# (calib/archive_auxweight0.1/ on box); re-run cells carry the _auxrev2 name suffix.
AUX_WEIGHT = 2.0                                  # was ph.AUX_WEIGHT_DEFAULT (0.1)

# sec 1.3.1.3 Rev 4 / sec 1.23 build-fix item 1: CE_answer -- pinned default 1.0, calibrated by
# the SAME step-500 gradient-ratio dial extended to three losses (item 2, below).
CE_ANSWER_WEIGHT = ph.CE_ANSWER_WEIGHT_DEFAULT     # 1.0

GATE_TOKEN_FILES = ("GATE6_MATCH_GATE_PASSED.token", "GATE7_PROBE_CAPACITY_NULL_PASSED.token")

# ---------------------------------------------------------------------------
# sec 1.23 build-fix item 2: three-loss step-500 gradient-ratio dial + R5-F1's round cap.
# ---------------------------------------------------------------------------

DIAL_TRIGGER_RATIO = 10.0          # sec 1.3.1.3's own exceeds_10x_trigger convention, extended
DIAL_ROUND_ENV = "H2H_DIAL_ROUND"  # R5-F1's own "wired as ... chain config": the orchestrating
                                    # chain script (h2h_rung1_chain.sh, OUTSIDE this scoped build-
                                    # fix's file list) owns bumping this between re-runs; this
                                    # module only READS and enforces the cap it implies.
DEFAULT_DIAL_ROUND = 3             # round 1 = aux_weight 0.1 (sec 1.21), round 2 = aux_weight 2.0
                                    # (_auxrev2, sec 1.21), round 3 = THIS Rev 4 recalibration's
                                    # own first pass at the three-loss dial (this build-fix's
                                    # default -- no round bump needed to exercise it the first time)
MAX_DIAL_ROUNDS = 4                # R5-F1's own pin: round 4 is the ONE permitted contingency
                                    # round; a trigger firing AT round 4 is a hard-stop, never a
                                    # round 5 dial revision
DIAL_SENTINEL_PATH_DEFAULT = "results/h2h_rung1/DIAL_EXHAUSTED.token"


class DialExhaustedError(RuntimeError):
    """R5-F1 (attack round 5, sec 1.23): the three-loss step-500 gradient-parity dial still
    fires on round MAX_DIAL_ROUNDS -- a HARD-STOP into a fresh sec-1.21-style diagnosis
    (chain-of-elimination on the saved checkpoints), never a further dial revision. Never
    caught by train_grammar_cell itself -- it is meant to propagate to the calling chain/CLI,
    exactly like the existing sec 1.7 gate 5/margin-freeze SystemExit refusals."""


def current_dial_round() -> int:
    """R5-F1's own round counter, read from DIAL_ROUND_ENV (coordinator/chain-set; see that
    constant's own comment for the file this build-fix does NOT touch)."""
    return int(os.environ.get(DIAL_ROUND_ENV, DEFAULT_DIAL_ROUND))


def _dial_ratio_and_fire(n_ce: float, n_other: float) -> tuple[float | None, bool]:
    """sec 1.3.1.3's own exceeds_10x_trigger convention: ratio = ce_norm / other_norm (both
    WEIGHTED by their own loss's current coefficient, matching the original aux_weight
    0.1->2.0 revision's own convention exactly); fires when ratio falls outside [0.1, 10.0].
    n_other<=0 (a dead/disconnected gradient) is treated as an unconditional fire -- there is
    no finite ratio to report, and a zero-gradient auxiliary is never a healthy calibration."""
    if n_other <= 0:
        return None, True
    ratio = n_ce / n_other
    return ratio, not (0.1 <= ratio <= DIAL_TRIGGER_RATIO)


def evaluate_three_loss_dial(ce_grad_norm: float, aux_grad_norm: float, ce_answer_grad_norm: float,
                              aux_weight: float, ce_answer_weight: float, round_idx: int) -> dict:
    """sec 1.3.1.3's step-500 THREE-loss gradient-ratio dial, extended per R5-F1's cap (sec
    1.23). Pure function (no I/O, no GPU) so it is exhaustively negative-testable with
    synthetic norms -- CLAUDE.md's own 'run the negative unit test... to completion' rule.

    round_idx: 1-indexed calibration-dial round counter (see DEFAULT_DIAL_ROUND's own comment
    for what each round number means historically).

    Revision rule, pinned exactly (sec 1.3.1.3): if exactly one auxiliary's ratio fires,
    revise that ONE weight toward parity (new_weight = old_weight * observed_ratio). If BOTH
    fire simultaneously, revise only the LARGER-ratio DEVIATION this round (max(ratio,
    1/ratio) -- symmetric around parity) and defer the other to the next round's own one
    revision -- never both in the same round. If round_idx >= MAX_DIAL_ROUNDS and a trigger
    STILL fires, this is R5-F1's DIAL_EXHAUSTED verdict: no further revision, a hard-stop."""
    r_aux, aux_fires = _dial_ratio_and_fire(ce_grad_norm, aux_grad_norm)
    r_ce_answer, ce_answer_fires = _dial_ratio_and_fire(ce_grad_norm, ce_answer_grad_norm)

    result = {
        "round_idx": round_idx, "max_dial_rounds": MAX_DIAL_ROUNDS,
        "ce_grad_norm_backbone": ce_grad_norm, "aux_grad_norm_backbone": aux_grad_norm,
        "ce_answer_grad_norm_backbone": ce_answer_grad_norm,
        "ratio_ce_over_aux": r_aux, "ratio_ce_over_ce_answer": r_ce_answer,
        "aux_fires": aux_fires, "ce_answer_fires": ce_answer_fires,
        "old_aux_weight": aux_weight, "old_ce_answer_weight": ce_answer_weight,
    }

    if not (aux_fires or ce_answer_fires):
        result.update(status="PASS", revise=None,
                      new_aux_weight=aux_weight, new_ce_answer_weight=ce_answer_weight)
        return result

    if round_idx >= MAX_DIAL_ROUNDS:
        result.update(status="DIAL_EXHAUSTED", revise=None,
                      new_aux_weight=aux_weight, new_ce_answer_weight=ce_answer_weight,
                      message=(
                          f"R5-F1 DIAL_EXHAUSTED at round {round_idx} (cap={MAX_DIAL_ROUNDS}): a "
                          f"gradient-parity trigger still fires (aux_fires={aux_fires} "
                          f"ratio_ce_over_aux={r_aux}, ce_answer_fires={ce_answer_fires} "
                          f"ratio_ce_over_ce_answer={r_ce_answer}) after the single R5-F1 "
                          f"contingency round. Per sec 1.21's own diagnosis discipline, this is a "
                          f"HARD-STOP into a FRESH diagnosis (chain-of-elimination on the saved "
                          f"checkpoints -- task-routing failure? tap-placement problem? "
                          f"probe-capacity/Nichani gap? see the diagnostic ladder's own "
                          f"attribution table, sec 1.7 gate 1a), never a further dial revision. "
                          f"Do not re-run the sweep; escalate to a design-revision round."))
        return result

    if aux_fires and ce_answer_fires:
        dev_aux = max(r_aux, 1.0 / r_aux) if r_aux is not None and r_aux > 0 else float("inf")
        dev_ce_answer = (max(r_ce_answer, 1.0 / r_ce_answer)
                         if r_ce_answer is not None and r_ce_answer > 0 else float("inf"))
        revise = "aux_weight" if dev_aux >= dev_ce_answer else "ce_answer_weight"
        deferred = "ce_answer_weight" if revise == "aux_weight" else "aux_weight"
    elif aux_fires:
        revise, deferred = "aux_weight", None
    else:
        revise, deferred = "ce_answer_weight", None

    new_aux = aux_weight * r_aux if (revise == "aux_weight" and r_aux is not None) else aux_weight
    new_ce_answer = (ce_answer_weight * r_ce_answer
                     if (revise == "ce_answer_weight" and r_ce_answer is not None) else ce_answer_weight)
    result.update(status="REVISE", revise=revise, deferred=deferred,
                  new_aux_weight=new_aux, new_ce_answer_weight=new_ce_answer)
    return result


def _write_dial_exhausted_sentinel(path: str, cell_name: str, dial: dict) -> None:
    _atomic_dump(path, {"cell_name": cell_name, "dial": dial, "message": dial.get("message")})


_POOLS_CACHE: dict = {}


def get_pools(device: str):
    """build_entity_pools returns (pools, report); tokenizer load is slow, so
    cache per (process, device)."""
    if device not in _POOLS_CACHE:
        pools, _report = build_entity_pools()
        _POOLS_CACHE[device] = pools.to(device)
    return _POOLS_CACHE[device]


def task_cfg(task: str, K: int | None, n_query: int | None) -> DeltaNetRDTaskConfig:
    """task1: single-hop recall (hop_set=(1,)); H_test=(2,)/H_extra=() are
    config-validity placeholders never sampled. task2: DEPLOY-PIN-5."""
    if task.startswith("task1"):
        return DeltaNetRDTaskConfig(K=K if K is not None else 32, conv_size=4, n_query=n_query,
                                    H_train=(1,), H_test=(2,), H_extra=())
    if task.startswith("task2"):
        return DeltaNetRDTaskConfig(K=32, conv_size=4, n_query=n_query,
                                    H_train=(1, 2), H_test=(3, 4), H_extra=())
    raise ValueError(f"task_cfg is for grammar tasks only, got {task!r}")


# ---------------------------------------------------------------------------
# sec 1.7 gate 5 -- launch-token enforcement (env pair + gates-6/7 box files)
# ---------------------------------------------------------------------------

def require_launch_tokens(gates_dir: str) -> None:
    """Hard-refuses (exit 3) unless BOTH env tokens are '1' AND both gate
    token files exist. The chain's negative-test triple drives this via
    --token-probe with each token independently removed."""
    missing = [v for v in ("HEADTOHEAD_PI_SIGNOFF", "HEADTOHEAD_MATCH_GATE_SIGNOFF")
               if os.environ.get(v) != "1"]
    absent = [f for f in GATE_TOKEN_FILES if not os.path.isfile(os.path.join(gates_dir, f))]
    if missing or absent:
        print(f"REFUSE (sec 1.7 gate 5): missing env tokens={missing} "
              f"missing gate token files={absent} (gates_dir={gates_dir})", file=sys.stderr)
        raise SystemExit(3)


def require_margins_frozen(margins_token: str) -> None:
    """CODE-LEVEL sweep-release gate (deploy audit MAJOR-2): the 27-cell
    sweep, the fan-out, and the horizon references are unreachable until
    the coordinator writes MARGINS_FROZEN.token -- enforced HERE, not only
    by the chain script's stage ordering. Hard-refuse: exit 4."""
    if not os.path.isfile(margins_token):
        print(f"REFUSE (margin freeze): {margins_token} missing -- the 27-cell sweep / fan-out "
              f"is NOT released until the coordinator records the margin freeze.", file=sys.stderr)
        raise SystemExit(4)


# ---------------------------------------------------------------------------
# Arms + probe rig
# ---------------------------------------------------------------------------

def build_arm_model(arch: str, vocab_size: int, seed: int, device: str):
    torch.manual_seed(seed)
    if arch == "contender":
        m = DeltaNetLM(vocab_size, **CONTENDER_KW)
    elif arch == "ablation":
        m = AblationLM(vocab_size, **ABLATION_KW)
    elif arch == "transformer":
        m = TransformerLM(vocab_size, **TRANSFORMER_KW)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return m.to(device)


class ProbeRig:
    """T_val (frozen buffer) + shared_probe + per-arm adapter, all built via
    the AUDITED probe_head_rd factories -- this class only owns placement
    and (de)serialization."""

    def __init__(self, arch: str, vocab_size_total: int, device: str):
        self.arch = arch
        self.T_val = ph.build_probe_target_table(vocab_size_total, VALUE_DIM).to(device)
        self.T_val.requires_grad_(False)
        self.adapter = ph.build_adapter_arm(TAP_DIM[arch], VALUE_DIM).to(device)
        self.probe = ph.build_shared_probe(VALUE_DIM).to(device)

    def parameters(self):
        return list(self.adapter.parameters()) + list(self.probe.parameters())

    def pred(self, tap: torch.Tensor) -> torch.Tensor:
        return self.probe(self.adapter(tap))

    def state_dict(self):
        return {"adapter": self.adapter.state_dict(), "probe": self.probe.state_dict()}

    def load_state_dict(self, d):
        self.adapter.load_state_dict(d["adapter"])
        self.probe.load_state_dict(d["probe"])


def answer_token_ids(batch: dict) -> torch.Tensor:
    """The true answer entity's token id per query -- tgt_slot/entity_ids.gather convention
    (sec 1.3.1.1). Factored out of probe_targets so CE_answer (sec 1.3.1.3, Rev 4) and the
    probe target share ONE gather, never two copies that could silently drift apart."""
    return torch.gather(batch["entity_ids"], 1, batch["tgt_slot"])   # (B,Q)


def probe_targets(rig: ProbeRig, batch: dict) -> torch.Tensor:
    """target = T_val[token id of the entity at the query's target slot]
    (sec 1.3.1.1's target(entity_token_id) = T_val[entity_token_id])."""
    return rig.T_val[answer_token_ids(batch)]                                  # (B,Q,VALUE_DIM)


# ---------------------------------------------------------------------------
# Taps: audited (eval) + fused (training -- ONE context forward feeds both
# CE and the tap, verified equivalent to the audited functions at step 0)
# ---------------------------------------------------------------------------

AUDITED_TAP = {"contender": ph.contender_native_tap, "ablation": ph.ablation_native_tap,
               "transformer": ph.transformer_native_tap}


def _q_last_pathway(model, query_tokens: torch.Tensor) -> torch.Tensor:
    """Mirrors probe_head_rd.contender_native_tap's query pathway VERBATIM
    (embed -> last block norm1 -> q_proj -> q_conv1d -> last position) --
    equivalence to the audited function is asserted at step 0 of every
    grammar cell (see fused_logits_and_tap)."""
    B, Q, qlen = query_tokens.shape
    last_block = model.blocks[-1]
    a = last_block.norm1(model.embed(query_tokens.reshape(B * Q, qlen)))
    q_conv, _ = last_block.mixer.q_conv1d(last_block.mixer.q_proj(a))
    return q_conv[:, -1, :].view(B, Q, -1)


def _recurrent_tap_from_states(arch: str, final_states: list, q_last: torch.Tensor) -> torch.Tensor:
    """sec 1.3.1.2's per-arm tap formula, factored out of both the training-time fused path and
    the (eval-time, no_grad) diagnostic-ladder path so the two never drift: contender ->
    S_T_last @ q_query (matvec); ablation -> s_T (.) q_query (Hadamard)."""
    if arch == "contender":
        S_T_last = final_states[-1]
        assert S_T_last.shape[1] == 1, "contender tap assumes num_heads==1 (rung config)"
        return torch.einsum("bij,bqj->bqi", S_T_last.squeeze(1), q_last)
    return final_states[-1].unsqueeze(1) * q_last


# ---------------------------------------------------------------------------
# sec 1.3.1.3 Rev 4 / sec 1.23 build-fix item 1: the CE_answer continuation.
#
# The query window is short (query_len = buf_len+3, e.g. 6 tokens at conv_size=4) --
# WELL BELOW lm_pretrain_rd._MIN_KERNEL_T (128), the real chunk_delta_rule kernel's own
# backward-crash floor (F15-LM), which DeltaNetLMMixer.forward enforces unconditionally
# (Python-level assert, fires identically under the CPU stub or the real kernel -- verified by
# reading lm_pretrain_rd.py directly, not assumed). A raw model.forward(query_tokens,
# initial_states=S_T) call as sec 1.3.1.3's prose literally states would therefore CRASH for
# the contender (and, for construction symmetry within "recurrent arms," is applied to the
# ablation too, even though its own mixer has no such floor) -- this is a real build-time gap
# the design prose does not address, disclosed here and resolved by right-padding the query
# with BUFFER filler up to _MIN_KERNEL_T BEFORE the continuation call, then reading the answer
# logit at the ORIGINAL (pre-padding) <Q> position. This is causally inert at that position:
# a causal model's logits at position t depend ONLY on tokens [0..t], never on anything
# appended AFTER t, so padding placed strictly after the <Q> position cannot affect the
# answer-position logit or its gradient -- the same "right-pad, read an earlier position"
# principle DEPLOY-PIN-4 already uses for axis 2's horizon extension (extend_context_to_horizon,
# below), applied here at continuation-call granularity. Padding BEFORE the query would NOT be
# safe (lm_pretrain_rd.py's own docstring: "LM mode has no state-neutral pad token" -- a
# padded prefix would keep updating state before the query is even read, breaking the "pure
# function of (S_T, query_tokens) alone" purity sec 1.3.1.3's own text requires) -- right-pad
# ONLY, never left-pad. FLAGGED for the coordinator: padding B*Q short queries to 128 tokens
# EACH is a real compute cost sec 1.3.1.3's own "~2.3 GPU-h re-calibration" estimate does not
# appear to account for (at DEPLOY-PIN-1's N_QUERY_TRAIN=8, GRAMMAR_BATCH=32, this is 256 rows
# x 128 tokens = 32,768 tokens/step for the continuation ALONE, comparable to or exceeding the
# K=32 bind-phase pass's own 32*224=7,168 tokens/step) -- not something this scoped build-fix
# is authorized to redesign away (that would be a design-doc change, out of sec 1.23's 6-item
# scope), reported here per this program's own "flag, don't paper over" convention.
# ---------------------------------------------------------------------------

def _pad_query_tokens_for_continuation(query_tokens_flat: torch.Tensor, buffer_id: int,
                                       min_t: int = _MIN_KERNEL_T) -> torch.Tensor:
    """Right-pads (N, qlen) query rows with buffer_id up to max(qlen, min_t). See the module
    section docstring above for why right-padding (never left-padding) is the only safe choice,
    and why padding is a no-op on the answer-position logit under causal masking."""
    N, qlen = query_tokens_flat.shape
    if qlen >= min_t:
        return query_tokens_flat
    pad = torch.full((N, min_t - qlen), buffer_id, dtype=query_tokens_flat.dtype,
                     device=query_tokens_flat.device)
    return torch.cat([query_tokens_flat, pad], dim=1)


def _repeat_states_for_queries(final_states: list, Q: int) -> list:
    """Each layer's cached bind-phase state (B, ...) is repeated Q times along the batch axis
    so the Q independent per-episode queries can be scored in ONE flattened (B*Q, ...) batched
    continuation call, each starting from the SAME S_T -- never chained query-to-query (sec
    1.3.1.3's own "a PURE function of (S_T, query_tokens) alone" purity, per query)."""
    out = []
    for s in final_states:
        if s is None:
            out.append(None)
            continue
        B = s.shape[0]
        out.append(s.unsqueeze(1).expand(B, Q, *s.shape[1:]).reshape(B * Q, *s.shape[1:]))
    return out


def _recurrent_continuation_answer_logits(arch: str, model, final_states: list,
                                          query_tokens: torch.Tensor, buffer_id: int) -> torch.Tensor:
    """sec 1.3.1.3 Rev 4's continuation: model.forward(query_tokens, initial_states=S_T),
    padded per the module section docstring above. Returns (B,Q,vocab) answer-position logits
    -- read at index qlen-1 (the <Q> position), which causal masking guarantees is IDENTICAL
    whether or not the padding is present (verified by this file's own blank-out-style smoke).

    AUD2-F1 fix (sec 1.24 pre-launch build-fix): the pre-fix version ran the vocab-size LM
    head (F.linear against embed.weight) over ALL _MIN_KERNEL_T (128) padded positions via a
    plain model.forward(...) call, then discarded 127/128 rows -- audited at 6.57x/6.81x added
    per-step cost (contender/ablation) over the pre-CE_answer baseline (lm_pretrain_rd.py:1298
    path). This version calls model.forward(..., return_hidden=True) to get the POST-norm_f
    hidden state (B*Q,T,d_model) -- NO vocab-size matmul yet -- slices to the ANSWER POSITION
    FIRST, then applies the LM head to ONLY that (B*Q,d_model) slice, cutting the wasted matmul
    ~99%. Bit-identical to the pre-fix path at the answer position (mode_selftest's own
    "AUD2-F1 fix verification" item re-derives the pre-fix computation locally and asserts
    equality -- not merely assumed)."""
    B, Q, qlen = query_tokens.shape
    flat = query_tokens.reshape(B * Q, qlen)
    padded = _pad_query_tokens_for_continuation(flat, buffer_id)
    states_rep = _repeat_states_for_queries(final_states, Q)
    hidden = model.forward(padded, initial_states=states_rep, return_hidden=True)
    answer_hidden = hidden[:, qlen - 1, :]                        # slice BEFORE the LM-head matmul
    answer_logits = F.linear(answer_hidden, model.embed.weight)   # (B*Q,vocab), not (B*Q,128,vocab)
    return answer_logits.view(B, Q, -1)


def _fused_transformer_tap_and_answer_logits(model, context_token_ids: torch.Tensor,
                                             query_tokens: torch.Tensor, attn_mask_fn=None):
    """Mirrors probe_head_rd.transformer_native_tap's internals VERBATIM (equivalence to the
    audited tap is asserted at step 0, see assert_fused_tap_matches_audited) but additionally
    keeps the SAME forward call's own LM-head logits at the <Q> position -- sec 1.3.1.3's own
    'reuse its existing single-pass logits, no second forward' for the Transformer arm.
    Returns (tap, answer_logits), each (B,Q,*)."""
    B, T_ctx = context_token_ids.shape
    _, Q, qlen = query_tokens.shape
    ctx_rep = context_token_ids.unsqueeze(1).expand(B, Q, T_ctx).reshape(B * Q, T_ctx)
    q_flat = query_tokens.reshape(B * Q, qlen)
    seq = torch.cat([ctx_rep, q_flat], dim=1)
    mask = attn_mask_fn(T_ctx + qlen) if attn_mask_fn is not None else None
    logits, hidden = model(seq, attn_mask=mask, return_hidden=True)
    tap = hidden[:, -1, :].view(B, Q, -1)
    answer_logits = logits[:, -1, :].view(B, Q, -1)
    return tap, answer_logits


def fused_logits_and_tap(arch: str, model, token_ids: torch.Tensor, query_tokens: torch.Tensor,
                         pools):
    """Training-time fused pass: contender/ablation reuse ONE context forward
    for both the CE_lm logits and the tap's final state (the audited tap
    functions run their own full context pass internally -- correct but 2x
    the forward cost when CE_lm needs the same pass anyway). Transformer has no
    shared pass to reuse for the TAP (its tap structurally requires the [ctx ++ query]
    replication, sec 1.3.1.2), but DOES reuse that SAME per-query pass for its answer_logits
    (sec 1.3.1.3's 'no second forward' for the Transformer arm).

    Returns (logits, tap, answer_logits) -- ALL THREE ARMS SYMMETRIC in this return shape
    (sec 1.23 build-fix item 1), asymmetric only in HOW answer_logits is produced (the SAME
    sec 1.3.1.2 asymmetry this design already discloses, sec 1.9 item 9)."""
    if arch == "transformer":
        logits = model(token_ids)
        tap, answer_logits = _fused_transformer_tap_and_answer_logits(
            model, token_ids, query_tokens, attn_mask_fn=None)
        return logits, tap, answer_logits
    logits, final_states = model(token_ids, return_states=True)
    q_last = _q_last_pathway(model, query_tokens)
    tap = _recurrent_tap_from_states(arch, final_states, q_last)
    answer_logits = _recurrent_continuation_answer_logits(arch, model, final_states, query_tokens,
                                                           pools.buffer_id)
    return logits, tap, answer_logits


def assert_fused_tap_matches_audited(arch: str, model, token_ids, query_tokens, pools) -> float:
    """Step-0 wiring check: the fused tap must numerically match the audited
    probe_head_rd tap on the same inputs (atol/rtol 1e-3 -- a WIRING check,
    not a numerics benchmark). Under the CPU stub the contender's state is
    identically zero, so the check is trivially true there but still runs
    (shape + code-path coverage); the real check happens on box. answer_logits
    (Rev 4, NEW) is checked only for finiteness + shape here -- it has no
    pre-existing audited reference to compare against (sec 1.3.1.3's CE_answer
    machinery is itself new this revision)."""
    with torch.no_grad():
        _, fused, answer_logits = fused_logits_and_tap(arch, model, token_ids, query_tokens, pools)
        audited = AUDITED_TAP[arch](model, token_ids, query_tokens)
        diff = (fused - audited).abs().max().item()
    assert torch.allclose(fused, audited, atol=1e-3, rtol=1e-3), (
        f"{arch}: fused training tap diverges from the audited probe_head_rd tap "
        f"(max abs diff {diff}) -- wiring bug, refusing to train")
    B, Q, _ = query_tokens.shape
    assert answer_logits.shape[:2] == (B, Q) and torch.isfinite(answer_logits).all(), (
        f"{arch}: answer_logits shape/finiteness wiring check failed, "
        f"shape={tuple(answer_logits.shape)}")
    return diff


EVAL_CHUNK_BUDGET_BYTES = 8e9    # per-chunk transient-tensor target; H100 80GB leaves wide margin


def _transformer_episode_chunks(b: dict, T_total: int):
    """VRAM guard for the transformer tap (context replicated ONCE PER QUERY,
    B*Q sequences): the dominant transients are the tied-head LOGITS,
    (B*Q, T, 50259) fp32 -- the house-rule bottleneck -- plus attention
    scores (B*Q, n_heads, T, T). Yields episode sub-batches sized so both
    stay within EVAL_CHUNK_BUDGET_BYTES (min 1 episode; at H8 one episode's
    ~13 GB transient is the floor, well inside 80 GB)."""
    q = b["query_tokens"].shape[1]
    per_episode = (q * T_total * (VOCAB_BASE + 2) * 4
                   + q * TRANSFORMER_KW["n_heads"] * T_total * T_total * 4)
    chunk = max(1, int(EVAL_CHUNK_BUDGET_BYTES / per_episode))
    for i in range(0, b["token_ids"].shape[0], chunk):
        yield {k: (v[i:i + chunk] if isinstance(v, torch.Tensor) else v) for k, v in b.items()}


@torch.no_grad()
def eval_recovered_frac(arch: str, model, rig: ProbeRig, eval_batches: list,
                        attn_mask_fn=None) -> tuple[float, float]:
    """recovered_frac@0.9 over the FIXED eval episode set, via the AUDITED
    tap functions. attn_mask_fn (transformer only): capped b-primary read.
    Query-weighted mean; transformer batches are VRAM-chunked.

    Returns (recovered_frac@0.9, probe_cos_mean). probe_cos_mean is the
    query-weighted mean cosine between pred and target BEFORE thresholding
    (diagnosability ADDITION, 2026-07-09 aux revision: distinguishes a
    learning-but-below-bar probe, cos rising toward 0.9, from a dead one,
    cos ~= 0). recovered_frac@0.9 stays the pre-registered decision metric;
    its formula (ph.cosine_recovery_frac) is untouched."""
    model.eval()
    n_hit, n_cos, n_tot = 0.0, 0.0, 0
    for b_full in eval_batches:
        if arch == "transformer":
            T_total = b_full["token_ids"].shape[1] + b_full["query_tokens"].shape[-1]
            chunks = _transformer_episode_chunks(b_full, T_total)
        else:
            chunks = (b_full,)
        for b in chunks:
            if arch == "transformer":
                tap = ph.transformer_native_tap(model, b["token_ids"], b["query_tokens"],
                                                attn_mask_fn=attn_mask_fn)
            else:
                tap = AUDITED_TAP[arch](model, b["token_ids"], b["query_tokens"])
            pred, tgt = rig.pred(tap), probe_targets(rig, b)
            frac = ph.cosine_recovery_frac(pred, tgt).item()
            cos_mean = F.cosine_similarity(pred, tgt, dim=-1).mean().item()
            n_q = b["query_tokens"].shape[0] * b["query_tokens"].shape[1]
            n_hit += frac * n_q
            n_cos += cos_mean * n_q
            n_tot += n_q
    model.train()
    return n_hit / max(1, n_tot), n_cos / max(1, n_tot)


# ---------------------------------------------------------------------------
# sec 1.7 gate 1a -- THE DIAGNOSTIC LADDER (sec 1.23 build-fix item 3).
# Rung 1 (K-restricted gather+argmax top-1) + the membership-oracle tell run over the SAME
# fixed eval episode set eval_recovered_frac (rung 3) uses. Rung 2 (identity classifier) is a
# SEPARATE, one-shot fit per cell (sec 1.6's own "one small extra linear-classifier fit/cell"
# cost accounting), not repeated every eval step.
# ---------------------------------------------------------------------------

def _rung1_k_restricted_pred_slot(answer_logits: torch.Tensor, entity_ids: torch.Tensor) -> torch.Tensor:
    """sec 1.23's own 'build precision note': K-RESTRICTED gather+argmax over the episode's OWN
    K candidate entities' answer-position logits, NEVER a global-vocab top-1 (AUD2-F2 fix, sec
    1.24 pre-launch build-fix: extracted to its own function so it is independently testable
    against a planted-answer synthetic -- mode_selftest's own item plants a NON-candidate token
    at the global-vocab max while a candidate holds the max AMONG the K candidates; this
    function must still return the candidate, never silently degrade to a global top-1)."""
    K = entity_ids.shape[1]
    restricted = torch.gather(
        answer_logits, 2, entity_ids.unsqueeze(1).expand(-1, answer_logits.shape[1], K))
    return restricted.argmax(dim=-1)


@torch.no_grad()
def eval_diagnostic_rung1_and_tell(arch: str, model, rig: ProbeRig, eval_batches: list, pools,
                                   attn_mask_fn=None) -> dict:
    """Rung 1: K-RESTRICTED gather+argmax top-1 -- gather the episode's OWN K candidate
    entities' answer-position logits (NOT a global-vocab top-1, sec 1.23's own 'build precision
    note'), argmax among them, compare to the true SLOT (tgt_slot). PASS bar: > 3x
    episode-restricted chance (1/K). Also logs the membership-oracle tell (sec 1.7 gate 1a):
    cos(pred, episode_mean_of_T_val_rows), the predicted probe vector's cosine to the MEAN of
    the episode's K target rows (not the true individual target) -- an elevated value here,
    together with probe_cos_mean near 1/sqrt(K), is sec 1.21's own diagnosed membership-oracle
    local optimum signature, distinct from genuine per-query recall."""
    model.eval()
    n_hit, n_tot = 0.0, 0
    cos_tell_sum, cos_tell_n = 0.0, 0
    K_seen = None
    for b_full in eval_batches:
        if arch == "transformer":
            T_total = b_full["token_ids"].shape[1] + b_full["query_tokens"].shape[-1]
            chunks = _transformer_episode_chunks(b_full, T_total)
        else:
            chunks = (b_full,)
        for b in chunks:
            if arch == "transformer":
                tap, answer_logits = _fused_transformer_tap_and_answer_logits(
                    model, b["token_ids"], b["query_tokens"], attn_mask_fn=attn_mask_fn)
            else:
                _, final_states = model(b["token_ids"], return_states=True)
                q_last = _q_last_pathway(model, b["query_tokens"])
                tap = _recurrent_tap_from_states(arch, final_states, q_last)
                answer_logits = _recurrent_continuation_answer_logits(
                    arch, model, final_states, b["query_tokens"], pools.buffer_id)

            K_seen = b["entity_ids"].shape[1]
            pred_slot = _rung1_k_restricted_pred_slot(answer_logits, b["entity_ids"])
            n_hit += (pred_slot == b["tgt_slot"]).float().sum().item()
            n_tot += b["tgt_slot"].numel()

            pred = rig.pred(tap)
            ep_mean = rig.T_val[b["entity_ids"]].mean(dim=1, keepdim=True).expand_as(pred)
            cos_tell_sum += F.cosine_similarity(pred, ep_mean, dim=-1).sum().item()
            cos_tell_n += pred.shape[0] * pred.shape[1]
    model.train()
    acc = n_hit / max(1, n_tot)
    chance = 1.0 / K_seen if K_seen else float("nan")
    return {"accuracy": acc, "chance": chance, "pass_bar": ph.RUNG_CHANCE_MULT * chance,
            "passed": acc > ph.RUNG_CHANCE_MULT * chance,
            "cos_pred_episode_mean": cos_tell_sum / max(1, cos_tell_n)}


RUNG2_TRAIN_STEPS = 300     # sec 1.6's own "one small extra linear-classifier fit/cell"
RUNG2_TRAIN_BATCH = 32
RUNG2_EVAL_BATCHES = 4      # mirrors EVAL_EPISODE_BATCHES's own convention


def fit_rung2_identity_classifier(arch: str, model, cfg_eval: DeltaNetRDTaskConfig, hop_set: tuple,
                                  pools, device: str, K: int, seed: int) -> dict:
    """Rung 2: a FRESHLY-trained linear identity-classifier (nn.Linear(native_tap_dim, K),
    probe_head_rd.build_identity_classifier -- SEPARATE from shared_probe, which does
    continuous regression, not classification), predicting WHICH of the episode's K candidate
    entities (the SLOT index, tgt_slot) is being queried, from the arm's own native tap
    (adapter_arm's own INPUT shape -- state_summary_raw, not the post-adapter value_dim).
    Trained on FRESH episodes (a fresh generator stream), evaluated on a SEPARATE held-out
    fresh draw never used in training (this program's own standing train/eval-split
    discipline, sec 1.3.1.4's convention, applied to a real-backbone tap). PASS bar: > 3x
    episode-restricted chance (1/K), same convention as rung 1. The backbone is FROZEN
    throughout (torch.no_grad() around every tap call) -- only the classifier trains."""
    clf = ph.build_identity_classifier(TAP_DIM[arch], K).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model.eval()
    for _ in range(RUNG2_TRAIN_STEPS):
        b = sample_batch_rd(cfg_eval, RUNG2_TRAIN_BATCH, gen, hop_set, pools, device=device)
        with torch.no_grad():
            tap = AUDITED_TAP[arch](model, b["token_ids"], b["query_tokens"])
        logits = clf(tap)
        loss = F.cross_entropy(logits.reshape(-1, K), b["tgt_slot"].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    eval_gen = torch.Generator(device=device)
    eval_gen.manual_seed(seed + 999_983)   # large odd offset -- collision-safe from the training stream
    n_hit, n_tot = 0.0, 0
    with torch.no_grad():
        for _ in range(RUNG2_EVAL_BATCHES):
            b = sample_batch_rd(cfg_eval, RUNG2_TRAIN_BATCH, eval_gen, hop_set, pools, device=device)
            tap = AUDITED_TAP[arch](model, b["token_ids"], b["query_tokens"])
            pred_slot = clf(tap).argmax(dim=-1)
            n_hit += (pred_slot == b["tgt_slot"]).float().sum().item()
            n_tot += b["tgt_slot"].numel()
    model.train()
    acc = n_hit / max(1, n_tot)
    chance = 1.0 / K
    return {"accuracy": acc, "chance": chance, "pass_bar": ph.RUNG_CHANCE_MULT * chance,
            "passed": acc > ph.RUNG_CHANCE_MULT * chance, "K": K, "n_train_steps": RUNG2_TRAIN_STEPS}


def ladder_applies(task: str, role: str | None) -> bool:
    """R5-F3 (attack round 5, sec 1.23): the ladder is grammar_rd-specific -- task1_calib
    (both the primary AND stress_locate_only loads) and task2_calib; NEVER task3_calib (no
    query/answer-position structure) and NEVER a sweep cell (role=='sweep', sec 1.6's own
    'run for EVERY calibration cell' scope -- the ladder is a gate-1 CALIBRATION extension,
    not a per-sweep-cell cost)."""
    return task.startswith(("task1", "task2")) and role != "sweep"


def make_eval_episodes(cfg_eval: DeltaNetRDTaskConfig, pools, device: str, hop_set: tuple,
                       n_batches: int | None = None) -> list:
    """Fixed eval episodes: same EVAL_SEED for every arm/seed/cell of a given
    (task, hop_set), so cross-arm deltas compare identical episodes."""
    if n_batches is None:
        n_batches = EVAL_EPISODE_BATCHES     # late-bound so the selftest's global shrink applies
    gen = torch.Generator(device=device)
    gen.manual_seed(EVAL_SEED + 1009 * cfg_eval.K + 31 * sum(hop_set))
    return [sample_batch_rd(cfg_eval, GRAMMAR_BATCH, gen, hop_set, pools, device=device)
            for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Task 1/2 -- grammar joint-training cell
# ---------------------------------------------------------------------------

def train_grammar_cell(cell: dict, device: str, ckpt_dir: str | None,
                       steps_override: int | None = None, timing_only: bool = False,
                       dial_sentinel_path: str = DIAL_SENTINEL_PATH_DEFAULT) -> dict:
    task, arch = cell["task"], cell["arch"]
    K = cell.get("K")
    steps = steps_override if steps_override is not None else int(round(FULL_STEPS * cell.get("budget_frac", 1.0)))
    pools = get_pools(device)
    vocab_total = pools.vocab_size_total                          # DEPLOY-PIN-3
    cfg_train = task_cfg(task, K, n_query=N_QUERY_TRAIN)
    cfg_eval = task_cfg(task, K, n_query=None)                    # full-K queries at eval
    K_eff = K if K is not None else cfg_eval.K                    # task2 leaves cell["K"] unset; task_cfg pins 32
    run_ladder = ladder_applies(task, cell.get("role"))            # sec 1.23 item 3 / R5-F3 scope
    is_full_cell = run_ladder and cell.get("role") == "primary"    # sec 1.23 item 4 gate-1b scope

    model = build_arm_model(arch, vocab_total, cell["seed"], device)
    rig = ProbeRig(arch, vocab_total, device)
    tf32_state = set_and_log_tf32()
    opt = torch.optim.AdamW(list(model.parameters()) + rig.parameters(),
                            lr=cell["lr"], weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device=device)
    gen.manual_seed(cell["seed"])

    eval_train_hops = None if timing_only else make_eval_episodes(cfg_eval, pools, device, tuple(cfg_train.H_train))
    eval_heldout = (make_eval_episodes(cfg_eval, pools, device, tuple(cfg_train.H_test))
                    if (task.startswith("task2") and not timing_only) else None)
    capped_mask_fn = (lambda T: sink_fifo_mask(T, cap_length=cap_length_tokens(2, 2, 256),
                                               k_sink=K_SINK, device=device))

    curve, grad_ratio = [], None
    loss_first, recent_losses = None, []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = get_lr(step, cell["lr"], WARMUP_STEPS, steps)
        batch = sample_batch_rd(cfg_train, GRAMMAR_BATCH, gen, tuple(cfg_train.H_train), pools,
                                device=device)
        if step == 1 and not timing_only:
            assert_fused_tap_matches_audited(arch, model, batch["token_ids"], batch["query_tokens"], pools)
        logits, tap, answer_logits = fused_logits_and_tap(arch, model, batch["token_ids"],
                                                           batch["query_tokens"], pools)
        loss_ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                  batch["token_ids"][:, 1:].reshape(-1))
        loss_ce_answer = F.cross_entropy(answer_logits.reshape(-1, answer_logits.shape[-1]),
                                         answer_token_ids(batch).reshape(-1))
        pred, target = rig.pred(tap), probe_targets(rig, batch)
        loss = ph.joint_loss(loss_ce, pred, target, aux_weight=AUX_WEIGHT,
                             loss_ce_answer=loss_ce_answer, ce_answer_weight=CE_ANSWER_WEIGHT)

        if step == GRAD_RATIO_STEP and not timing_only and cell.get("role") != "sweep":
            # AUD2-F4 fix (sec 1.24 pre-launch build-fix, Stage-D guard): the dial only ever
            # fires for CALIBRATION-role cells (every role sweep_cells() emits is "sweep";
            # every calibration role -- "primary", "stress_locate_only", "reused_not_relaunched",
            # "lr_grid_*", "pilot", "selftest" -- is calibration-only by construction). This is
            # STRUCTURAL (reads the cell dict already threaded through this call), not env-based:
            # the audit's alternative (unset/reset H2H_DIAL_ROUND before Stage D in the chain
            # script) depends on the chain script remembering to do so at exactly the right
            # point, is invisible to a cell run outside the chain (e.g. --run-cell by hand), and
            # a stale/leaked env var would silently re-arm the dial for a noisy sweep-cell
            # step-500 ratio -> spurious DialExhausted-abort (the exact failure AUD2-F4(ii)
            # flags). Gating on cell["role"] instead cannot leak across stages or invocations.
            #
            # sec 1.3.1.3 / sec 1.23 build-fix item 2: THREE isolated backward passes (each
            # loss ALONE) measuring backbone grad norms -- the shared backbone parameters are
            # where the three losses actually MEET (the CE losses structurally never touch the
            # tap/answer_logits tensors of the OTHER terms directly, so a literal at-the-tap
            # cross-norm is identically zero for every arm). Measured + evaluated via
            # evaluate_three_loss_dial (pure function, unit-tested in mode_selftest); the
            # revision itself stays the coordinator's decision at margin freeze UNLESS R5-F1's
            # round cap has been reached AND a trigger still fires, in which case this function
            # hard-stops (DialExhaustedError) rather than merely reporting.
            params = [p for p in model.parameters() if p.requires_grad]
            g_ce = torch.autograd.grad(loss_ce, params, retain_graph=True, allow_unused=True)
            g_ce_answer = torch.autograd.grad(CE_ANSWER_WEIGHT * loss_ce_answer, params,
                                              retain_graph=True, allow_unused=True)
            g_aux = torch.autograd.grad(AUX_WEIGHT * ph.probe_aux_loss(pred, target), params,
                                        retain_graph=True, allow_unused=True)
            n_ce = torch.sqrt(sum((g ** 2).sum() for g in g_ce if g is not None)).item()
            n_ce_answer_l = [g for g in g_ce_answer if g is not None]
            n_ce_answer = (torch.sqrt(sum((g ** 2).sum() for g in n_ce_answer_l)).item()
                          if n_ce_answer_l else 0.0)
            n_aux_l = [g for g in g_aux if g is not None]
            n_aux = torch.sqrt(sum((g ** 2).sum() for g in n_aux_l)).item() if n_aux_l else 0.0

            dial = evaluate_three_loss_dial(n_ce, n_aux, n_ce_answer, AUX_WEIGHT, CE_ANSWER_WEIGHT,
                                            current_dial_round())
            grad_ratio = {"step": step, **dial}
            if dial["status"] == "DIAL_EXHAUSTED":
                _write_dial_exhausted_sentinel(dial_sentinel_path, cell["name"], dial)
                raise DialExhaustedError(dial["message"])

        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in list(model.parameters()) + rig.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + rig.parameters(), 1.0)
            opt.step()
        if loss_first is None:
            loss_first = loss.item()
        recent_losses = (recent_losses + [loss.item()])[-5:]

        if not timing_only and (step % GRAMMAR_EVAL_EVERY == 0 or step == steps):
            rf_tr, cos_tr = eval_recovered_frac(arch, model, rig, eval_train_hops)
            point = {"step": step, "train_loss": loss.item(),
                     "recovered_frac_train_hops": rf_tr,
                     "probe_cos_mean_train_hops": cos_tr}
            if eval_heldout is not None:
                rf_ho, cos_ho = eval_recovered_frac(arch, model, rig, eval_heldout)
                point["recovered_frac_heldout_hops"] = rf_ho
                point["probe_cos_mean_heldout_hops"] = cos_ho
            if arch == "transformer" and task.startswith("task1"):
                # sec 1.6 item C: the Transformer's own capped-cache (b-primary) behavior at this
                # K/d, report-only at calibration (M=2, the smallest floor-eligible point).
                rf_m2, cos_m2 = eval_recovered_frac(arch, model, rig, eval_train_hops,
                                                    attn_mask_fn=capped_mask_fn)
                point["recovered_frac_capped_M2"] = rf_m2
                point["probe_cos_mean_capped_M2"] = cos_m2
            if run_ladder:
                # sec 1.7 gate 1a (sec 1.23 item 3): rung 1 (K-restricted top-1) + the
                # membership-oracle tell, over the SAME fixed train-hops eval set rf@0.9
                # (rung 3) already uses. Rung 2 is a separate, one-shot post-training fit
                # (below) -- sec 1.6's own cost accounting, not repeated every eval step.
                rung1 = eval_diagnostic_rung1_and_tell(arch, model, rig, eval_train_hops, pools)
                point["rung1_accuracy"] = rung1["accuracy"]
                point["rung1_chance"] = rung1["chance"]
                point["rung1_passed"] = rung1["passed"]
                point["cos_pred_episode_mean"] = rung1["cos_pred_episode_mean"]
            curve.append(point)
            print(f"  [{cell['name']}] step {step}/{steps} loss {loss.item():.4f} "
                  f"rf_train {point['recovered_frac_train_hops']:.4f} "
                  f"cos_train {point['probe_cos_mean_train_hops']:.4f}", flush=True)

    rung2 = None
    if run_ladder and not timing_only:
        # sec 1.7 gate 1a rung 2 (sec 1.23 item 3): ONE fit at the end of training, on the
        # FINAL trained (now-frozen) backbone -- sec 1.6's own "one small extra
        # linear-classifier fit/cell" cost accounting.
        rung2 = fit_rung2_identity_classifier(arch, model, cfg_eval, tuple(cfg_train.H_train),
                                              pools, device, K_eff, seed=cell["seed"] + 7_919)

    wall_s = time.time() - t0
    result = {"arch": arch, "task": task, "seed_idx": cell.get("seed_idx", 0),
              "seed": cell["seed"], "K": K, "role": cell.get("role"), "lr": cell["lr"],
              "step_count": steps, "wall_s": wall_s, "s_per_step": wall_s / max(1, steps),
              "n_params": sum(p.numel() for p in model.parameters()),
              "vocab_size_total": vocab_total, "n_query_train": N_QUERY_TRAIN,
              "aux_weight": AUX_WEIGHT, "ce_answer_weight": CE_ANSWER_WEIGHT, "tf32": tf32_state,
              "loss_first": loss_first, "loss_final_mean5": sum(recent_losses) / max(1, len(recent_losses)),
              "grad_ratio_at_tap_step500": grad_ratio, "curve": curve,
              "rung2_identity_classifier": rung2}
    if curve:
        result["final_recovered_frac_train_hops"] = curve[-1]["recovered_frac_train_hops"]
        result["final_probe_cos_mean_train_hops"] = curve[-1]["probe_cos_mean_train_hops"]
        result["final_metric"] = curve[-1].get("recovered_frac_heldout_hops",
                                               curve[-1]["recovered_frac_train_hops"])
        if "recovered_frac_heldout_hops" in curve[-1]:
            result["final_recovered_frac_heldout_hops"] = curve[-1]["recovered_frac_heldout_hops"]
            result["final_probe_cos_mean_heldout_hops"] = curve[-1]["probe_cos_mean_heldout_hops"]
        if run_ladder:
            result["final_rung1_accuracy"] = curve[-1].get("rung1_accuracy")
            result["final_cos_pred_episode_mean"] = curve[-1].get("cos_pred_episode_mean")
    else:
        result["final_metric"] = float("nan")

    if is_full_cell and curve:
        # sec 1.7 gate 1b (sec 1.23 build-fix item 4): task1_calib/task2_calib FULL cells
        # (role=='primary') ONLY -- task1_stress stays exempt (run_ladder True, is_full_cell
        # False). Uses final_metric (heldout-preferring, task2's own convention) for rf@0.9.
        result["gate1_band"] = check_gate1_full_cell_band(
            result["final_rung1_accuracy"], K_eff, result["final_metric"])

    if ckpt_dir and not timing_only:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, cell["name"] + ".pt")
        torch.save({"arch": arch, "task": task, "K": K, "vocab_size_total": vocab_total,
                    "model": model.state_dict(), "rig": rig.state_dict(), "cell": cell},
                   ckpt_path)
        result["ckpt_path"] = ckpt_path
    return result


# ---------------------------------------------------------------------------
# Task 3 -- arch-generic LM cell (mirrors lm_pretrain_rd.train()'s regime)
# ---------------------------------------------------------------------------

def train_lm_cell(cell: dict, device: str, ckpt_dir: str | None,
                  steps_override: int | None = None, timing_only: bool = False,
                  data_dir: str = DEFAULT_DATA_DIR) -> dict:
    arch = cell["arch"]
    corpus = cell.get("corpus", TASK3_CORPUS)
    steps = steps_override if steps_override is not None else int(round(FULL_STEPS * cell.get("budget_frac", 1.0)))
    train_toks, val_toks, _meta, train_offs, val_offs = load_corpus(data_dir, corpus, device)
    other = OTHER_CORPUS[corpus]
    o_train, o_val, _om, _oto, o_voffs = load_corpus(data_dir, other, device)
    del o_train, _oto

    model = build_arm_model(arch, VOCAB_BASE, cell["seed"], device)
    tf32_state = set_and_log_tf32()
    opt = torch.optim.AdamW(model.parameters(), lr=cell["lr"], weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device=device)
    gen.manual_seed(cell["seed"])

    @torch.no_grad()
    def _val(tokens, corpus_name):
        # FIX-1 discipline: eval windows drawn from corpus_fixed_seed, identical across seeds/arms.
        eg = torch.Generator(device=device)
        eg.manual_seed(corpus_fixed_seed(corpus_name))
        model.eval()
        tot = 0.0
        for _ in range(LM_EVAL_BATCHES):
            x, y = get_batch(tokens, LM_EVAL_BATCH_SIZE, LM_SEQ_LEN, eg)
            lg = model(x)
            tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1)).item()
        model.train()
        return tot / LM_EVAL_BATCHES

    init_val = None if timing_only else _val(val_toks, corpus)
    curve, n_skipped = [], 0
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = get_lr(step, cell["lr"], WARMUP_STEPS, steps)
        x, y = get_batch(train_toks, LM_BATCH, LM_SEQ_LEN, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1
        if not timing_only and (step % LM_EVAL_EVERY == 0 or step == steps):
            point = {"step": step, "train_loss": loss.item(), "val_loss_own": _val(val_toks, corpus)}
            curve.append(point)
            print(f"  [{cell['name']}] step {step}/{steps} train {loss.item():.4f} "
                  f"val_own {point['val_loss_own']:.4f}", flush=True)

    wall_s = time.time() - t0
    result = {"arch": arch, "task": cell["task"], "seed_idx": cell.get("seed_idx", 0),
              "seed": cell["seed"], "role": cell.get("role"), "lr": cell["lr"], "corpus": corpus,
              "step_count": steps, "wall_s": wall_s, "s_per_step": wall_s / max(1, steps),
              "n_params": sum(p.numel() for p in model.parameters()),
              "skip_rate": n_skipped / max(1, steps), "tf32": tf32_state,
              "init_val_loss_own": init_val, "curve": curve}
    if curve:
        result["final_val_loss_own"] = curve[-1]["val_loss_own"]
        result["final_val_loss_other"] = _val(o_val, other)
        result["final_metric"] = result["final_val_loss_own"]
        del o_val, o_voffs
    else:
        result["final_metric"] = float("nan")
    if ckpt_dir and not timing_only:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, cell["name"] + ".pt")
        torch.save({"arch": arch, "task": cell["task"], "corpus": corpus,
                    "vocab_size_total": VOCAB_BASE, "model": model.state_dict(), "cell": cell},
                   ckpt_path)
        result["ckpt_path"] = ckpt_path
    return result


def run_one_cell(cell: dict, device: str, ckpt_dir: str | None, **kw) -> dict:
    if cell["task"].startswith("task3"):
        return train_lm_cell(cell, device, ckpt_dir, **kw)
    return train_grammar_cell(cell, device, ckpt_dir, **kw)


# ---------------------------------------------------------------------------
# Cell naming / manifests (single source of truth for the chain script)
# ---------------------------------------------------------------------------

AUX_REV_SUFFIX = "_auxrev2"     # aux_weight=2.0 re-run naming (task-1/2 cells only): resume-safety
                                # re-runs the probe cells WITHOUT clobbering the failed aux_weight=0.1
                                # artifacts (archived record); task-3 cells have no probe/aux term, so
                                # their names -- and their PASSED band results -- stand unchanged.


def calibration_cells() -> list[dict]:
    cells = []
    for c in build_full_calibration_manifest():
        if c["role"] == "reused_not_relaunched":
            continue    # sec 1.6 item E: contender task3 = FROZEN_BIAS rung-1, never re-launched
        name = f"h2h_calib_{c['arch']}_{c['task']}_{c['role']}"
        if "K" in c:
            name += f"_K{c['K']}"
        if not c["task"].startswith("task3"):
            name += AUX_REV_SUFFIX
        c = {**c, "name": name, "seed_idx": 0}
        cells.append(c)
    names = [c["name"] for c in cells]
    assert len(names) == len(set(names)) == 13, f"expected 13 unique launchable calibration cells, got {names}"
    return cells


def sweep_cells() -> list[dict]:
    out = []
    for c in build_27_cell_manifest():
        c = dict(c)
        c["budget_frac"], c["role"], c["lr"] = 1.0, "sweep", 3e-4
        # M3: every arm reuses the contender's pinned LR by default; the Transformer's task-3 LR
        # is REVISED at margin freeze from the calibration LR grid (the chain blocks on
        # MARGINS_FROZEN.token before any sweep cell launches -- coordinator writes the chosen LR
        # into that token file; see apply_margin_freeze_overrides below).
        if c["task"] == "task1_sweep":
            c["K"] = 32
        out.append(c)
    return out


def apply_margin_freeze_overrides(cells: list[dict], token_path: str) -> list[dict]:
    """MARGINS_FROZEN.token may carry {'transformer_task3_lr': <float>} --
    the coordinator's LR-grid decision applied to the sweep manifest. Any
    other overrides are deliberately NOT honored (no silent re-scoping)."""
    try:
        with open(token_path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return cells
    lr = doc.get("transformer_task3_lr")
    if lr is not None:
        for c in cells:
            if c["arch"] == "transformer" and c["task"] == "task3_sweep":
                c["lr"] = float(lr)
    return cells


# ---------------------------------------------------------------------------
# Axis 2 -- capped-inference eval pass (M-sweep pilot + 90-pass fan-out)
# ---------------------------------------------------------------------------

def extend_context_to_horizon(token_ids: torch.Tensor, horizon_tokens: int, buffer_id: int):
    """DEPLOY-PIN-4: right-pad the bind sequence with BUFFER filler to the
    absolute horizon length."""
    B, T = token_ids.shape
    assert horizon_tokens >= T, f"horizon {horizon_tokens} < bind length {T}"
    pad = torch.full((B, horizon_tokens - T), buffer_id, dtype=token_ids.dtype,
                     device=token_ids.device)
    return torch.cat([token_ids, pad], dim=1)


def load_h2h_checkpoint(path: str, device: str):
    doc = torch.load(path, map_location=device)
    arch, vocab = doc["arch"], doc["vocab_size_total"]
    model = build_arm_model(arch, vocab, seed=0, device=device)
    model.load_state_dict(doc["model"])
    model.eval()
    rig = None
    if "rig" in doc:
        rig = ProbeRig(arch, vocab, device)
        rig.load_state_dict(doc["rig"])
    return model, rig, doc


def capped_eval_pass(model, rig: ProbeRig, arch: str, task: str, K: int | None, M: int | None,
                     horizon: str, pools, device: str) -> dict:
    """ONE axis-2 inference pass: fixed eval episodes at the task's own eval
    hop set, context extended to the horizon; transformer capped via
    sink+FIFO at cap_length(M) (M=None -> uncapped: contender reference /
    b-control)."""
    cfg_eval = task_cfg(task, K, n_query=None)
    hop_set = tuple(cfg_eval.H_test) if task.startswith("task2") else tuple(cfg_eval.H_train)
    batches = make_eval_episodes(cfg_eval, pools, device, hop_set)
    H = HORIZON_TOKENS[horizon]
    mask_fn = None
    if M is not None:
        cl = cap_length_tokens(M, TRANSFORMER_KW["n_layers"], TRANSFORMER_KW["d_model"])
        mask_fn = lambda T: sink_fifo_mask(T, cap_length=cl, k_sink=K_SINK, device=device)  # noqa: E731

    def _episode_chunks(b):
        # VRAM guard (transformer only): see _transformer_episode_chunks -- the recurrent
        # arms have no per-query replication or T^2 term and run unchunked.
        if arch != "transformer":
            yield b
            return
        yield from _transformer_episode_chunks(b, H + b["query_tokens"].shape[-1])

    n_hit, n_tot = 0.0, 0
    with torch.no_grad():
        for b_full in batches:
            for b in _episode_chunks(b_full):
                ctx = extend_context_to_horizon(b["token_ids"], H, pools.buffer_id)
                if arch == "transformer":
                    tap = ph.transformer_native_tap(model, ctx, b["query_tokens"], attn_mask_fn=mask_fn)
                else:
                    tap = AUDITED_TAP[arch](model, ctx, b["query_tokens"])
                frac = ph.cosine_recovery_frac(rig.pred(tap), probe_targets(rig, b)).item()
                n_q = b["query_tokens"].shape[0] * b["query_tokens"].shape[1]
                n_hit += frac * n_q
                n_tot += n_q
    return {"recovered_frac": n_hit / max(1, n_tot), "horizon_tokens": H,
            "cap_length": (None if M is None else
                           cap_length_tokens(M, TRANSFORMER_KW["n_layers"], TRANSFORMER_KW["d_model"])),
            "n_episodes": len(batches) * GRAMMAR_BATCH}


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def _atomic_dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def mode_run_cell(args) -> int:
    require_launch_tokens(args.gates_dir)
    cells = {c["name"]: c for c in (calibration_cells() if args.set == "calibration" else sweep_cells())}
    assert args.run_cell in cells, f"unknown cell {args.run_cell!r} in set {args.set}"
    cell = cells[args.run_cell]
    if args.set == "sweep":
        require_margins_frozen(args.margins_token)          # audit MAJOR-2: code-level release gate
        cell = apply_margin_freeze_overrides([cell], args.margins_token)[0]
    if is_valid_result(args.out):
        print(f"SKIP (already valid): {args.out}")
        return 0
    result = run_one_cell(cell, args.device, args.ckpt_dir)
    result = {**cell, **result}
    assert all(k in result for k in REQUIRED_RESULT_KEYS)
    _atomic_dump(args.out, result)
    print(f"CELL COMPLETE: {args.run_cell} final_metric={result['final_metric']} "
          f"wall_s={result['wall_s']:.1f}")
    return 0


def mode_pilot_pair(args) -> int:
    """sec 1.7 gate 2: ONE real short run per arch x task at the pair's own
    primary config, measured s/step -> projected s/full-cell."""
    require_launch_tokens(args.gates_dir)
    arch, task = args.pilot_pair
    base = {"arch": arch, "task": f"{task}_calib", "role": "pilot", "budget_frac": 1.0,
            "seed": 12345, "lr": 3e-4, "name": f"pilot_{arch}_{task}", "seed_idx": 0}
    if task == "task1":
        base["K"] = 32
    warmup, timed = 10, 40
    t_all = run_one_cell(dict(base), args.device, ckpt_dir=None,
                         steps_override=warmup + timed, timing_only=True)
    # First-steps include kernel autotune/compile warmup; measure a second, warm run for the rate.
    t_warm = run_one_cell(dict(base), args.device, ckpt_dir=None,
                          steps_override=timed, timing_only=True)
    s_per_step = t_warm["wall_s"] / timed
    doc = {"arch": arch, "task": task, "s_per_step_warm": s_per_step,
           "cold_wall_s": t_all["wall_s"], "projected_s_per_full_cell": s_per_step * FULL_STEPS,
           "projected_gpu_h_per_full_cell": s_per_step * FULL_STEPS / 3600.0}
    _atomic_dump(args.out, doc)
    print(f"PILOT {arch}x{task}: {s_per_step:.4f} s/step -> "
          f"{doc['projected_gpu_h_per_full_cell']:.4f} GPU-h/full-cell")
    return 0


# Pair shares (sec 1.6 arithmetic): full-cell-equivalents this pair owes across
# calibration + sweep, x 0.2524 GPU-h x the 10x bracket = the pair's gate ceiling.
PAIR_CELL_EQUIVS = {
    ("contender", "task1"): 1 + 0.25 + 3, ("ablation", "task1"): 1 + 0.25 + 3,
    ("transformer", "task1"): 1 + 0.25 + 3,
    ("contender", "task2"): 1 + 3, ("ablation", "task2"): 1 + 3, ("transformer", "task2"): 1 + 3,
    ("contender", "task3"): 3, ("ablation", "task3"): 1 + 3, ("transformer", "task3"): 0.75 + 3,
}
PLANNING_RATE_GPU_H = 0.2524
BRACKET_MULT = 10.0


def mode_gate_pilots(args) -> int:
    rows, ok_all = [], True
    for (arch, task), equivs in PAIR_CELL_EQUIVS.items():
        path = os.path.join(args.pilots_dir, f"pilot_{arch}_{task}.json")
        with open(path) as f:
            p = json.load(f)
        projected = p["projected_gpu_h_per_full_cell"] * equivs
        ceiling = equivs * PLANNING_RATE_GPU_H * BRACKET_MULT
        ok = projected <= ceiling
        ok_all = ok_all and ok
        rows.append({"arch": arch, "task": task, "cell_equivs": equivs,
                     "projected_gpu_h": projected, "share_ceiling_gpu_h": ceiling, "ok": ok})
        print(f"[pilot gate] {arch}x{task}: projected {projected:.3f} GPU-h vs "
              f"share ceiling {ceiling:.3f} -- {'OK' if ok else 'ABORT'}")
    total = sum(r["projected_gpu_h"] for r in rows)
    doc = {"rows": rows, "total_projected_train_gpu_h": total, "ok": ok_all,
           "planning_rate_gpu_h": PLANNING_RATE_GPU_H, "bracket_mult": BRACKET_MULT}
    _atomic_dump(args.out, doc)
    if not ok_all:
        print("PILOT GATE: ABORT (sec 1.7 gate 2 mechanical enforced abort)", file=sys.stderr)
        return 1
    print(f"PILOT GATE: OK (total projected training {total:.3f} GPU-h)")
    return 0


def mode_msweep_pilot(args) -> int:
    require_launch_tokens(args.gates_dir)
    device = args.device
    pools = get_pools(device)

    def _loader():
        model, rig, doc = load_h2h_checkpoint(args.ckpt, device)
        assert doc["arch"] == "transformer", "M-sweep pilot must run on a (b-primary) transformer ckpt"
        return (model, rig, doc)

    def _eval(ckpt, m, horizon):
        model, rig, doc = ckpt
        capped_eval_pass(model, rig, "transformer", doc["task"], doc.get("K"), m, horizon,
                         pools, device)

    pilot = run_msweep_timing_pilot(_eval, _loader)      # audited wrapper: residency-asserted
    gate = project_and_gate_msweep_fanout(pilot["mean_s_per_pass"],
                                          remaining_headroom_gpu_h=args.headroom_gpu_h)
    doc = {"pilot": pilot, "fanout_gate": gate, "ckpt": args.ckpt,
           "design_time_assumption_s_per_pass_REPLACED": 5.0}
    _atomic_dump(args.out, doc)
    print(f"M-SWEEP PILOT: measured {pilot['mean_s_per_pass']:.3f} s/pass "
          f"(design assumption 5.0 REPLACED); 90-pass projection "
          f"{gate['projected_gpu_h']:.4f} GPU-h ok={gate['ok']}")
    return 0 if gate["ok"] else 1


def mode_fanout_all(args) -> int:
    require_launch_tokens(args.gates_dir)
    require_margins_frozen(args.margins_token)               # audit MAJOR-2: stage-D-only mode
    device = args.device
    pools = get_pools(device)
    with open(args.ckpt_map) as f:
        ckpt_map = {tuple(k.split("|")): v for k, v in json.load(f).items()}

    def _loader(cell):
        model, rig, doc = load_h2h_checkpoint(ckpt_map[(cell["task"], str(cell["seed_idx"]))], device)
        assert doc["arch"] == "transformer", "fan-out checkpoints must be (b-primary) transformer cells"
        return (model, rig, doc)

    def _eval(ckpt, m, horizon):
        # signature pinned by run_pass_exception_isolated: eval_pass_fn(ckpt, M, horizon);
        # task/K come from the resident checkpoint's own doc.
        model, rig, doc = ckpt
        return capped_eval_pass(model, rig, "transformer", doc["task"], doc.get("K", 32),
                                m, horizon, pools, device)

    manifest = build_90_pass_manifest()
    summary = run_msweep_fanout(manifest, _eval, _loader, args.out_dir)
    _atomic_dump(os.path.join(args.out_dir, "FANOUT_SUMMARY.json"), summary)
    print(f"FANOUT: {summary}")
    return 0 if summary.get("failed", 0) == 0 else 1


def mode_horizon_ref(args) -> int:
    """Contender (uncapped, constant-state) reference passes per (task, seed,
    horizon) -- axis 2's own comparison side (18 passes)."""
    require_launch_tokens(args.gates_dir)
    require_margins_frozen(args.margins_token)               # audit MAJOR-2: stage-D-only mode
    device = args.device
    pools = get_pools(device)
    with open(args.ckpt_map) as f:
        ckpt_map = {tuple(k.split("|")): v for k, v in json.load(f).items()}
    out = {}
    for (task, seed_idx), path in sorted(ckpt_map.items()):
        model, rig, doc = load_h2h_checkpoint(path, device)
        for horizon in HORIZON_TOKENS:
            r = capped_eval_pass(model, rig, doc["arch"], task, doc.get("K", 32), None, horizon,
                                 pools, device)
            out[f"{task}|s{seed_idx}|{horizon}"] = r
            print(f"HORIZON-REF {doc['arch']} {task} s{seed_idx} {horizon}: "
                  f"rf={r['recovered_frac']:.4f}")
    _atomic_dump(args.out, out)
    return 0


def mode_calibration_report(args) -> int:
    calib_dir = os.path.join(args.res_dir, "calib")
    pilots_dir = os.path.join(args.res_dir, "pilots")
    cells = {}
    for c in calibration_cells():
        path = os.path.join(calib_dir, c["name"] + ".json")
        with open(path) as f:
            cells[c["name"]] = json.load(f)
    pilots = {}
    for fn in sorted(os.listdir(pilots_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(pilots_dir, fn)) as f:
                pilots[fn] = json.load(f)
    grad_ratios = {n: r.get("grad_ratio_at_tap_step500") for n, r in cells.items()
                   if r.get("grad_ratio_at_tap_step500") is not None}
    report = {
        "wave": "HEAD_TO_HEAD rung-1 (sec 1.19 items 3-4 complete)",
        "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_cells": cells,
        "reused_not_relaunched": {"contender/task3": "FROZEN_BIAS_LM_DESIGN.md rung-1 "
                                  "(Arm 2 per_token lam=0.58): val 2.1184/4.3426 openr1/wikitext "
                                  "(sec 12.1 verified-numbers table)"},
        "pilots": pilots,
        "aux_grad_ratio_table_sec_1_3_1_3": grad_ratios,
        "deploy_stage_pins_for_review": [
            "DEPLOY-PIN-1: n_query_train=8 (symmetric, all arms; full-K at eval)",
            "DEPLOY-PIN-2: task3 corpus=openr1-mix-ext (both corpora's val logged)",
            "DEPLOY-PIN-3: tasks 1/2 vocab=50259 (reserved rows learned, all arms)",
            "DEPLOY-PIN-4: horizon filler=BUFFER right-padding to absolute horizon",
            "DEPLOY-PIN-5: task2 K=32 H_train=(1,2) H_test=(3,4)",
            "transformer task3 sweep LR awaits margin-freeze override "
            "(transformer_task3_lr in MARGINS_FROZEN.token; default 3e-4)",
        ],
        "next_gate": "MARGINS_FROZEN.token (coordinator-only write) releases the 27-cell sweep",
    }
    _atomic_dump(args.out, report)
    print(f"CALIBRATION_COMPLETE report written: {args.out}")
    return 0


# ---------------------------------------------------------------------------
# CPU selftest (forced stub) -- logic-only; every real-kernel claim is box-side
# ---------------------------------------------------------------------------

def mode_selftest() -> int:
    import subprocess
    import tempfile
    global evaluate_three_loss_dial   # declared up front: selftest 9-10 CALL it, selftest 11
                                       # monkeypatches it -- Python requires 'global' to precede
                                       # any use of a name a function later assigns to.
    failures = []

    def rep(item, ok, detail=""):
        print(f"[{item}] {'PASS' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}", flush=True)
        if not ok:
            failures.append(item)

    # 1. manifests: 13 launchable calibration cells, 27 sweep cells, unique names.
    cal, sw = calibration_cells(), sweep_cells()
    rep("selftest 1: manifests (13 calibration launchable, 27 sweep)",
        len(cal) == 13 and len(sw) == 27)

    # 2. task cfgs valid (task1 K=32/48, task2) -- __post_init__ guards run.
    ok2 = True
    for t, k in (("task1_calib", 32), ("task1_calib", 48), ("task2_calib", None)):
        cfg = task_cfg(t, k, n_query=N_QUERY_TRAIN)
        ok2 = ok2 and cfg.queries == N_QUERY_TRAIN
    rep("selftest 2: task cfgs construct with periodicity guards", ok2)

    # 3. launch-token refusal (negative, subprocess so SystemExit is a real exit code).
    with tempfile.TemporaryDirectory() as td:
        env = {k: v for k, v in os.environ.items()
               if k not in ("HEADTOHEAD_PI_SIGNOFF", "HEADTOHEAD_MATCH_GATE_SIGNOFF")}
        env["REASONING_LINK_FORCE_CPU_STUB"] = "1"
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--token-probe",
                            "--gates-dir", td], env=env, capture_output=True)
        rep("selftest 3: --token-probe REFUSES (exit 3) without env tokens + gate files",
            r.returncode == 3)
        # positive: env set AND token files present
        for fn in GATE_TOKEN_FILES:
            with open(os.path.join(td, fn), "w") as f:
                f.write("{}")
        env2 = dict(env, HEADTOHEAD_PI_SIGNOFF="1", HEADTOHEAD_MATCH_GATE_SIGNOFF="1")
        r2 = subprocess.run([sys.executable, os.path.abspath(__file__), "--token-probe",
                             "--gates-dir", td], env=env2, capture_output=True)
        rep("selftest 4: --token-probe passes with both env tokens + both gate files",
            r2.returncode == 0)
        # negative triple: each of the three requirements independently removed
        neg_ok = True
        for kill_env in ("HEADTOHEAD_PI_SIGNOFF", "HEADTOHEAD_MATCH_GATE_SIGNOFF"):
            e = {k: v for k, v in env2.items() if k != kill_env}
            neg_ok = neg_ok and subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--token-probe", "--gates-dir", td],
                env=e, capture_output=True).returncode == 3
        os.remove(os.path.join(td, GATE_TOKEN_FILES[0]))
        neg_ok = neg_ok and subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--token-probe", "--gates-dir", td],
            env=env2, capture_output=True).returncode == 3
        rep("selftest 5: negative-test triple (each token independently removed -> refusal)", neg_ok)

    # 6. grammar joint-training micro-run (ablation + transformer, CPU-provable arms), tiny steps.
    #    Exercises: episode gen, fused tap + step-0 audited-tap equivalence, THREE-term joint
    #    loss (sec 1.23 item 1), curve, grad-ratio dial skipped (step<500), the diagnostic
    #    ladder (sec 1.23 item 3: rung 1 + membership tell + rung 2, role != 'sweep' triggers
    #    ladder_applies), the gate-1b band (sec 1.23 item 4, role=='primary' only), result keys.
    for arch in ("ablation", "transformer"):
        cell = {"arch": arch, "task": "task1_calib", "role": "selftest", "budget_frac": 1.0,
                "seed": 7, "lr": 1e-3, "K": 8, "name": f"selftest_{arch}", "seed_idx": 0}
        global GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH
        saved = (GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH)
        GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH = 3, 1, 4
        try:
            r = train_grammar_cell(cell, "cpu", ckpt_dir=None, steps_override=3)
            ok = (all(k in r for k in REQUIRED_RESULT_KEYS) and len(r["curve"]) >= 1
                  and 0.0 <= r["final_recovered_frac_train_hops"] <= 1.0
                  and torch.isfinite(torch.tensor(r["loss_final_mean5"])).item()
                  and "rung1_accuracy" in r["curve"][-1] and "cos_pred_episode_mean" in r["curve"][-1]
                  and 0.0 <= r["final_rung1_accuracy"] <= 1.0
                  and r["rung2_identity_classifier"] is not None
                  and 0.0 <= r["rung2_identity_classifier"]["accuracy"] <= 1.0
                  and "gate1_band" not in r)   # role='selftest' != 'primary' -> NOT a full cell
        finally:
            GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH = saved
        rep(f"selftest 6({arch}): grammar micro-cell trains, taps match audited, three-term "
            f"loss, ladder rungs 1-2 + membership tell logged, gate1_band absent (non-primary "
            f"role), keys complete", ok)

    # 6b. gate1_band DOES appear for a role=='primary' FULL cell (ablation only, cost).
    cell_full = {"arch": "ablation", "task": "task1_calib", "role": "primary", "budget_frac": 1.0,
                "seed": 8, "lr": 1e-3, "K": 8, "name": "selftest_ablation_full", "seed_idx": 0}
    saved = (GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH)
    GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH = 3, 1, 4
    try:
        r_full = train_grammar_cell(cell_full, "cpu", ckpt_dir=None, steps_override=3)
        ok6b = ("gate1_band" in r_full and "within_band" in r_full["gate1_band"]
                and r_full["gate1_band"]["K"] == 8)
    finally:
        GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH = saved
    rep("selftest 6b: gate1_band IS populated for a role=='primary' FULL cell (sec 1.23 item 4)",
        ok6b)

    # 7. horizon extension: length + query-distance arithmetic.
    ids = torch.full((2, 20), 5, dtype=torch.int64)
    ext = extend_context_to_horizon(ids, 33, buffer_id=9)
    rep("selftest 7: extend_context_to_horizon pads with BUFFER to exact length",
        ext.shape == (2, 33) and (ext[:, 20:] == 9).all().item())

    # 8. margin-freeze override honored for transformer/task3 ONLY.
    with tempfile.TemporaryDirectory() as td:
        tok = os.path.join(td, "MARGINS_FROZEN.token")
        with open(tok, "w") as f:
            json.dump({"transformer_task3_lr": 1e-4}, f)
        cells2 = apply_margin_freeze_overrides(sweep_cells(), tok)
        t3 = [c for c in cells2 if c["arch"] == "transformer" and c["task"] == "task3_sweep"]
        others = [c for c in cells2 if not (c["arch"] == "transformer" and c["task"] == "task3_sweep")]
        rep("selftest 8: margin-freeze LR override hits transformer/task3 only",
            all(c["lr"] == 1e-4 for c in t3) and all(c["lr"] == 3e-4 for c in others))

    # 9. evaluate_three_loss_dial (sec 1.23 item 2): pure-function PASS / REVISE (single- and
    #    both-firing, larger-ratio-DEVIATION-first tie-break) -- synthetic norms, no GPU.
    d_pass = evaluate_three_loss_dial(1.0, 1.0, 1.0, aux_weight=2.0, ce_answer_weight=1.0, round_idx=3)
    d_revise_aux = evaluate_three_loss_dial(1.0, 0.01, 1.0, aux_weight=2.0, ce_answer_weight=1.0, round_idx=3)
    d_revise_ce_answer = evaluate_three_loss_dial(1.0, 1.0, 0.01, aux_weight=2.0, ce_answer_weight=1.0,
                                                  round_idx=3)
    # both fire; aux's deviation (ratio 1000) >> ce_answer's (ratio 50) -> aux revised, ce_answer deferred.
    d_both_fire = evaluate_three_loss_dial(1.0, 0.001, 0.02, aux_weight=2.0, ce_answer_weight=1.0,
                                           round_idx=3)
    ok9 = (d_pass["status"] == "PASS"
           and d_revise_aux["status"] == "REVISE" and d_revise_aux["revise"] == "aux_weight"
           and d_revise_aux["new_aux_weight"] > 2.0
           and d_revise_ce_answer["status"] == "REVISE" and d_revise_ce_answer["revise"] == "ce_answer_weight"
           and d_revise_ce_answer["new_ce_answer_weight"] > 1.0
           and d_both_fire["status"] == "REVISE" and d_both_fire["revise"] == "aux_weight"
           and d_both_fire["deferred"] == "ce_answer_weight")
    rep("selftest 9: evaluate_three_loss_dial PASS/REVISE + larger-ratio-deviation-first "
        "tie-break when both auxiliaries fire (synthetic norms)", ok9,
        f"pass={d_pass['status']} revise_aux={d_revise_aux['revise']} "
        f"revise_ce_answer={d_revise_ce_answer['revise']} both_fire_revise={d_both_fire['revise']} "
        f"both_fire_deferred={d_both_fire['deferred']}")

    # 10 (R5-F1 NEGATIVE TEST, run to completion): a trigger firing AT/PAST round_idx ==
    #     MAX_DIAL_ROUNDS must return DIAL_EXHAUSTED, NEVER a further REVISE.
    d_exhausted = evaluate_three_loss_dial(1.0, 0.001, 1.0, aux_weight=2.0, ce_answer_weight=1.0,
                                           round_idx=MAX_DIAL_ROUNDS)
    d_exhausted_over = evaluate_three_loss_dial(1.0, 0.001, 1.0, aux_weight=2.0, ce_answer_weight=1.0,
                                                round_idx=MAX_DIAL_ROUNDS + 1)
    d_not_yet_exhausted = evaluate_three_loss_dial(1.0, 0.001, 1.0, aux_weight=2.0, ce_answer_weight=1.0,
                                                   round_idx=MAX_DIAL_ROUNDS - 1)
    ok10 = (d_exhausted["status"] == "DIAL_EXHAUSTED" and "message" in d_exhausted
            and d_exhausted_over["status"] == "DIAL_EXHAUSTED"
            and d_not_yet_exhausted["status"] == "REVISE")
    rep("selftest 10 (R5-F1 NEGATIVE TEST): dial firing at round_idx>=MAX_DIAL_ROUNDS returns "
        "DIAL_EXHAUSTED (never a further REVISE); round_idx==MAX_DIAL_ROUNDS-1 still revises",
        ok10, f"status_at_cap={d_exhausted['status']} status_over_cap={d_exhausted_over['status']} "
        f"status_below_cap={d_not_yet_exhausted['status']}")

    # 11. DialExhaustedError WIRING (integration, not the pure logic re-tested above):
    #     monkeypatch evaluate_three_loss_dial to force DIAL_EXHAUSTED, confirm
    #     train_grammar_cell actually RAISES and writes the sentinel file.
    global GRAD_RATIO_STEP
    saved_grs, saved_dial_fn = GRAD_RATIO_STEP, evaluate_three_loss_dial
    GRAD_RATIO_STEP = 1

    def _force_dial_exhausted(*_a, **_kw):
        return {"status": "DIAL_EXHAUSTED", "message": "FORCED (selftest 11)", "round_idx": 99}

    evaluate_three_loss_dial = _force_dial_exhausted
    with tempfile.TemporaryDirectory() as td:
        sentinel = os.path.join(td, "DIAL_EXHAUSTED.token")
        cell = {"arch": "ablation", "task": "task1_calib", "role": "selftest", "budget_frac": 1.0,
                "seed": 9, "lr": 1e-3, "K": 8, "name": "selftest_dial_exhausted", "seed_idx": 0}
        saved = (GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH)
        GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH = 3, 1, 4
        raised = False
        try:
            train_grammar_cell(cell, "cpu", ckpt_dir=None, steps_override=2, dial_sentinel_path=sentinel)
        except DialExhaustedError:
            raised = True
        finally:
            GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH = saved
            GRAD_RATIO_STEP = saved_grs
            evaluate_three_loss_dial = saved_dial_fn
        sentinel_written = os.path.isfile(sentinel)
    ok11 = raised and sentinel_written
    rep("selftest 11: DialExhaustedError wiring -- a forced DIAL_EXHAUSTED verdict actually "
        "RAISES from train_grammar_cell and writes the sentinel file", ok11,
        f"raised={raised} sentinel_written={sentinel_written}")

    # 12. Blank-out (a) (sec 1.23 item 5a): in-place bind_tokens corruption AFTER caching S_T
    #     must not change the continuation's answer-position logits. CPU-PROVABLE IN FULL
    #     (unlike smoke_3/smoke_8's stub-vacuity limitation): the claim is purely structural --
    #     the continuation call has no argument, and therefore no path, back to bind_tokens --
    #     true identically whether S_T is the CPU stub's zero constant or a real nonzero state.
    ok12 = True
    for arch in ("contender", "ablation"):
        torch.manual_seed(51)
        vocab = 300
        m = (DeltaNetLM(vocab, d_model=32, d_state=64, n_layers=1, conv_size=4) if arch == "contender"
             else AblationLM(vocab, d_model=32, d_state=16, n_layers=1, conv_size=4))
        bind_tokens = torch.randint(0, vocab, (2, _MIN_KERNEL_T))
        query_tokens = torch.randint(0, vocab, (2, 3, 6))
        with torch.no_grad():
            _, final_states = m(bind_tokens, return_states=True)
            logits_before = _recurrent_continuation_answer_logits(arch, m, final_states, query_tokens,
                                                                   buffer_id=vocab - 1)
            bind_tokens.copy_(torch.randint_like(bind_tokens, 0, vocab))   # hard in-place corruption
            logits_after = _recurrent_continuation_answer_logits(arch, m, final_states, query_tokens,
                                                                  buffer_id=vocab - 1)
        identical = torch.equal(logits_before, logits_after)
        ok12 = ok12 and identical
        print(f"    arch={arch}: blank-out in-place identical={identical}", flush=True)
    rep("selftest 12 (blank-out 5a, CPU-PROVABLE IN FULL): in-place bind_tokens corruption "
        "after caching S_T leaves the continuation's answer logits bit-identical", ok12)

    # 13. Blank-out (b), R5-F4 companion (sec 1.23 item 5b): a FRESH model instance loaded with
    #     the SAME weights, given the SAME cached S_T and query tokens, must reproduce
    #     BIT-IDENTICAL continuation logits -- closes the hidden-module-cache channel the
    #     in-place test above cannot. CPU-provable for ablation (real recurrence, no stub
    #     disconnect) and for the contender's OWN stub-constant case (S_T=0 regardless of which
    #     instance produced it, so CPU proves determinism-given-equal-inputs); the REAL Triton
    #     kernel's own instance-level determinism is registered BOX-ONLY (mirrors
    #     probe_head_rd.py smoke_3/smoke_8's own box-only-registration discipline for the
    #     contender arm specifically -- see h2h_box_smoke_checklist.py's
    #     continuation_blankout_fresh_instance item).
    # AUD2-F3 fix (sec 1.24 pre-launch build-fix, defense-in-depth): the "fresh instance"
    # claim above is invisible to a shape/equality check alone -- a weakened mutation (e.g.
    # accidentally aliasing m2 = m1, or a copy that shares storage) would still pass the
    # logits-equality assertion trivially, since a genuinely fresh instance is not what's being
    # exercised. Guard it explicitly: m1/m2 must be distinct Python objects (id()) AND every
    # parameter tensor must be a distinct storage (data_ptr()) -- cheap, and closes mutation (d).
    ok13 = True
    for arch in ("contender", "ablation"):
        torch.manual_seed(52)
        vocab = 300
        m1 = (DeltaNetLM(vocab, d_model=32, d_state=64, n_layers=1, conv_size=4) if arch == "contender"
              else AblationLM(vocab, d_model=32, d_state=16, n_layers=1, conv_size=4))
        bind_tokens = torch.randint(0, vocab, (2, _MIN_KERNEL_T))
        query_tokens = torch.randint(0, vocab, (2, 3, 6))
        with torch.no_grad():
            _, final_states = m1(bind_tokens, return_states=True)
            logits1 = _recurrent_continuation_answer_logits(arch, m1, final_states, query_tokens,
                                                             buffer_id=vocab - 1)
            m2 = (DeltaNetLM(vocab, d_model=32, d_state=64, n_layers=1, conv_size=4) if arch == "contender"
                  else AblationLM(vocab, d_model=32, d_state=16, n_layers=1, conv_size=4))
            m2.load_state_dict(m1.state_dict())
            logits2 = _recurrent_continuation_answer_logits(arch, m2, final_states, query_tokens,
                                                             buffer_id=vocab - 1)
        identical = torch.equal(logits1, logits2)
        distinct_objects = (m1 is not m2) and (id(m1) != id(m2))
        distinct_params = all(p1.data_ptr() != p2.data_ptr()
                              for p1, p2 in zip(m1.parameters(), m2.parameters()))
        ok13 = ok13 and identical and distinct_objects and distinct_params
        print(f"    arch={arch}: fresh-instance blank-out identical={identical} "
              f"distinct_objects={distinct_objects} distinct_params={distinct_params}", flush=True)
    rep("selftest 13 (blank-out 5b, R5-F4 companion, CPU-provable for ablation + the "
        "contender's stub-constant case; real-kernel residual BOX-ONLY; AUD2-F3 hardened with "
        "id()/data_ptr() distinctness so the test cannot be trivially satisfied by an aliased "
        "instance): a fresh model instance loaded with the same weights reproduces "
        "BIT-IDENTICAL continuation logits from the same cached S_T", ok13)

    # 14. Padding causal-inertness (supports item 5a/5b's own soundness): the FILLER VALUE used
    #     to satisfy _MIN_KERNEL_T must not affect the answer-position logit at all (causal
    #     masking guarantees positions after <Q> are invisible to it) -- two different filler
    #     ids must reproduce identical logits.
    ok14 = True
    for arch in ("contender", "ablation"):
        torch.manual_seed(53)
        vocab = 300
        m = (DeltaNetLM(vocab, d_model=32, d_state=64, n_layers=1, conv_size=4) if arch == "contender"
             else AblationLM(vocab, d_model=32, d_state=16, n_layers=1, conv_size=4))
        bind_tokens = torch.randint(0, vocab, (2, _MIN_KERNEL_T))
        query_tokens = torch.randint(0, vocab, (2, 3, 6))
        with torch.no_grad():
            _, final_states = m(bind_tokens, return_states=True)
            logits_pad_a = _recurrent_continuation_answer_logits(arch, m, final_states, query_tokens,
                                                                  buffer_id=1)
            logits_pad_b = _recurrent_continuation_answer_logits(arch, m, final_states, query_tokens,
                                                                  buffer_id=vocab - 1)
        identical = torch.equal(logits_pad_a, logits_pad_b)
        ok14 = ok14 and identical
    rep("selftest 14: continuation padding VALUE is causally inert on the answer-position "
        "logit (two different filler ids reproduce identical logits)", ok14)

    # 15 (item 1's own CPU-stub caveat, mirroring probe_head_rd.py smoke_3/smoke_8's exact
    #     two-half discipline): PLUMBING is CPU-provable (answer_logits has the right shape, is
    #     finite, and genuinely changes with the QUERY); the VALUE-level claim -- that
    #     answer_logits carries real RECALL signal from a nonzero S_T -- is NOT CPU-provable
    #     for the contender: the stub's all-zero final_state makes its continuation
    #     data-INDEPENDENT of bind_tokens (checked directly: two DIFFERENT bind_tokens draws
    #     give IDENTICAL contender answer_logits under the stub), registered BOX-ONLY (see
    #     h2h_box_smoke_checklist.py real_kernel_fwd_bwd_grad). The ablation has no such stub
    #     disconnect (real recurrence even on CPU) and is exercised as the contrast case.
    torch.manual_seed(61)
    vocab = 300
    m_con = DeltaNetLM(vocab, d_model=32, d_state=64, n_layers=1, conv_size=4)
    m_abl = AblationLM(vocab, d_model=32, d_state=16, n_layers=1, conv_size=4)
    bind_a = torch.randint(0, vocab, (2, _MIN_KERNEL_T))
    bind_b = torch.randint(0, vocab, (2, _MIN_KERNEL_T))
    query_tokens = torch.randint(0, vocab, (2, 3, 6))
    query_tokens_b = torch.randint(0, vocab, (2, 3, 6))
    with torch.no_grad():
        _, fs_con_a = m_con(bind_a, return_states=True)
        _, fs_con_b = m_con(bind_b, return_states=True)
        logits_con_bind_a = _recurrent_continuation_answer_logits(
            "contender", m_con, fs_con_a, query_tokens, buffer_id=vocab - 1)
        logits_con_bind_b = _recurrent_continuation_answer_logits(
            "contender", m_con, fs_con_b, query_tokens, buffer_id=vocab - 1)
        logits_con_query_b = _recurrent_continuation_answer_logits(
            "contender", m_con, fs_con_a, query_tokens_b, buffer_id=vocab - 1)

        _, fs_abl_a = m_abl(bind_a, return_states=True)
        _, fs_abl_b = m_abl(bind_b, return_states=True)
        logits_abl_bind_a = _recurrent_continuation_answer_logits(
            "ablation", m_abl, fs_abl_a, query_tokens, buffer_id=vocab - 1)
        logits_abl_bind_b = _recurrent_continuation_answer_logits(
            "ablation", m_abl, fs_abl_b, query_tokens, buffer_id=vocab - 1)

    plumbing_ok = (logits_con_bind_a.shape == (2, 3, vocab) and torch.isfinite(logits_con_bind_a).all().item()
                  and not torch.allclose(logits_con_bind_a, logits_con_query_b))   # query reaches output
    stub_degenerate_confirmed = torch.equal(logits_con_bind_a, logits_con_bind_b)  # S_T=0 -> bind-independent
    ablation_bind_dependent = not torch.allclose(logits_abl_bind_a, logits_abl_bind_b)  # real recurrence
    ok15 = plumbing_ok and stub_degenerate_confirmed and ablation_bind_dependent
    rep("selftest 15 (item 1's CPU-stub caveat, mirrors smoke_3/smoke_8's two-half "
        "discipline): PLUMBING CPU-provable (shape/finite/query-reaches-output); the "
        "VALUE-level claim (does CE_answer carry real recall signal from a nonzero S_T) is "
        "NOT CPU-provable for the contender -- confirmed degenerate/bind-independent under the "
        "stub's zero final_state, registered BOX-ONLY; the ablation (no stub disconnect) IS "
        "bind-dependent on CPU, the expected contrast", ok15,
        f"plumbing_ok={plumbing_ok} stub_degenerate_confirmed={stub_degenerate_confirmed} "
        f"ablation_bind_dependent={ablation_bind_dependent}")

    # 16 (AUD2-F1 fix verification, sec 1.24 pre-launch build-fix): the NEW slice-before-matmul
    #     continuation (model.forward(..., return_hidden=True), slice, THEN F.linear) must be
    #     BIT-IDENTICAL at the answer position to the OLD full-position-then-discard path
    #     (model.forward(...) computing vocab logits at ALL 128 padded positions, then indexing
    #     one). The OLD path is re-derived HERE ONLY (a local reference, never imported/shipped)
    #     so this test does not just compare the new code against itself.
    def _old_recurrent_continuation_answer_logits(arch, model, final_states, query_tokens, buffer_id):
        B, Q, qlen = query_tokens.shape
        flat = query_tokens.reshape(B * Q, qlen)
        padded = _pad_query_tokens_for_continuation(flat, buffer_id)
        states_rep = _repeat_states_for_queries(final_states, Q)
        logits_query = model.forward(padded, initial_states=states_rep)   # pre-fix: full LM head, all T
        answer_logits = logits_query[:, qlen - 1, :]
        return answer_logits.view(B, Q, -1)

    ok16 = True
    for arch in ("contender", "ablation"):
        torch.manual_seed(71)
        vocab = 300
        m = (DeltaNetLM(vocab, d_model=32, d_state=64, n_layers=1, conv_size=4) if arch == "contender"
             else AblationLM(vocab, d_model=32, d_state=16, n_layers=1, conv_size=4))
        bind_tokens = torch.randint(0, vocab, (2, _MIN_KERNEL_T))
        query_tokens = torch.randint(0, vocab, (2, 3, 6))
        with torch.no_grad():
            _, final_states = m(bind_tokens, return_states=True)
            logits_new = _recurrent_continuation_answer_logits(arch, m, final_states, query_tokens,
                                                                buffer_id=vocab - 1)
            logits_old = _old_recurrent_continuation_answer_logits(arch, m, final_states, query_tokens,
                                                                    buffer_id=vocab - 1)
        identical = torch.equal(logits_new, logits_old)
        ok16 = ok16 and identical
        print(f"    arch={arch}: old-path vs new-path answer logits bit-identical={identical}",
              flush=True)
    rep("selftest 16 (AUD2-F1 fix verification): slice-before-matmul continuation reproduces "
        "BIT-IDENTICAL answer-position logits vs. the pre-fix full-position-then-discard "
        "reference (reproduced locally, not imported)", ok16)

    # 17 (AUD2-F2 fix, sec 1.24 pre-launch build-fix): planted-answer test for the K-restricted
    #     gather+argmax (sec 1.23's own "build precision note") -- a NON-candidate token holds
    #     the GLOBAL-vocab max while a candidate holds the max AMONG the K candidates; the real
    #     production function (_rung1_k_restricted_pred_slot) must still return the candidate. A
    #     deliberately-globalized MUTANT (drops the K-restriction, does a plain global-vocab
    #     argmax, maps back to a slot ONLY if the winner happens to be a candidate -- else -1,
    #     a guaranteed miss) must FAIL the identical comparison -- run to completion to prove
    #     the test has teeth (house hard rule: negative tests must actually be exercised).
    def _rung1_globalized_mutant(answer_logits, entity_ids):
        K = entity_ids.shape[1]
        entity_ids_exp = entity_ids.unsqueeze(1).expand(-1, answer_logits.shape[1], K)  # (B,Q,K)
        global_top = answer_logits.argmax(dim=-1, keepdim=True)     # (B,Q,1) VOCAB id, unrestricted
        match = (entity_ids_exp == global_top)                      # (B,Q,K)
        any_match = match.any(dim=-1)
        slot = match.float().argmax(dim=-1)
        return torch.where(any_match, slot, torch.full_like(slot, -1))

    torch.manual_seed(81)
    B_t, Q_t, K_t, V_t = 2, 3, 5, 50
    answer_logits_t = torch.randn(B_t, Q_t, V_t) * 0.1
    entity_ids_t = torch.stack([torch.randperm(V_t)[:K_t] for _ in range(B_t)])    # (B,K), episode-shared
    tgt_slot_t = torch.randint(0, K_t, (B_t, Q_t))
    for b in range(B_t):
        cand_ids = entity_ids_t[b]
        non_cand_pool = [v for v in range(V_t) if v not in cand_ids.tolist()]
        for q in range(Q_t):
            answer_logits_t[b, q, cand_ids[tgt_slot_t[b, q]]] = 5.0     # candidate: max AMONG the K
            answer_logits_t[b, q, non_cand_pool[q % len(non_cand_pool)]] = 10.0  # NON-candidate: GLOBAL max

    pred_slot_restricted = _rung1_k_restricted_pred_slot(answer_logits_t, entity_ids_t)
    restricted_correct = torch.equal(pred_slot_restricted, tgt_slot_t)
    pred_slot_mutant = _rung1_globalized_mutant(answer_logits_t, entity_ids_t)
    mutant_correct = torch.equal(pred_slot_mutant, tgt_slot_t)
    ok17 = restricted_correct and not mutant_correct
    rep("selftest 17 (AUD2-F2, planted-answer): K-restricted gather+argmax returns the "
        "candidate despite a non-candidate global-vocab max; the deliberately-globalized "
        "mutant FAILS the identical comparison (proves the test has teeth)", ok17,
        f"restricted_correct={restricted_correct} mutant_correct={mutant_correct}")

    print("=" * 70)
    if failures:
        print(f"SELFTEST: {len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        return 1
    print("SELFTEST: ALL ITEMS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="alias for --selftest (house convention)")
    ap.add_argument("--token-probe", action="store_true",
                    help="exit 0 iff launch tokens present (gate 5 negative-test hook)")
    ap.add_argument("--list-cells", choices=["calibration", "sweep"])
    ap.add_argument("--run-cell", type=str)
    ap.add_argument("--set", choices=["calibration", "sweep"], default="calibration")
    ap.add_argument("--pilot-pair", nargs=2, metavar=("ARCH", "TASK"))
    ap.add_argument("--gate-pilots", action="store_true")
    ap.add_argument("--pilots-dir", type=str)
    ap.add_argument("--msweep-pilot", action="store_true")
    ap.add_argument("--fanout-all", action="store_true")
    ap.add_argument("--horizon-ref", action="store_true")
    ap.add_argument("--write-calibration-report", action="store_true")
    ap.add_argument("--ckpt", type=str)
    ap.add_argument("--ckpt-map", type=str)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--out", type=str)
    ap.add_argument("--out-dir", type=str)
    ap.add_argument("--res-dir", type=str)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gates-dir", type=str, default="results/h2h_rung1/gates")
    ap.add_argument("--margins-token", type=str, default="results/h2h_rung1/MARGINS_FROZEN.token")
    ap.add_argument("--headroom-gpu-h", type=float, default=100.0)
    args = ap.parse_args()

    if args.selftest or args.smoke:
        return mode_selftest()
    if args.token_probe:
        require_launch_tokens(args.gates_dir)
        print("launch tokens present (gate 5)")
        return 0
    if args.list_cells:
        for c in (calibration_cells() if args.list_cells == "calibration" else sweep_cells()):
            print(c["name"])
        return 0
    if args.run_cell:
        return mode_run_cell(args)
    if args.pilot_pair:
        return mode_pilot_pair(args)
    if args.gate_pilots:
        return mode_gate_pilots(args)
    if args.msweep_pilot:
        return mode_msweep_pilot(args)
    if args.fanout_all:
        return mode_fanout_all(args)
    if args.horizon_ref:
        return mode_horizon_ref(args)
    if args.write_calibration_report:
        return mode_calibration_report(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
