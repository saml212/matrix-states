"""probe_head_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1: the Wave -1
shared probe head (F1's entire resolution). ONE architecture-neutral,
TRAINED readout, reused across all three H2H arms, closing the FATAL gap
Rev 0 shipped with no working readout for any of its three arms.

Pieces, each pinned exactly per sec 1.3.1.1-1.3.1.4:
  - `build_probe_target_table` -- the frozen T_val ground-truth table
    (`key_anchoring.random_unit_rows_init`, reused UNMODIFIED, at the NEW
    `PROBE_TARGET_SEED` -- sec 1.3.1.1).
  - `build_shared_probe` / `build_adapter_arm` -- the IDENTICAL-class/init
    `shared_probe` (nn.Linear(value_dim,value_dim,bias=True)) and per-arm
    `adapter_arm` (nn.Linear(native_tap_dim,value_dim,bias=False)) factories
    -- sec 1.3.1.2.
  - `contender_native_tap` / `ablation_native_tap` / `transformer_native_tap`
    -- the three per-arm `state_summary_raw` extraction functions, EXACTLY
    per sec 1.3.1.2's own construction (contender: `S_T_last @ q_query` via
    a non-recurrent query pathway; ablation: Hadamard `s_T (.) q_query`;
    Transformer: final-block post-norm hidden at `<Q>` from a real
    context-extending forward pass -- the one arm whose query DOES enter
    its own context, a disclosed, structural asymmetry, sec 1.9 item 9).
  - `cosine_recovery_frac` / `probe_aux_loss` -- the frozen metric/loss
    formula (`F.cosine_similarity(...) @ 0.9`; `1 - cos` aux loss,
    `aux_weight=0.1` default) -- sec 1.3.1.3.
  - `run_probe_capacity_null` -- the MANDATORY sec 1.3.1.4 sanity null
    (gate 7): frozen-random `state_summary_raw`, fresh draws every step,
    held-out fresh eval draws, pass bar `recovered_frac@0.9 < 0.05`.

AUD-F1 fix (build audit §1.20, this file's own substantive finding): smoke_7's
JOINT (CE+aux) loss gradient check is CONFOUNDED evidence that "the aux loss
backprops into the backbone" (sec 1.3.1.3's core premise) -- loss_ce alone
already makes q_proj nonzero, aux_weight=0 would pass the same check. smoke_8
below replaces it as the load-bearing claim: an AUX-LOSS-ONLY (CE-excluded)
gradient-isolation test. CPU/box split, stated precisely: the CPU stub's
`final_state = torch.zeros(..., requires_grad=False)` is a disconnected
constant, so the contender's tap (`S_T_last @ q_last`) is identically zero
regardless of q_last -- this kills BOTH the forward value AND the gradient of
the contender's aux path on CPU (not just k/v/b -- q_proj too, for a
stub-specific reason distinct from the real-hardware q_proj exclusion smoke_8
documents). The contender arm is therefore registered BOX-ONLY (mirrors
smoke_3's box-only registration discipline exactly); ablation + Transformer
have no such stub disconnect and are PROVEN genuinely nonzero on CPU now.

Run the smoke gate (incl. the gate-7 null for all three arms' REAL adapter
shapes at rung-1, and smoke_8's aux-ONLY gradient isolation): python
probe_head_rd.py --smoke
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_constants_rd import PROBE_TARGET_SEED
from key_anchoring import random_unit_rows_init                        # no fla dep -- safe pre-stub
from h2h_fla_stub_rd import ensure_fla_stub

_STUB_INSTALLED = ensure_fla_stub()
from lm_pretrain_rd import DeltaNetLM                                   # noqa: E402
from ablation_mixer_rd import AblationLM                                # noqa: E402
from transformer_baseline_rd import TransformerLM                       # noqa: E402

RECOVERY_THRESHOLD = 0.9
AUX_WEIGHT_DEFAULT = 0.1
NULL_PASS_BAR = 0.05
CE_ANSWER_WEIGHT_DEFAULT = 1.0    # sec 1.3.1.3 Rev 4: pinned default (h2h_cell_train_rd.py's own
                                    # CE_ANSWER_WEIGHT mirrors this; calibrated by the step-500 dial)
RUNG_CHANCE_MULT = 3.0            # sec 1.7 gate 1a: rungs 1/2 PASS bar, > 3x episode-restricted chance
RUNG2_NULL_PASS_MULT = 1.5        # R5-F2: the rung-2 classifier's OWN capacity null, strictly
                                    # SEPARATED from the real 3x gate -- a looser <=1.5x bar
# lm_pretrain_rd._MIN_KERNEL_T (128): the real chunk_delta_rule kernel's own backward-crash floor
# (F15-LM measurement) -- every smoke item below that instantiates a REAL DeltaNetLM must feed it
# a context >= this length; used only inside this file's own smoke suite (not re-exported).
_CONTENDER_SMOKE_CTX_LEN = 128


# ---------------------------------------------------------------------------
# sec 1.3.1.1 -- frozen ground-truth target table
# ---------------------------------------------------------------------------

def build_probe_target_table(vocab_size_total: int, value_dim: int,
                              seed: int = PROBE_TARGET_SEED) -> torch.Tensor:
    """T_val: (vocab_size_total, value_dim), frozen (caller registers as a buffer +
    requires_grad_(False) -- this function only builds the tensor, matching
    random_unit_rows_init's own init-vs-freeze separation of concerns)."""
    return random_unit_rows_init(vocab_size_total, value_dim, seed)


# ---------------------------------------------------------------------------
# sec 1.3.1.2 -- shared probe + per-arm adapter factories
# ---------------------------------------------------------------------------

def build_shared_probe(value_dim: int) -> nn.Linear:
    return nn.Linear(value_dim, value_dim, bias=True)


def build_adapter_arm(native_tap_dim: int, value_dim: int) -> nn.Linear:
    return nn.Linear(native_tap_dim, value_dim, bias=False)


# ---------------------------------------------------------------------------
# sec 1.3.1.2 -- per-arm native taps
# ---------------------------------------------------------------------------

def contender_native_tap(model: DeltaNetLM, context_token_ids: torch.Tensor,
                          query_tokens: torch.Tensor) -> torch.Tensor:
    """S_T_last @ q_query. context_token_ids: (B,T_ctx) -- streamed through
    model.forward(..., return_states=True); the query window is NEVER
    appended to this call (protects the P=1 bottleneck). query_tokens:
    (B,Q,query_len) -- run through the LAST block's OWN
    embed->norm1->q_proj->q_conv1d pathway ONLY (generalizes
    model_rd.py's `effective_key_window` precedent to the contender's real
    trained q_proj/q_conv1d). Returns (B,Q,d_state)."""
    _, final_states = model(context_token_ids, return_states=True)
    S_T_last = final_states[-1]                       # (B,H,head_dim,head_dim), H==1 every rung
    assert S_T_last.shape[1] == 1, (
        f"contender_native_tap assumes num_heads==1 (every registered rung config), got "
        f"num_heads={S_T_last.shape[1]}")
    S = S_T_last.squeeze(1)                             # (B,d_state,d_state)

    B, Q, qlen = query_tokens.shape
    last_block = model.blocks[-1]
    x_embed = model.embed(query_tokens.reshape(B * Q, qlen))
    a = last_block.norm1(x_embed)
    q_conv, _ = last_block.mixer.q_conv1d(last_block.mixer.q_proj(a))
    q_last = q_conv[:, -1, :].view(B, Q, -1)             # <Q> position's post-conv q, (B,Q,d_state)

    return torch.einsum("bij,bqj->bqi", S, q_last)       # per-query matvec read, (B,Q,d_state)


def ablation_native_tap(model: AblationLM, context_token_ids: torch.Tensor,
                         query_tokens: torch.Tensor) -> torch.Tensor:
    """s_T (.) q_query (Hadamard) -- same non-recurrent query-pathway
    discipline as the contender's own tap. Returns (B,Q,d_state)."""
    _, final_states = model(context_token_ids, return_states=True)
    s_T = final_states[-1]                                # (B,d_state)

    B, Q, qlen = query_tokens.shape
    last_block = model.blocks[-1]
    x_embed = model.embed(query_tokens.reshape(B * Q, qlen))
    a = last_block.norm1(x_embed)
    q_conv, _ = last_block.mixer.q_conv1d(last_block.mixer.q_proj(a))
    q_last = q_conv[:, -1, :].view(B, Q, -1)

    return s_T.unsqueeze(1) * q_last                      # (B,1,d_state) * (B,Q,d_state)


def transformer_native_tap(model: TransformerLM, context_token_ids: torch.Tensor,
                            query_tokens: torch.Tensor, attn_mask_fn=None) -> torch.Tensor:
    """Final-block (post-norm_f) hidden state at each query's own `<Q>`
    position, from a SINGLE forward pass per (batch,query) pair over
    `[context ++ that one query_window]` -- the ONE arm whose query DOES
    enter its own context (sec 1.9 item 9's disclosed, structural
    asymmetry: attention has no non-attention read channel). `attn_mask_fn`:
    None (uncapped, b-control/training) or `callable(T_total) -> (T_total,
    T_total) bool mask` (hard-capped, b-primary, INFERENCE-ONLY). Returns
    (B,Q,d_model). NOTE (disclosed compute cost, R3-F4's own concern): this
    replicates the context Q times (once per query) since the Transformer
    has no query-free read channel to reuse a single context pass across
    queries -- exactly why R3-F4 gates the M-sweep fan-out behind a REAL
    timing pilot rather than a design-time assumption.

    sec 1.31.4 item 3 (OOM fix): `model(..., return_hidden=True)` now
    returns hidden ONLY (never a (logits, hidden) tuple -- see
    transformer_baseline_rd.TransformerLM.forward's own updated
    docstring), so this function was already computing NO wasted
    full-vocab matmul once that fix landed -- it reads `hidden[:, -1, :]`
    directly. This IS, by construction, the §1.31.2-pinned Leg-B tap
    ("the post-block-1, pre-LM-head hidden") for the Transformer arm: its
    native tap has always been the pre-LM-head hidden at `<Q>`, unlike the
    recurrent arms' native tap (S1@q_shallow, found causally inert/
    linearly dead at §1.30) -- no separate Leg-B extraction path is needed
    for this arch."""
    B, T_ctx = context_token_ids.shape
    _, Q, qlen = query_tokens.shape
    ctx_rep = context_token_ids.unsqueeze(1).expand(B, Q, T_ctx).reshape(B * Q, T_ctx)
    q_flat = query_tokens.reshape(B * Q, qlen)
    seq = torch.cat([ctx_rep, q_flat], dim=1)
    mask = attn_mask_fn(T_ctx + qlen) if attn_mask_fn is not None else None
    hidden = model(seq, attn_mask=mask, return_hidden=True)
    return hidden[:, -1, :].view(B, Q, -1)


# ---------------------------------------------------------------------------
# sec 1.3.1.2 (metric) / sec 1.3.1.3 (training regime)
# ---------------------------------------------------------------------------

def cosine_recovery_frac(pred: torch.Tensor, target: torch.Tensor,
                          threshold: float = RECOVERY_THRESHOLD) -> torch.Tensor:
    """The frozen `recovered_frac@0.9` metric formula, unchanged across
    every revision of this design -- fraction of rows clearing the
    threshold."""
    cos = F.cosine_similarity(pred, target, dim=-1)
    return (cos >= threshold).float().mean()


def probe_aux_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """mean(1 - cos(pred, target)) -- sec 1.3.1.3's joint aux loss term."""
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def joint_loss(loss_ce: torch.Tensor, pred: torch.Tensor, target: torch.Tensor,
                aux_weight: float = AUX_WEIGHT_DEFAULT,
                loss_ce_answer: torch.Tensor | None = None,
                ce_answer_weight: float = CE_ANSWER_WEIGHT_DEFAULT) -> torch.Tensor:
    """sec 1.3.1.3, Rev 4: loss_total = loss_CE_lm + ce_answer_weight * CE_answer +
    aux_weight * mean(1 - cos(pred,target)). `loss_ce_answer=None` (the default) reproduces
    the ORIGINAL Rev 0-3 two-term formula EXACTLY -- smoke_7's own still-valid joint-loss
    sanity check passes an unmodified call here, deliberately never silently promoted to a
    three-term loss it wasn't written to exercise. CE_answer participates in loss_total ONLY
    when the caller explicitly supplies `loss_ce_answer` (h2h_cell_train_rd.py's own
    train_grammar_cell, every step, Rev 4)."""
    total = loss_ce + aux_weight * probe_aux_loss(pred, target)
    if loss_ce_answer is not None:
        total = total + ce_answer_weight * loss_ce_answer
    return total


# ---------------------------------------------------------------------------
# sec 1.3.1.4 -- MANDATORY probe-capacity sanity null (gate 7)
# ---------------------------------------------------------------------------

def run_probe_capacity_null(native_tap_dim: int, value_dim: int, T_val: torch.Tensor,
                             n_train_steps: int = 500, batch_size: int = 64, lr: float = 1e-3,
                             n_eval: int = 2000, threshold: float = RECOVERY_THRESHOLD,
                             seed: int = 0) -> dict:
    """Trains `adapter_arm + shared_probe` (this arm's REAL adapter shape)
    on FRESH i.i.d. Gaussian `state_summary_raw` every step (never a fixed
    pool -- m-new-2's own pinned protocol) against FRESH random rows of the
    SAME frozen `T_val` table, then scores `recovered_frac@0.9` on a
    SEPARATE held-out set of fresh draws never seen during training. Pass:
    `recovered_frac < 0.05` (the probe alone, with no real state to read,
    must not "solve" the task)."""
    torch.manual_seed(seed)
    adapter = build_adapter_arm(native_tap_dim, value_dim)
    probe = build_shared_probe(value_dim)
    opt = torch.optim.Adam(list(adapter.parameters()) + list(probe.parameters()), lr=lr)
    n_rows = T_val.shape[0]

    for _ in range(n_train_steps):
        x = torch.randn(batch_size, native_tap_dim)
        target = T_val[torch.randint(0, n_rows, (batch_size,))]
        pred = probe(adapter(x))
        loss = probe_aux_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        x_eval = torch.randn(n_eval, native_tap_dim)
        target_eval = T_val[torch.randint(0, n_rows, (n_eval,))]
        pred_eval = probe(adapter(x_eval))
        frac = cosine_recovery_frac(pred_eval, target_eval, threshold).item()

    return {"recovered_frac": frac, "passed": frac < NULL_PASS_BAR, "native_tap_dim": native_tap_dim,
            "value_dim": value_dim, "n_train_steps": n_train_steps, "n_eval": n_eval}


# ---------------------------------------------------------------------------
# sec 1.7 gate 1a rung 2 -- identity classifier factory + its OWN R5-F2 capacity null
# ---------------------------------------------------------------------------

def build_identity_classifier(native_tap_dim: int, K: int) -> nn.Linear:
    """Rung 2's classifier: SEPARATE from `shared_probe` (which does continuous cosine
    regression against T_val, not classification) -- a plain linear map from the arm's own
    native tap (`adapter_arm`'s own INPUT shape, native_tap_dim -- NOT the post-adapter
    value_dim) to K SLOT logits, predicting which of the episode's K candidate entities is
    being queried (sec 1.7 gate 1a rung 2). Trained fresh per calibration cell in
    h2h_cell_train_rd.py on REAL episodes; this factory is reused by both that real-data
    trainer and this file's own R5-F2 synthetic capacity null below, so the two share
    byte-identical class/init/shape."""
    return nn.Linear(native_tap_dim, K, bias=True)


def run_identity_classifier_capacity_null(native_tap_dim: int, K: int, n_train_steps: int = 300,
                                           batch_size: int = 64, lr: float = 1e-3, n_eval: int = 2000,
                                           seed: int = 0) -> dict:
    """R5-F2 (attack round 5, sec 1.23): rung 2's OWN frozen-random-tap capacity null --
    unlike `shared_probe`, rung 2 is a NEW classifier introduced by Rev 4, so a rung-2 PASS on
    real data would not otherwise be distinguishable from "the classifier itself is oddly
    well-suited to noise." Mirrors `run_probe_capacity_null`'s protocol (fresh i.i.d. Gaussian
    `state_summary_raw` every step, never a fixed pool; held-out fresh eval draws never seen
    in training) with ONE deliberate difference required by R5-F2's own spec: LABELS
    INDEPENDENT -- the K-class label is drawn uniformly at random, INDEPENDENT of the (already
    label-free) Gaussian input, so there is no learnable input-to-label relationship at all for
    the classifier to memorize even in principle. Pass bar, STRICTLY SEPARATED from the real
    rung-2 gate's `> 3x chance` bar (sec 1.7 gate 1a): `accuracy <= RUNG2_NULL_PASS_MULT (1.5) x
    chance (1/K)` -- looser than the real gate on purpose (R5-F2's own text), because this null
    has zero signal to find and must not spuriously fail from ordinary finite-sample noise at a
    tight bar."""
    torch.manual_seed(seed)
    clf = build_identity_classifier(native_tap_dim, K)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)

    for _ in range(n_train_steps):
        x = torch.randn(batch_size, native_tap_dim)
        y = torch.randint(0, K, (batch_size,))
        logits = clf(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        x_eval = torch.randn(n_eval, native_tap_dim)
        y_eval = torch.randint(0, K, (n_eval,))
        pred = clf(x_eval).argmax(dim=-1)
        acc = (pred == y_eval).float().mean().item()

    chance = 1.0 / K
    pass_bar = RUNG2_NULL_PASS_MULT * chance
    return {"accuracy": acc, "chance": chance, "pass_bar": pass_bar, "passed": acc <= pass_bar,
            "native_tap_dim": native_tap_dim, "K": K, "n_train_steps": n_train_steps, "n_eval": n_eval}


# ---------------------------------------------------------------------------
# sec 1.31.4 item 2 -- rung-2's BOTH-DIRECTIONS teeth: the positive-control half (the negative
# half is `run_identity_classifier_capacity_null` above, R5-F2, unchanged).
# ---------------------------------------------------------------------------

RUNG2_PLANTED_SIGNAL_PASS_BAR = 0.90   # sec 1.31.4 item 2: "must score near-ceiling (>=90%)"


def rung2_planted_signal_positive_control(tap_dim: int, n_classes: int, n_slots: int = 32,
                                           n_train_steps: int = 300, batch_size: int = 64,
                                           lr: float = 1e-3, n_eval: int = 2000,
                                           seed: int = 0) -> dict:
    """sec 1.31.4 item 2 (the "both-directions teeth rule"): a SYNTHETIC tap with LINEARLY
    PLANTED entity identity (`tap = W_plant[identity] + small_noise`, `W_plant` a fixed random
    projection) that a freshly-trained identity classifier MUST recover near-ceiling
    (`>=RUNG2_PLANTED_SIGNAL_PASS_BAR`) -- proving the classifier + training loop have the
    CAPACITY to read an actually-present linear identity signal, the positive-direction
    complement to `run_identity_classifier_capacity_null`'s negative-direction "no signal
    present" null. An instrument is trusted only when it passes BOTH directions (sec 1.28's
    own new rule).

    ALSO returns the sec 1.31.4 item 1 "relabel matters" proof required alongside the relabel
    itself: on the EXACT SAME planted tap, a classifier trained against the OLD `tgt_slot`
    labels (drawn uniformly at random, INDEPENDENT of the planted identity -- grammar_rd's own
    "slots are uniform given identity" property, sec 1.28's diagnosed structural-vacuity
    defect) must read ~chance, because there is no learnable identity-to-slot relationship in
    this construction even though identity itself is perfectly linear. This is the single
    experiment sec 1.31.4 item 1 calls for: "a synthetic tap that encodes identity perfectly
    must score >>chance; the old slot-labeled construction must read ~chance on the same
    plant -- proving the relabel matters.\""""
    torch.manual_seed(seed)
    W_plant = torch.randn(n_classes, tap_dim) / (tap_dim ** 0.5)

    def _make_batch(n: int):
        y_identity = torch.randint(0, n_classes, (n,))
        tap = W_plant[y_identity] + 0.01 * torch.randn(n, tap_dim)
        y_slot = torch.randint(0, n_slots, (n,))          # uniform, INDEPENDENT of y_identity/tap
        return tap, y_identity, y_slot

    clf_identity = build_identity_classifier(tap_dim, n_classes)
    clf_slot = build_identity_classifier(tap_dim, n_slots)     # the OLD (pre-Rev-5.1) label scheme
    opt_i = torch.optim.Adam(clf_identity.parameters(), lr=lr)
    opt_s = torch.optim.Adam(clf_slot.parameters(), lr=lr)
    for _ in range(n_train_steps):
        tap, y_identity, y_slot = _make_batch(batch_size)
        loss_i = F.cross_entropy(clf_identity(tap), y_identity)
        opt_i.zero_grad(); loss_i.backward(); opt_i.step()
        loss_s = F.cross_entropy(clf_slot(tap), y_slot)
        opt_s.zero_grad(); loss_s.backward(); opt_s.step()

    with torch.no_grad():
        tap_e, y_identity_e, y_slot_e = _make_batch(n_eval)
        acc_identity = (clf_identity(tap_e).argmax(-1) == y_identity_e).float().mean().item()
        acc_slot = (clf_slot(tap_e).argmax(-1) == y_slot_e).float().mean().item()

    slot_chance = 1.0 / n_slots
    old_labels_reads_chance = acc_slot <= RUNG2_NULL_PASS_MULT * slot_chance
    identity_passed = acc_identity >= RUNG2_PLANTED_SIGNAL_PASS_BAR
    return {"identity_accuracy": acc_identity, "identity_pass_bar": RUNG2_PLANTED_SIGNAL_PASS_BAR,
            "identity_passed": identity_passed,
            "old_slot_labeled_accuracy": acc_slot, "old_slot_labeled_chance": slot_chance,
            "old_slot_labeled_reads_chance": old_labels_reads_chance,
            "n_classes": n_classes, "n_slots": n_slots, "tap_dim": tap_dim,
            "passed": identity_passed and old_labels_reads_chance}


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def smoke_1_target_table_frozen_unit_rows():
    t1 = build_probe_target_table(500, 16)
    t2 = build_probe_target_table(500, 16)
    same_seed_identical = torch.equal(t1, t2)
    t3 = build_probe_target_table(500, 16, seed=999)
    different_seed_different = not torch.equal(t1, t3)
    unit_rows = torch.allclose(t1.norm(dim=-1), torch.ones(500), atol=1e-5)
    ok = same_seed_identical and different_seed_different and unit_rows
    _report("smoke 1: build_probe_target_table determinism + unit-row property", ok,
            f"same_seed_identical={same_seed_identical} different_seed_different="
            f"{different_seed_different} unit_rows={unit_rows}")


def smoke_2_contender_tap_forward_backward():
    # d_state PINNED to 64 (contender's own _SAFE_D_STATE floor for the real Triton kernel's
    # backward -- model_rd.py's F15-LM measurement; smaller d_state is not a valid contender shape
    # anywhere in this codebase, not just an arbitrary smoke-test choice).
    torch.manual_seed(1)
    m = DeltaNetLM(300, d_model=32, d_state=64, n_layers=2, conv_size=4)
    adapter = build_adapter_arm(64, 12)
    probe = build_shared_probe(12)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    tap = contender_native_tap(m, ctx, q_tok)
    pred = probe(adapter(tap))
    target = torch.randn(2, 3, 12)
    loss = probe_aux_loss(pred, target)
    loss.backward()
    ok = (tap.shape == (2, 3, 64) and torch.isfinite(tap).all().item()
          and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    _report("smoke 2: contender_native_tap forward + backward-through-adapter+probe", ok,
            f"tap.shape={tuple(tap.shape)}")


def smoke_3_contender_tap_p1_bottleneck_blankout():
    """Blank-out check (CLAUDE.md's own required discipline), decomposed
    into two independently CPU-stub-provable halves:
      (a) S_T_last is a pure function of context ALONE -- it comes from a
          `model(context_token_ids, return_states=True)` call that
          structurally never receives query_tokens, so perturbing the
          query cannot change it (checked directly: bit-identical
          final_states across two different query draws).
      (b) q_last (the LAST block's own embed->norm1->q_proj->q_conv1d
          pathway output) IS a function of the query window and changes
          when it is perturbed -- checked directly on the intermediate,
          not the full tap.
    DISCLOSED BOX-ONLY LIMITATION (h2h_fla_stub_rd.py's own documented
    stub behavior): `_stub_chunk_delta_rule` always returns an all-ZERO
    `final_state`, so `state_summary_raw = S_T_last @ q_last` is
    IDENTICALLY ZERO under the CPU stub regardless of q_last -- the
    end-to-end claim "the FULL tap changes when q changes" cannot be
    exercised meaningfully on this dev box; it requires the real Triton
    kernel (a nonzero, trained S_T_last) and is registered here as a
    box-only follow-up smoke item for the deploy stage, not silently
    skipped."""
    torch.manual_seed(2)
    m = DeltaNetLM(300, d_model=32, d_state=64, n_layers=1, conv_size=4)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    q_tok_perturbed = torch.randint(0, 300, (2, 3, 5))
    with torch.no_grad():
        _, fs1 = m(ctx, return_states=True)
        _, fs2 = m(ctx, return_states=True)
        last_block = m.blocks[-1]

        def _q_last(qt):
            B, Q, qlen = qt.shape
            a = last_block.norm1(m.embed(qt.reshape(B * Q, qlen)))
            qc, _ = last_block.mixer.q_conv1d(last_block.mixer.q_proj(a))
            return qc[:, -1, :].view(B, Q, -1)

        q1, q2 = _q_last(q_tok), _q_last(q_tok_perturbed)
    state_unchanged_by_query = torch.equal(fs1[-1], fs2[-1])
    q_last_changed_by_query = not torch.allclose(q1, q2)
    ok = state_unchanged_by_query and q_last_changed_by_query
    _report("smoke 3: contender P=1 bottleneck blank-out (state is context-only; q_last IS "
            "query-dependent; end-to-end tap-changes-with-q needs a real nonzero S_T -- box-only, "
            "see docstring)", ok,
            f"state_unchanged_by_query={state_unchanged_by_query} "
            f"q_last_changed_by_query={q_last_changed_by_query}")


def smoke_4_ablation_tap_forward_and_bottleneck():
    torch.manual_seed(4)
    m = AblationLM(300, d_model=32, d_state=16, n_layers=1, conv_size=4)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    tap = ablation_native_tap(m, ctx, q_tok)
    with torch.no_grad():
        _, fs1 = m(ctx, return_states=True)
        q_tok_perturbed = torch.randint(0, 300, (2, 3, 5))
        _, fs2 = m(ctx, return_states=True)
        tap2 = ablation_native_tap(m, ctx, q_tok_perturbed)
    state_unchanged = torch.equal(fs1[-1], fs2[-1])
    tap_changed = not torch.allclose(tap, tap2)
    ok = tap.shape == (2, 3, 16) and state_unchanged and tap_changed
    _report("smoke 4: ablation_native_tap shape + P=1 bottleneck blank-out", ok,
            f"tap.shape={tuple(tap.shape)} state_unchanged={state_unchanged} tap_changed={tap_changed}")


def smoke_5_transformer_tap_uncapped_vs_capped():
    from transformer_baseline_rd import sink_fifo_mask
    torch.manual_seed(6)
    m = TransformerLM(300, d_model=32, n_layers=2, n_heads=4, ffn_mult=2)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    tap_uncapped = transformer_native_tap(m, ctx, q_tok, attn_mask_fn=None)
    tap_capped = transformer_native_tap(
        m, ctx, q_tok, attn_mask_fn=lambda t: sink_fifo_mask(t, cap_length=10, k_sink=4))
    ok = (tap_uncapped.shape == (2, 3, 32) and tap_capped.shape == (2, 3, 32)
          and not torch.allclose(tap_uncapped, tap_capped))
    _report("smoke 5: transformer_native_tap shape + uncapped-vs-capped modes actually differ", ok,
            f"tap_uncapped.shape={tuple(tap_uncapped.shape)}")


def smoke_6_probe_capacity_null_all_three_arm_shapes():
    """Gate 7 itself: the mandatory null MUST pass for every arm's REAL
    rung-1 adapter shape before HEADTOHEAD_MATCH_GATE_SIGNOFF may be set."""
    T_val = build_probe_target_table(2000, 64, seed=PROBE_TARGET_SEED)
    # native_tap_dim per arm, rung-1; a FIXED, deterministic per-arm seed offset (never
    # Python's hash() -- CLAUDE.md's own hard rule: hash() is salted per-process, zlib.crc32 or a
    # literal pin is the reproducible alternative; a literal enumeration is simplest here).
    shapes = {"contender": (64, 1), "ablation": (64, 2), "transformer": (256, 3)}
    all_pass = True
    for arm, (dim, seed_offset) in shapes.items():
        result = run_probe_capacity_null(dim, 64, T_val, n_train_steps=300, seed=seed_offset)
        print(f"    arm={arm:>11s}: native_tap_dim={dim:>3d} recovered_frac={result['recovered_frac']:.4f} "
              f"passed={result['passed']}")
        all_pass = all_pass and result["passed"]
    _report("smoke 6 (GATE 7): probe-capacity null PASSES for all 3 arms' real rung-1 adapter "
            "shapes (bar < 0.05)", all_pass)


def smoke_7_joint_loss_gradients_flow_to_backbone():
    """AUD-F1 (audit §1.20): this JOINT-loss (CE + aux) check alone is
    CONFOUNDED evidence for "the aux loss reaches the backbone" -- loss_ce
    touches q_proj regardless of aux_weight (a plain LM loss would make
    this PASS even with aux_weight=0), so a PASS here proves CE reaches
    q_proj, not that the aux term specifically does. Retained as a
    mechanical joint-loss sanity check (the real training regime always
    uses the joint loss, so this remains a genuine, non-vacuous
    "no accidental detach anywhere in the combined graph" smoke), but
    smoke_8 below is the ONLY item this suite treats as evidence for
    sec 1.3.1.3's own "aux loss backprops into the backbone" premise --
    see its docstring for the honest CPU/box split the audit required."""
    torch.manual_seed(8)
    m = DeltaNetLM(300, d_model=32, d_state=64, n_layers=1, conv_size=4)
    adapter = build_adapter_arm(64, 12)
    probe = build_shared_probe(12)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    y = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    logits = m(ctx)
    loss_ce = F.cross_entropy(logits.reshape(-1, 300), y.reshape(-1))
    tap = contender_native_tap(m, ctx, q_tok)
    pred = probe(adapter(tap))
    target = torch.randn(2, 3, 12)
    loss = joint_loss(loss_ce, pred, target)
    loss.backward()
    q_proj_grad_nonzero = m.blocks[-1].mixer.q_proj.weight.grad.abs().sum().item() > 0
    _report("smoke 7: JOINT (CE+aux) loss produces nonzero gradient on the backbone's own q_proj "
            "-- CONFOUNDED evidence (CE alone would pass this too); see smoke 8 for the isolated "
            "aux-ONLY claim", q_proj_grad_nonzero)


# Per-arm "state-update-feeding" backbone param names checked by smoke_8's aux-ONLY isolation bar.
# q_proj/q_conv1d are EXCLUDED for EVERY arm (not a per-arm special case): the tap's own dedicated
# query-pathway call shares q_proj's WEIGHT with the main forward's per-token read use, and the
# real fla `chunk_delta_rule(q, k, v, beta, initial_state, ...)` kernel signature takes q as an
# input to the SAME call that produces `final_state` -- this design cannot rule out an internal
# q-dependence inside the closed production kernel without inspecting its internals, so q_proj is
# treated as a permanently confounded parameter for this bar, CPU or box, for every arm. The
# ablation has no k-analog (Hadamard vector state has no key vector) and the Transformer has no
# beta-gate analog (no b_proj) -- each arm's tuple lists only the projections that genuinely exist
# and unambiguously feed that arm's own state/context representation.
_ISOLATION_CHECK_ATTR_PATH = {
    "contender": ("mixer.k_proj", "mixer.v_proj", "mixer.b_proj"),
    "ablation": ("mixer.v_proj", "mixer.g_proj"),
    "transformer": ("attn.k_proj", "attn.v_proj"),
}


def _grad_abs_sums(last_block, attr_paths: tuple[str, ...]) -> dict[str, float | None]:
    """Reads `.weight.grad` off `last_block.<dotted attr path>` for each path in `attr_paths`,
    returning None for an untouched (never-backprop'd) param rather than crashing -- the honest
    way to represent "no gradient reached this parameter" (PyTorch leaves `.grad` as None, it does
    NOT materialize a zero tensor, until backward() actually routes something there)."""
    out: dict[str, float | None] = {}
    for path in attr_paths:
        mod = last_block
        for part in path.split("."):
            mod = getattr(mod, part)
        grad = mod.weight.grad
        out[path] = None if grad is None else grad.abs().sum().item()
    return out


def _all_zero_or_none(sums: dict[str, float | None]) -> bool:
    return all(v is None or v == 0.0 for v in sums.values())


def _all_positive(sums: dict[str, float | None]) -> bool:
    return all(v is not None and v > 0.0 for v in sums.values())


def smoke_8_aux_loss_only_gradient_isolation():
    """AUD-F1 fix (audit §1.20's substantive finding): an AUX-LOSS-ONLY
    (CE-EXCLUDED) gradient-isolation test, checking grad_abs_sum ONLY on
    each arm's unambiguous state-feeding projections (see
    `_ISOLATION_CHECK_ATTR_PATH` above for the per-arm set and the
    q_proj-exclusion rationale). Backprops ONLY `aux_weight *
    probe_aux_loss(...)` -- loss_ce is never constructed here, closing the
    confound smoke_7 could not (loss_ce touching q_proj through the
    stub's own gate, per the audit's exact wording).

    HONEST CPU/BOX SPLIT (the audit's own confound analysis, restated
    precisely): the CPU stub's `_stub_chunk_delta_rule` returns
    `final_state = torch.zeros(..., dtype=q.dtype, device=q.device)` --
    freshly constructed, `requires_grad=False`, with NO autograd history
    connecting it to q/k/v/beta. This kills BOTH halves of the contender's
    tap on CPU: the FORWARD VALUE (`S_T_last @ q_last` is identically the
    zero tensor, a constant, regardless of q_last) AND the GRADIENT (the
    einsum's own local derivative w.r.t. q_last IS `S_T_last`, itself
    zero, so even q_proj's OWN dedicated query-pathway call receives
    exactly zero gradient from this loss on CPU -- not merely k/v/b).
    Because the tap is a CONSTANT on CPU, this isolation test cannot
    exercise the contender's real claim there; the contender arm is
    therefore registered BOX-ONLY below (mirrors smoke_3's box-only
    registration discipline exactly: PASS is reported for what CPU CAN
    prove -- that the stub's disconnection behaves exactly as diagnosed,
    i.e. k/v/b grad is confirmed zero -- with the limitation stated in
    the detail string, never silently skipped); the box-side variant
    (real, differentiable Triton kernel, genuinely nonzero S_T_last) runs
    in the SAME function below, gated on `not _STUB_INSTALLED`, so the
    deploy stage exercises it with zero new wiring (checklist item 3,
    `h2h_box_smoke_checklist.py`).

    Ablation + Transformer arms have NO such stub disconnect -- their
    state recurrence is real, differentiable PyTorch ops even on CPU --
    and are PROVEN genuinely nonzero here (the audit's own verification).

    NEGATIVE TEST (required, run to completion for every arm exercised
    here): the SAME check is repeated with `tap.detach()` fed into the
    probe -- this must drive every checked grad to EXACTLY zero, proving
    the positive result above is measuring the aux path specifically and
    not some unrelated confound (e.g. a stray shared-parameter path)."""
    ok_all = True
    aux_w = AUX_WEIGHT_DEFAULT

    def _aux_only_loss(adapter, probe, tap, target, detach: bool = False):
        fed = tap.detach() if detach else tap
        pred = probe(adapter(fed))
        return aux_w * probe_aux_loss(pred, target)

    # ---- ablation arm (CPU-provable, genuinely nonzero) ----
    torch.manual_seed(41)
    m_abl = AblationLM(300, d_model=32, d_state=16, n_layers=1, conv_size=4)
    ctx = torch.randint(0, 300, (2, _CONTENDER_SMOKE_CTX_LEN))
    q_tok = torch.randint(0, 300, (2, 3, 5))
    adapter_abl = build_adapter_arm(16, 12)
    probe_abl = build_shared_probe(12)
    tap_abl = ablation_native_tap(m_abl, ctx, q_tok)
    target_abl = torch.randn(2, 3, 12)

    _aux_only_loss(adapter_abl, probe_abl, tap_abl, target_abl).backward()
    abl_pos = _grad_abs_sums(m_abl.blocks[-1], _ISOLATION_CHECK_ATTR_PATH["ablation"])
    abl_pos_ok = _all_positive(abl_pos)
    _report("smoke 8a: ablation arm aux-ONLY (CE-excluded) grad reaches v_proj+g_proj, nonzero "
            "(CPU-provable)", abl_pos_ok, f"grad_abs_sums={abl_pos}")
    ok_all = ok_all and abl_pos_ok

    m_abl.zero_grad()
    _aux_only_loss(adapter_abl, probe_abl, tap_abl, target_abl, detach=True).backward()
    abl_neg = _grad_abs_sums(m_abl.blocks[-1], _ISOLATION_CHECK_ATTR_PATH["ablation"])
    abl_neg_ok = _all_zero_or_none(abl_neg)
    _report("smoke 8a-neg: ablation arm NEGATIVE control -- tap.detach() drives v_proj+g_proj grad "
            "to exactly zero (proves the positive result measures the aux path specifically)",
            abl_neg_ok, f"grad_abs_sums={abl_neg}")
    ok_all = ok_all and abl_neg_ok

    # ---- transformer arm (CPU-provable, genuinely nonzero) ----
    torch.manual_seed(43)
    m_tr = TransformerLM(300, d_model=32, n_layers=1, n_heads=4, ffn_mult=2)
    adapter_tr = build_adapter_arm(32, 12)
    probe_tr = build_shared_probe(12)
    tap_tr = transformer_native_tap(m_tr, ctx, q_tok, attn_mask_fn=None)
    target_tr = torch.randn(2, 3, 12)

    _aux_only_loss(adapter_tr, probe_tr, tap_tr, target_tr).backward()
    tr_pos = _grad_abs_sums(m_tr.blocks[-1], _ISOLATION_CHECK_ATTR_PATH["transformer"])
    tr_pos_ok = _all_positive(tr_pos)
    _report("smoke 8b: transformer arm aux-ONLY (CE-excluded) grad reaches k_proj+v_proj, nonzero "
            "(CPU-provable)", tr_pos_ok, f"grad_abs_sums={tr_pos}")
    ok_all = ok_all and tr_pos_ok

    m_tr.zero_grad()
    _aux_only_loss(adapter_tr, probe_tr, tap_tr, target_tr, detach=True).backward()
    tr_neg = _grad_abs_sums(m_tr.blocks[-1], _ISOLATION_CHECK_ATTR_PATH["transformer"])
    tr_neg_ok = _all_zero_or_none(tr_neg)
    _report("smoke 8b-neg: transformer arm NEGATIVE control -- tap.detach() drives k_proj+v_proj "
            "grad to exactly zero", tr_neg_ok, f"grad_abs_sums={tr_neg}")
    ok_all = ok_all and tr_neg_ok

    # ---- contender arm: BOX-ONLY registration (mirrors smoke_3's discipline exactly) ----
    torch.manual_seed(42)
    m_con = DeltaNetLM(300, d_model=32, d_state=64, n_layers=1, conv_size=4)
    adapter_con = build_adapter_arm(64, 12)
    probe_con = build_shared_probe(12)
    tap_con = contender_native_tap(m_con, ctx, q_tok)
    target_con = torch.randn(2, 3, 12)

    _aux_only_loss(adapter_con, probe_con, tap_con, target_con).backward()
    con_sums = _grad_abs_sums(m_con.blocks[-1], _ISOLATION_CHECK_ATTR_PATH["contender"])

    if _STUB_INSTALLED:
        # CPU stub: the tap is a disconnected constant (see docstring) -- the ONLY CPU-provable
        # claim is that k/v/b grad is CONFIRMED exactly zero, i.e. the stub-vacuity diagnosis
        # itself is correct, NOT that the contender's real aux path works. Registered as a
        # box-only follow-up smoke item for the deploy stage, not silently skipped.
        stub_confirms_diagnosis = _all_zero_or_none(con_sums)
        _report(
            "smoke 8c: contender arm aux-ONLY grad isolation -- BOX-ONLY (CPU stub's zero, "
            "disconnected final_state makes S_T_last @ q_last a CONSTANT; k/v/b grad=0.0 here "
            "CONFIRMS that diagnosis, it is NOT a real pass/fail on the aux path itself -- the "
            "real claim requires a nonzero, differentiable S_T_last from the actual Triton "
            "kernel, registered here as a box-only follow-up smoke item for the deploy stage, "
            "see docstring)", stub_confirms_diagnosis, f"grad_abs_sums={con_sums}")
        ok_all = ok_all and stub_confirms_diagnosis
    else:
        # Box-side (real fla): S_T_last is real and differentiable -- run the actual isolation
        # claim, positive AND negative arms, exactly like the CPU-provable arms above.
        con_pos_ok = _all_positive(con_sums)
        _report("smoke 8c (BOX): contender arm aux-ONLY grad reaches k_proj+v_proj+b_proj, "
                "nonzero (real Triton kernel)", con_pos_ok, f"grad_abs_sums={con_sums}")
        ok_all = ok_all and con_pos_ok

        m_con.zero_grad()
        _aux_only_loss(adapter_con, probe_con, tap_con, target_con, detach=True).backward()
        con_neg = _grad_abs_sums(m_con.blocks[-1], _ISOLATION_CHECK_ATTR_PATH["contender"])
        con_neg_ok = _all_zero_or_none(con_neg)
        _report("smoke 8c-neg (BOX): contender arm NEGATIVE control -- tap.detach() drives "
                "k_proj+v_proj+b_proj grad to exactly zero", con_neg_ok, f"grad_abs_sums={con_neg}")
        ok_all = ok_all and con_neg_ok

    _report("smoke 8 (AUD-F1 fix): aux-loss-ONLY (CE-excluded) gradient isolation, all 3 arms "
            "(ablation+transformer CPU-proven; contender box-only-registered)", ok_all)


def smoke_9_rung2_identity_classifier_capacity_null_all_three_arm_shapes():
    """R5-F2 (GATE, sec 1.7 gate 1a rung 2): the identity-classifier's OWN capacity null MUST
    pass (bar <= 1.5x chance) for every arm's REAL rung-1 adapter shape, at the primary load
    K=32, before a real rung-2 PASS is trusted at face value."""
    shapes = {"contender": (64, 11), "ablation": (64, 12), "transformer": (256, 13)}
    K = 32
    all_pass = True
    for arm, (dim, seed_offset) in shapes.items():
        result = run_identity_classifier_capacity_null(dim, K, n_train_steps=300, seed=seed_offset)
        print(f"    arm={arm:>11s}: native_tap_dim={dim:>3d} K={K} accuracy={result['accuracy']:.4f} "
              f"pass_bar={result['pass_bar']:.4f} passed={result['passed']}")
        all_pass = all_pass and result["passed"]
    _report("smoke 9 (R5-F2 GATE): rung-2 identity-classifier capacity null PASSES for all 3 "
            "arms' real rung-1 adapter shapes (bar <= 1.5x chance, strictly separated from the "
            "real 3x gate)", all_pass)


def smoke_10_rung2_planted_signal_positive_control_both_directions():
    """sec 1.31.4 item 2 (GATE): the positive-control half must actually have teeth in BOTH
    directions it claims -- (a) a perfectly-linear planted identity signal is recovered
    near-ceiling, proving the classifier/training loop CAN read a real signal (never trivially
    weak); (b) the OLD tgt_slot-style labels read ~chance on that SAME tap, proving sec 1.31.4
    item 1's relabel is not a cosmetic change -- the old construction really was structurally
    vacuous. Run for a representative tap_dim (256, the Leg-B tap's own d_model) and the real
    107-ish class count is NOT re-derived here (that requires the real GPT-2 pool, box-decoupled
    on purpose); a stand-in n_classes=107 is used, matching the design doc's own cited pool size."""
    result = rung2_planted_signal_positive_control(tap_dim=256, n_classes=107, seed=21)
    _report("smoke 10 (sec 1.31.4 item 2 GATE): planted-signal positive control passes BOTH "
            "directions (identity recovered >=90%; OLD slot-labeled construction on the SAME "
            "plant reads <=1.5x chance)", result["passed"],
            f"identity_accuracy={result['identity_accuracy']:.4f} "
            f"old_slot_labeled_accuracy={result['old_slot_labeled_accuracy']:.4f} "
            f"old_slot_labeled_chance={result['old_slot_labeled_chance']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    print("=" * 70)
    print("probe_head_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3.1 smoke suite")
    print(f"fla stub installed (real fla absent or forced): {_STUB_INSTALLED}")
    print("=" * 70)
    smoke_1_target_table_frozen_unit_rows()
    smoke_2_contender_tap_forward_backward()
    smoke_3_contender_tap_p1_bottleneck_blankout()
    smoke_4_ablation_tap_forward_and_bottleneck()
    smoke_5_transformer_tap_uncapped_vs_capped()
    smoke_6_probe_capacity_null_all_three_arm_shapes()
    smoke_7_joint_loss_gradients_flow_to_backbone()
    smoke_8_aux_loss_only_gradient_isolation()
    smoke_9_rung2_identity_classifier_capacity_null_all_three_arm_shapes()
    smoke_10_rung2_planted_signal_positive_control_both_directions()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
