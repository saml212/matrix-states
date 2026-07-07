"""reasoning_link_probe.py -- REASONING_LINK_DESIGN.md (Rev 6,
CLEARED-FOR-BUILD) Phase 1 shared instrument. Implements sec 4 (episode
construction adaptation + the two-forward Option-1/Option-2 readout),
sec 5.2a's blend-toggle surgery, sec 8 (metrics/thresholds/premises), and
the pure, mechanical outcome-routing gates of sec 12 -- NOT the harvest-time
multi-cell aggregation across arms/rungs/K/h (that is a later analysis
script's job once real checkpoint numbers exist; this file provides the
per-cell measurement + the reusable statistical primitives a harvest script
calls, per this build's own three-file scope: probe / Stage -1 selftests /
chain).

Two legs share this ONE instrument (sec 3): Leg A (frozen-bias intervention,
`/data/deltanet_rd_frozenbias_ckpts/`) and Leg B (Track C scale ladder,
`/data/lm_rd_trackc_ckpts/`). Zero new training -- eval-only, hooked forward
passes over already-archived checkpoints.

**Design contradictions/underspecified points resolved during BUILD (recorded
here, not silently assumed -- see the BUILD agent's own report for the full
list):**

1. **Which layer's `S_T`/`k_eff`/`v_eff`/`q_eff` realize the PRIMARY Option-1
   readout, for multi-layer checkpoints.** Sec 4.4 states only that `q_eff`
   is "extracted from the FINAL layer" (the framing-adjudication paragraph);
   it never states this as a single explicit sentence for `S_T`/`k_eff`/
   `v_eff`. Resolved by the design's own "single-layer state self-iteration"
   framing (`S_T` from ONE layer, matrix-powered against THAT SAME layer's
   own query) -- if `q_eff` is the final layer's, `S_T`/`k_eff`/`v_eff` must
   be the SAME (final) layer's for the "same layer" framing to hold, and for
   the `Im(S_T) subseteq span{v_eff_j}` family argument to stay well-typed.
   `READOUT_LAYER_INDEX = n_layers - 1` below implements this reading. Every
   layer's own k/v/q/S_T is still captured (hooks fire on every layer, zero
   extra cost) and exposed for optional per-layer exploratory diagnostics,
   clearly separate from the primary readout.
2. **Premises (i)-(ii)'s exact formula.** Sec 4.4/sec 9 cite these as
   "`DELTANET_REALDATA_DESIGN.md` sec 5.2, R2-2" without restating the
   formula in THIS document. Cross-checked against that design directly
   (sec 5.2/C16): premise (i) = `gram_deviation(normalize(k_eff_items))`,
   premise (ii) = the SAME instrument applied to the value side,
   `gram_deviation(normalize(v_eff_items))` -- both reuse `model_rd.py`'s
   own, already-audited `gram_deviation` function verbatim (imported below),
   never reimplemented.
3. **The label-shuffle null's exact construction.** Sec 4.5/8.4 describe it
   in prose ("the bind-clause (key,value) pairing shuffled across
   episodes... only which value follows which key is scrambled") without a
   literal formula. Implemented here as: keep the REAL rendered episode
   (real `succ`, real token stream) from a normal `sample_batch_rd` draw,
   but grade `prev_slot`/`v_eff_target` using an INDEPENDENTLY-drawn second
   Hamiltonian K-cycle (`succ_null`, via the verbatim `_permutation_graph`)
   -- i.e., the SAME forward pass's real `pred(a,h)` is compared against a
   deliberately WRONG target. This satisfies "surface form / entity pool /
   token statistics held fixed, only the (key,value) pairing used for
   grading is scrambled" literally, needs no extra forward pass (reuses the
   real cell's own captured tensors), and destroys true compositional
   structure by construction (the grading target has no relationship to
   what was actually written into the state).
4. **The query-marker fallback phrase.** Sec 4.1 item 2 registers a
   multi-token phrase fallback ("e.g. ' then who ?'") ONLY if no
   single-token candidate qualifies. Verified this build session against
   the real GPT-2 tokenizer: "%c2%a7" (section sign) and "%c2%b6" (pilcrow)
   ARE both single GPT-2 BPE tokens, collision-free against the entity/
   relation/period pools -- so the multi-token phrase path is NOT built
   (the single-token path covers sec 7.6's two-marker robustness replicate
   with margin: 5 more single-token candidates verified). If a future
   tokenizer version invalidates this, `choose_two_markers` raises loudly
   rather than silently falling back to an unbuilt code path.

None of these are contradictions in the sense the BUILD brief asks to STOP
on -- each is a textually well-supported completion of an underspecified
(but not self-contradictory) point, recorded here for audit rather than
silently assumed.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports, matches every sibling script in this dir

DESIGN_REF = "REASONING_LINK_DESIGN.md Rev 6 (2026-07-07, CLEARED-FOR-BUILD)"

# ---------------------------------------------------------------------------
# fla stub -- installed ONLY if the real package is not importable (dev
# machine). Cloned from frozen_bias_gradflow_probe.py's own stub (the
# k-ROUTING FIXED version: `o` genuinely depends on `k` via a cheap
# `sigmoid((q*k).sum(-1))` gate, unlike the sibling smoke files'
# `o = v * sigmoid(beta)`-only stub, which has no k-dependency at all and
# would make any k-gradient/attention-to-k check pass vacuously). This
# probe's own Stage -1 items never need backward/gradients, but the SAME
# fixed stub is reused (not the broken sibling) per this build's explicit
# instruction -- consistency with the codebase's one "good" stub, and it
# costs nothing extra. MUST run BEFORE `from lm_pretrain_rd import ...`
# below (that import executes lm_pretrain_rd.py's own top-level
# `from fla... import` lines immediately).
# ---------------------------------------------------------------------------

def _ensure_fla_stub() -> bool:
    # LAUNCH FIX (2026-07-07, first box launch attempt): the Stage -1 suite
    # is a CPU-fp32 suite BY REGISTRATION (design sec 9's tolerances are
    # fp32/CPU), and its toy models live on CPU. On the box the REAL fla
    # package imports fine, so without this override items 6/11/13/14 route
    # CPU tensors into the real Triton chunk_delta_rule -> "Pointer argument
    # cannot be accessed from Triton (cpu tensor?)". Setting
    # REASONING_LINK_FORCE_CPU_STUB=1 (chain step 1 does) installs this
    # file's own CPU stub even when real fla exists, preserving the EXACT
    # audited selftest semantics (both independent audits ran the suite
    # through this stub locally). The real kernel is still exercised for
    # real at Stage 0, which re-runs the item-11 causality assertion on a
    # real checkpoint per sec 9's own purpose (a).
    if os.environ.get("REASONING_LINK_FORCE_CPU_STUB", "0") != "1":
        try:
            import fla  # noqa: F401
        # MINOR-3 fix (this audit round) -- the double-execution misreport. When this file runs
        # as `__main__` (e.g. `python reasoning_link_probe.py --mode selftest`) AND is separately
        # imported by module name (`reasoning_link_stage_minus1.py`'s own `import
        # reasoning_link_probe as rlp`), Python loads it as TWO DISTINCT module objects
        # (`sys.modules["__main__"]` and `sys.modules["reasoning_link_probe"]`), each running this
        # top-level code independently. The __main__ execution installs the stub into
        # `sys.modules["fla"]`; the SECOND execution's `import fla` above then succeeds trivially
        # (it just finds the first execution's stub already sitting in sys.modules), which without
        # this check would make the second module instance's `FLA_STUB_INSTALLED` read `False` --
        # falsely reporting "the real fla package was found" when what is actually installed is
        # still this file's own CPU stub. Detect that case by a marker attribute on the module
        # object itself (never by import success/failure alone, which cannot distinguish "real fla"
        # from "a stub some earlier execution already installed").
            if getattr(fla, "_REASONING_LINK_CPU_STUB", False):
                return True
            return False
        except ImportError:
            pass

    class _StubShortConvolution(nn.Module):
        def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False,
                     activation: str | None = "silu"):
            super().__init__()
            self.activation = activation
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                                   padding=kernel_size - 1, bias=bias)

        def forward(self, x: torch.Tensor, cache=None):
            B, T, D = x.shape
            out = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
            if self.activation == "silu":
                out = F.silu(out)
            return out, None

    class _StubRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    def _stub_chunk_delta_rule(q, k, v, beta, initial_state=None, output_final_state=True,
                                use_qk_l2norm_in_kernel=True):
        """CPU-safe stand-in for the CUDA-only Triton kernel. `o` genuinely
        depends on `k` (via a cheap per-position gate) so any hook-based
        check that expects the mixer's output to be causally/functionally
        connected to `k` is exercised meaningfully. `final_state` is a
        (deliberately) zero matrix -- box-only-verifiable items depending on
        a NONZERO, trained `S_T` (the real recurrent state) cannot be given
        meaningful numbers by this stub; every Stage -1 item that needs a
        real nonzero S_T is disclosed as box-only in this build's report."""
        B, T, H, Dh = q.shape
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        qk_gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        o = v * torch.sigmoid(beta).unsqueeze(-1) * qk_gate
        final_state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype, device=q.device)
        return o, final_state

    fla_mod = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_delta_rule = types.ModuleType("fla.ops.delta_rule")
    fla_modules.ShortConvolution = _StubShortConvolution
    fla_modules.RMSNorm = _StubRMSNorm
    fla_ops_delta_rule.chunk_delta_rule = _stub_chunk_delta_rule
    fla_mod.modules = fla_modules
    fla_mod.ops = fla_ops
    fla_ops.delta_rule = fla_ops_delta_rule
    fla_mod._REASONING_LINK_CPU_STUB = True  # MINOR-3 fix's own marker -- see the read-side check above
    sys.modules["fla"] = fla_mod
    sys.modules["fla.modules"] = fla_modules
    sys.modules["fla.ops"] = fla_ops
    sys.modules["fla.ops.delta_rule"] = fla_ops_delta_rule
    return True


FLA_STUB_INSTALLED = _ensure_fla_stub()

import grammar_rd  # noqa: E402
from lm_pretrain_rd import DeltaNetLM, FROZEN_BIAS_ARM_MODES, _MIN_KERNEL_T  # noqa: E402
# (apply_frozen_bias_blend/build_frozen_bias_table/frozen_bias_global_vector/_SAFE_D_STATE are NOT
# imported here: sec 5.2a's surgery toggle works by flipping the LOADED model's own
# `frozen_bias_arm` attribute [see frozen_bias_surgery() below] and letting the model's OWN
# forward() call these functions internally -- this probe never calls them directly. They ARE
# imported directly by reasoning_link_stage_minus1.py's own item 6, which DOES need to call
# apply_frozen_bias_blend directly to reproduce forward()'s pre-kernel sequence for its two
# negative controls.)
from model_rd import gram_deviation  # noqa: E402

# ---------------------------------------------------------------------------
# Constants -- every threshold/sweep value pre-registered in the design,
# reproduced here as literal, checkable constants (never re-derived by eye).
# ---------------------------------------------------------------------------

H_TEST = (1, 2, 3, 4)                    # sec 4.3
CI_T_975_DF2 = 4.303                     # t(2,.975), this project's standing n=3-seed CI formula
RECOVERY_COS_THRESHOLD = 0.9             # sec 8.1
H1_ABSOLUTE_FLOOR = 0.10                 # sec 8.4 MAJOR-2

BATCH_SIZE_DEFAULT = 16                  # sec 4.2, pinned

# sec 4.3's K sweep, per leg / per d_state (committed vs named extension).
K_SWEEP = {
    "leg_a": {64: {"committed": (20, 32), "extension": (40, 48)}},
    "leg_b": {
        64: {"committed": (32,), "extension": (20, 40, 48)},
        128: {"committed": (64,), "extension": (20, 32, 96)},
    },
}
LEGAL_K_BY_DSTATE = {64: (20, 32, 40, 48), 128: (20, 32, 64, 96)}

# sec 4.6 -- exact numeric seed-allocation formula, reproduced verbatim.
PURPOSE_BASE = {"eval": 0, "null_shuffle": 10_000_000, "calibration": 20_000_000}
LEG_BASE = {"leg_a": 0, "leg_b": 5_000_000}
STRIDE_CONDITION = 1_000_000
STRIDE_CORPUS = 100_000
STRIDE_SEED = 10_000
STRIDE_K = 1_000
# "k_idx ... over the registered K's {20,32,40,48,64,96}" -- sec 4.6's own listed order.
K_INDEX = {20: 0, 32: 1, 40: 2, 48: 3, 64: 4, 96: 5}
CORPUS_INDEX = {"openr1-mix-ext": 0, "wikitext-mix-ext": 1}
LEG_A_ARM_INDEX = {"off": 0, "per_token": 1, "global": 2}
# (Leg B's rung condition_idx is already the plain int 0-3 the CLI's --rung takes directly, and
# LEG_B_RUNG_CFG below maps that same int -> name/architecture -- no separate name->index table needed.)

FROZEN_BIAS_CKPT_ROOT_DEFAULT = "/data/deltanet_rd_frozenbias_ckpts"
TRACKC_CKPT_ROOT_DEFAULT = "/data/lm_rd_trackc_ckpts"

# sec 4.1 item 2: candidates tried in order; first single-token, collision-free
# survivor is the primary marker, second survivor is sec 7.6's robustness replicate.
QUERY_MARKER_CANDIDATES = ["§", "¶", "µ", "~", "^", "`", "\\"]
QUERY_MARKER_FALLBACK_PHRASE = " then who ?"  # registered, NOT built this phase -- see module docstring

LEG_B_RUNG_CFG = {
    0: {"name": "14M", "tag": "mixcontrol", "d_model": 256, "d_state": 64, "n_layers": 2},
    1: {"name": "98M", "tag": "wave1ext", "d_model": 768, "d_state": 64, "n_layers": 12},
    2: {"name": "392M", "tag": "wave2", "d_model": 1536, "d_state": 128, "n_layers": 16},
    3: {"name": "1.31B", "tag": "wave3", "d_model": 2560, "d_state": 128, "n_layers": 22},
}


# ---------------------------------------------------------------------------
# sec 4.6 -- episode-seed allocation (exact formula, collision-free by
# mixed-radix construction; Stage -1 item 10 mechanically verifies this).
# ---------------------------------------------------------------------------

def episode_seed(purpose: str, leg: str, condition_idx: int, corpus_idx: int,
                  ckpt_seed_idx: int, k_idx: int) -> int:
    assert purpose in PURPOSE_BASE, f"unknown purpose {purpose!r}"
    assert leg in LEG_BASE, f"unknown leg {leg!r}"
    assert 0 <= condition_idx <= 3, condition_idx
    assert 0 <= corpus_idx <= 1, corpus_idx
    assert 0 <= ckpt_seed_idx <= 2, ckpt_seed_idx
    assert 0 <= k_idx <= 5, k_idx
    return (PURPOSE_BASE[purpose] + LEG_BASE[leg]
            + condition_idx * STRIDE_CONDITION
            + corpus_idx * STRIDE_CORPUS
            + ckpt_seed_idx * STRIDE_SEED
            + k_idx * STRIDE_K)


def enumerate_registered_seed_combinations() -> list[tuple]:
    """Every (purpose, leg, condition_idx, corpus_idx, ckpt_seed_idx, k_idx)
    combination the committed grid + named extensions actually instantiate
    (sec 4.3/sec 10) -- used by Stage -1 item 10's non-collision assertion.
    Pure Python enumeration, no checkpoints/GPU involved."""
    combos = []
    # Leg A: eval purpose, 3 arms x 2 corpora x 3 seeds x {20,32,40,48}
    for arm_idx in range(3):
        for corpus_idx in range(2):
            for seed_idx in range(3):
                for k in (20, 32, 40, 48):
                    combos.append(("eval", "leg_a", arm_idx, corpus_idx, seed_idx, K_INDEX[k]))
    # Leg B: eval purpose, 4 rungs. Rungs 0-2 have 2 corpora x 3 seeds; rung 3 has 2 corpora x 1 seed
    # (sec 6.1's PINNED rung-3 configuration). K's per d_state per sec 4.3.
    for rung_idx in range(4):
        d_state = 64 if rung_idx < 2 else 128
        n_seeds = 3 if rung_idx < 3 else 1
        for corpus_idx in range(2):
            for seed_idx in range(n_seeds):
                for k in LEGAL_K_BY_DSTATE[d_state]:
                    combos.append(("eval", "leg_b", rung_idx, corpus_idx, seed_idx, K_INDEX[k]))
    # null_shuffle: Stage 0's single calibration cell only (condition=off=0, corpus idx 0, seed 0, K=32)
    combos.append(("null_shuffle", "leg_a", 0, 0, 0, K_INDEX[32]))
    # calibration: Stage 0 (condition_idx=0) and Stage 0.5 (condition_idx=3, leg_b, K=64)
    combos.append(("calibration", "leg_a", 0, 0, 0, K_INDEX[32]))
    combos.append(("calibration", "leg_b", 3, 0, 0, K_INDEX[64]))
    return combos


# ---------------------------------------------------------------------------
# sec 4.1 -- per-checkpoint episode config, the K-floor gate.
# ---------------------------------------------------------------------------

def clause_len_for_conv_size(conv_size: int) -> int:
    return max(1, conv_size - 1) + 4


def k_min_for_conv_size(conv_size: int, min_kernel_t: int = _MIN_KERNEL_T) -> int:
    """sec 4.1's registered formula: K_min(conv_size) = ceil(_MIN_KERNEL_T / clause_len(conv_size))."""
    return math.ceil(min_kernel_t / clause_len_for_conv_size(conv_size))


def episode_config_for_checkpoint(conv_size: int, K: int) -> grammar_rd.DeltaNetRDTaskConfig:
    """sec 4.1 Rev 1's registered commitment: a FRESH DeltaNetRDTaskConfig per
    checkpoint, conv_size read from THAT checkpoint's own saved config, never
    a shared cfg reused across checkpoints whose conv_size could differ.
    H_train=(5,)/H_test=(1,2,3,4)/H_extra=() per sec 4.3's registered,
    semantically-inert placeholder convention (5 < min(K)=20 across every
    swept K, never colliding mod K)."""
    k_min = k_min_for_conv_size(conv_size)
    assert K >= k_min, (
        f"K={K} < K_min(conv_size={conv_size})={k_min} -- T_bind={K * clause_len_for_conv_size(conv_size)} "
        f"would be below chunk_delta_rule's _MIN_KERNEL_T={_MIN_KERNEL_T} floor (sec 4.1's FATAL-2 fix).")
    return grammar_rd.DeltaNetRDTaskConfig(K=K, conv_size=conv_size, H_train=(5,), H_test=H_TEST, H_extra=())


# ---------------------------------------------------------------------------
# sec 4.1 -- query-marker verification + the pools substitution (buffer ->
# period token, <Q> -> a verified ordinary single-token marker). Reuses
# grammar_rd.build_entity_pools()/_permutation_graph()/sample_batch_rd()
# VERBATIM -- the adaptation is realized entirely as a data substitution on
# the returned EntityPools dataclass, never a code change to grammar_rd.py.
# ---------------------------------------------------------------------------

def verify_query_marker_candidates(tokenizer, pools: grammar_rd.EntityPools,
                                    candidates=QUERY_MARKER_CANDIDATES) -> dict:
    used_ids = (set(pools.train_name_ids.tolist()) | set(pools.heldout_name_ids.tolist())
                | set(pools.rel_a_ids.tolist()) | set(pools.rel_b_ids.tolist()) | {pools.period_id})
    tried = []
    for cand in candidates:
        ids = tokenizer.encode(" " + cand)
        entry = {"candidate": cand, "ids": ids}
        if len(ids) != 1:
            entry["status"] = "multi-token"
        else:
            tid = ids[0]
            decoded = tokenizer.decode([tid])
            if decoded != (" " + cand):
                entry["status"] = "decode-mismatch"
            elif tid in used_ids:
                entry["status"] = "collides-with-entity-or-relation-or-period"
            else:
                entry["status"] = "OK"
                entry["token_id"] = tid
        tried.append(entry)
    ok = [e for e in tried if e["status"] == "OK"]
    return {"candidates_tried": tried, "n_ok": len(ok), "ok_candidates": ok}


def choose_two_markers(tokenizer, pools: grammar_rd.EntityPools) -> tuple[dict, dict, dict]:
    """sec 7.6's two-marker robustness replicate: the first two independently
    single-token, collision-free candidates. Raises loudly (never silently
    falls back to the unbuilt multi-token-phrase path, see module docstring
    point 4) if fewer than 2 qualify."""
    report = verify_query_marker_candidates(tokenizer, pools)
    if report["n_ok"] < 2:
        raise RuntimeError(
            "reasoning_link_probe: fewer than 2 single-token, collision-free query-marker "
            f"candidates qualified (tried {QUERY_MARKER_CANDIDATES}) -- sec 4.1 item 2's "
            f"multi-token fallback phrase ({QUERY_MARKER_FALLBACK_PHRASE!r}) is REGISTERED "
            f"but NOT BUILT this phase (see module docstring point 4). STOP and report; do "
            f"not silently improvise a phrase-marker code path.")
    return report["ok_candidates"][0], report["ok_candidates"][1], report


def build_reasoning_link_pools(tokenizer=None, heldout_frac: float = 0.5, seed: int = 0,
                                marker_word: str | None = None) -> tuple[grammar_rd.EntityPools, dict]:
    """sec 4.1's registered adaptation. Returns (pools, report). `pools` is a
    real `grammar_rd.EntityPools` -- `sample_batch_rd`/`self_query_tokens`
    consume it with ZERO code changes; only `buffer_id` (-> the tokenizer's
    own period token, repeated conv_size-1 times between clauses by
    `sample_batch_rd`'s existing buf_len-driven layout) and `query_id` (-> a
    verified ordinary single-token marker) differ from grammar_rd's own
    reserved-out-of-vocab convention."""
    if tokenizer is None:
        tokenizer = grammar_rd.load_gpt2_tokenizer()
    base_pools, base_report = grammar_rd.build_entity_pools(tokenizer, heldout_frac=heldout_frac, seed=seed)
    if marker_word is None:
        marker_a, marker_b, marker_report = choose_two_markers(tokenizer, base_pools)
        chosen_word, chosen_id = marker_a["candidate"], marker_a["token_id"]
    else:
        ids = tokenizer.encode(" " + marker_word)
        assert len(ids) == 1, f"marker_word {marker_word!r} must be single-token, got {ids}"
        chosen_word, chosen_id = marker_word, ids[0]
        marker_report = None
    rl_pools = dataclasses.replace(base_pools, buffer_id=int(base_pools.period_id), query_id=int(chosen_id),
                                    vocab_size_total=int(base_pools.vocab_size_base))
    report = {
        "base_entity_pool_report": base_report, "marker_word": chosen_word, "marker_id": int(chosen_id),
        "marker_selection_report": marker_report,
        "buffer_substitution": "period_id reused as buffer_id (sec 4.1 item 1)",
        "vocab_size_total": rl_pools.vocab_size_total,
    }
    return rl_pools, report


# ---------------------------------------------------------------------------
# Hooks + the slim forward body (no full-sequence LM head -- CLAUDE.md's own
# hard rule + sec 4.4's explicit registration: "never the full-sequence LM
# head over every position, which would materialize the standing-known
# 50,257-vocab logits tensor at every token").
# ---------------------------------------------------------------------------

def register_kqv_hooks(model: DeltaNetLM):
    """Registers one forward hook per layer on k_conv1d, v_conv1d, AND
    q_conv1d (Rev 4's v_conv1d hook + Rev 5's q_conv1d hook, both ~5-line
    copies of lm_attractor_probe_rd.py's own capture_raw_keys pattern).
    Returns (handles, captured) where captured[kind][layer_idx] is the
    (B,T,d_state) raw post-conv tensor from the LAST forward call."""
    captured = {"k": {i: None for i in range(model.n_layers)},
                "v": {i: None for i in range(model.n_layers)},
                "q": {i: None for i in range(model.n_layers)}}
    handles = []

    def make_hook(kind, i):
        def hook(module, inp, out):
            raw = out[0] if isinstance(out, tuple) else out
            captured[kind][i] = raw.detach()
        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.mixer.k_conv1d.register_forward_hook(make_hook("k", i)))
        handles.append(blk.mixer.v_conv1d.register_forward_hook(make_hook("v", i)))
        handles.append(blk.mixer.q_conv1d.register_forward_hook(make_hook("q", i)))
    return handles, captured


def remove_hooks(handles) -> None:
    for h in handles:
        h.remove()


def forward_body(model: DeltaNetLM, token_ids: torch.Tensor, initial_states=None,
                  step: int | None = None, need_hidden: bool = False):
    """Reproduces `DeltaNetLM.forward()`'s own body (embed -> blocks loop ->
    optionally norm_f) WITHOUT the unconditional `F.linear(x, embed.weight)`
    full-vocab projection that method always computes -- forward-A never
    needs logits at all (T_bind positions x 50,257 vocab would be wasted,
    multi-GB, VRAM-bottleneck compute per CLAUDE.md's own hard rule); Option
    2 (sec 4.4) needs the LM head applied at exactly ONE position per row,
    which the CALLER does separately (see `option2_margin` below) --
    computing it inside this function for every position would reintroduce
    exactly the bottleneck this split is designed to avoid. This calls the
    SAME real `model.embed`/`model.blocks[i]`/`model.norm_f` submodules the
    class method would -- no invented computation, only a deferred/narrowed
    LM-head application, which is sec 4.4's own explicit registration."""
    B, T = token_ids.shape
    x = model.embed(token_ids)
    if initial_states is None:
        initial_states = [None] * model.n_layers
    assert len(initial_states) == model.n_layers
    final_states = []
    for blk, s0 in zip(model.blocks, initial_states):
        x, s_final = blk(x, initial_state=s0, token_ids=token_ids, step=step)
        final_states.append(s_final)
    if need_hidden:
        x = model.norm_f(x)
        return x, final_states
    return None, final_states


@contextlib.contextmanager
def frozen_bias_surgery(model: DeltaNetLM, force_off: bool):
    """sec 5.2a's surgery toggle: `frozen_bias_arm` is a plain Python string
    INSTANCE attribute (lm_pretrain_rd.py, `self.frozen_bias_arm = ...` in
    `DeltaNetLMMixer.__init__`), overwritable post-`load_state_dict` WITHOUT
    touching any trained weight -- the exact fact
    `frozen_bias_retrofit_eval_rd.py`'s own mode dispatch already proves
    code-path-equivalent to the model's live blend (its own smoke item [1]).
    Restores the ORIGINAL per-block value on exit (even on exception) so a
    caller can toggle surgery on/off across repeated calls on one loaded
    model without reloading."""
    if not force_off:
        yield
        return
    originals = [blk.mixer.frozen_bias_arm for blk in model.blocks]
    try:
        for blk in model.blocks:
            blk.mixer.frozen_bias_arm = "off"
        yield
    finally:
        for blk, orig in zip(model.blocks, originals):
            blk.mixer.frozen_bias_arm = orig


def load_checkpoint(path: str, device: str) -> tuple[DeltaNetLM, dict]:
    """Clones lm_attractor_probe_rd.py's/frozen_bias_retrofit_eval_rd.py's own
    load_checkpoint pattern exactly (pod-safety: duplicated, not cross-
    imported)."""
    ckpt = torch.load(path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def readout_layer_index(model: DeltaNetLM) -> int:
    """See module docstring point 1: the primary Option-1 readout uses the
    FINAL layer's S_T/k_eff/v_eff/q_eff (matches sec 4.4's explicit "q_eff
    extracted from the FINAL layer" and the "single-layer state
    self-iteration" same-layer framing)."""
    return model.n_layers - 1


# ---------------------------------------------------------------------------
# Forward-A (bind-only) / Forward-B (bind+query) -- sec 4.4's two-forward
# protocol.
# ---------------------------------------------------------------------------

def run_forward_a(model: DeltaNetLM, bind_token_ids: torch.Tensor, surgery_off: bool = False):
    """Bind-phase-only forward. Returns (final_states: list[Tensor or None],
    k_raw_by_layer: dict[int, Tensor], v_raw_by_layer: dict[int, Tensor]).
    final_states[i] is fla's own (B,H,head_dim,head_dim) layout; k/v_raw are
    (B,T_bind,d_state) pre-reshape, pre-blend-irrelevant (v is never
    touched by the blend at all; k is captured PRE-blend by construction,
    sec 4.4's own disclosed strength, sec 7.9)."""
    handles, captured = register_kqv_hooks(model)
    try:
        with torch.no_grad(), frozen_bias_surgery(model, surgery_off):
            _, final_states = forward_body(model, bind_token_ids, need_hidden=False)
    finally:
        remove_hooks(handles)
    return final_states, captured["k"], captured["v"]


def run_forward_b(model: DeltaNetLM, concat_token_ids: torch.Tensor, need_option2_hidden: bool = False,
                   surgery_off: bool = False):
    """Bind+query-concatenated forward. Returns (hidden_or_None, final_states,
    q_raw_by_layer). `hidden` (if requested) is (B, T, d_model) POST-norm_f,
    PRE-LM-head -- Option 2's own caller slices the last position only."""
    handles, captured = register_kqv_hooks(model)
    try:
        with torch.no_grad(), frozen_bias_surgery(model, surgery_off):
            hidden, final_states = forward_body(model, concat_token_ids, need_hidden=need_option2_hidden)
    finally:
        remove_hooks(handles)
    return hidden, final_states, captured["q"]


def causality_check(model: DeltaNetLM, bind_token_ids: torch.Tensor, query_tokens_one: torch.Tensor,
                     tol: float = 1e-6) -> dict:
    """Stage -1 item 11: forward-A (bind-only) and forward-B (bind+query
    concatenated) must produce IDENTICAL k/v_conv1d outputs over the shared
    BIND-phase prefix -- the mechanical proof that appending the query in
    forward-B never leaks backward into the captured S_T. bind_token_ids:
    (B,T_bind); query_tokens_one: (B,query_len) (ONE query row per episode,
    enough to exercise the check)."""
    _, k_a, v_a = run_forward_a(model, bind_token_ids)
    concat = torch.cat([bind_token_ids, query_tokens_one], dim=1)
    _, _, q_b_raw = run_forward_b(model, concat, need_option2_hidden=False)
    # re-run forward-A style hooks alongside forward-B to also capture k/v over
    # the SAME forward-B call, for the prefix comparison.
    handles, captured_b = register_kqv_hooks(model)
    try:
        with torch.no_grad():
            _ = forward_body(model, concat, need_hidden=False)
    finally:
        remove_hooks(handles)
    T_bind = bind_token_ids.shape[1]
    per_layer = {}
    max_abs_diff = 0.0
    for i in range(model.n_layers):
        k_prefix_b = captured_b["k"][i][:, :T_bind, :]
        v_prefix_b = captured_b["v"][i][:, :T_bind, :]
        k_diff = (k_a[i] - k_prefix_b).abs().max().item()
        v_diff = (v_a[i] - v_prefix_b).abs().max().item()
        per_layer[i] = {"k_max_abs_diff": k_diff, "v_max_abs_diff": v_diff}
        max_abs_diff = max(max_abs_diff, k_diff, v_diff)
    return {"per_layer": per_layer, "max_abs_diff": max_abs_diff, "tol": tol, "pass": max_abs_diff <= tol}


# ---------------------------------------------------------------------------
# Gather / pred(a,h) / scoring.
# ---------------------------------------------------------------------------

def gather_at_positions(raw: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """raw: (B,T,d); positions: (B,K) int64 -> (B,K,d)."""
    d = raw.shape[-1]
    idx = positions.unsqueeze(-1).expand(-1, -1, d)
    return torch.gather(raw, 1, idx)


def squeeze_state_head(final_state: torch.Tensor) -> torch.Tensor:
    """fla's own (B,H,head_dim,head_dim) layout -> (B,d_state,d_state) for
    num_heads==1 (every registered checkpoint in this design, sec 0/sec 6.1).
    Hard-asserts num_heads==1 rather than silently summing/averaging heads
    -- a future num_heads>1 checkpoint needs an explicit, examined decision
    about per-head vs pooled state, not a silent default."""
    B, H, Dh, _ = final_state.shape
    assert H == 1, (
        f"squeeze_state_head: num_heads={H} != 1 -- every checkpoint this design registers "
        f"(sec 0/6.1) uses num_heads=1; a genuine multi-head checkpoint needs an explicit "
        f"per-head-vs-pooled decision before this function can handle it.")
    return final_state[:, 0, :, :]


def apply_state_power(S_T: torch.Tensor, q_eff: torch.Tensor, h: int) -> torch.Tensor:
    """pred(a,h) = S_T^h @ q_eff_a, plain iterative matrix-vector application
    (h<=4 always, per sec 4.3 -- cheap). S_T: (B,d,d); q_eff: (B,Q,d).
    Returns (B,Q,d)."""
    assert h >= 0
    pred = q_eff
    for _ in range(h):
        pred = torch.einsum('bij,bqj->bqi', S_T, pred)
    return pred


def cosine_and_recovered(pred: torch.Tensor, target: torch.Tensor,
                          threshold: float = RECOVERY_COS_THRESHOLD) -> tuple[torch.Tensor, torch.Tensor]:
    """sec 8.1: recovered_frac@0.9 = fraction with cos(pred, target) > 0.9,
    ABSOLUTE cosine (never argmax/nearest-neighbor over a codebook --
    CLAUDE.md's own hard rule, satisfied by construction here)."""
    cos = F.cosine_similarity(pred, target, dim=-1)
    recovered = cos.abs() > threshold
    return cos, recovered


def compute_prev_slot_and_target(succ: torch.Tensor, a_slot: torch.Tensor, hops: torch.Tensor,
                                  v_eff_items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """sec 4.4's off-by-one-hop fix, reproduced EXACTLY (not tgt_slot):
    prev_slot = _iterate_permutation(succ, a_slot, hops-1);
    v_eff_target = gather(v_eff_items, 1, prev_slot)."""
    prev_slot = grammar_rd._iterate_permutation(succ, a_slot, hops - 1)
    v_eff_target = gather_at_positions(v_eff_items, prev_slot)
    return prev_slot, v_eff_target


def build_null_labeling(succ_true: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """sec 4.5/8.4's probe-free label-shuffle null (module docstring point
    3): an INDEPENDENTLY drawn Hamiltonian K-cycle, uncorrelated with
    `succ_true`, used ONLY to compute a deliberately-wrong prev_slot/
    v_eff_target grading target against the SAME real forward pass's own
    pred(a,h). No new episode/forward pass needed."""
    B, K = succ_true.shape
    return grammar_rd._permutation_graph(B, K, gen, succ_true.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Premises (i)-(iv) -- sec 4.4/sec 9's action-rule table.
# ---------------------------------------------------------------------------

def premise_i_ii(k_eff_items: torch.Tensor, v_eff_items: torch.Tensor) -> dict:
    """DELTANET_REALDATA_DESIGN.md sec 5.2/C16, premise (i)/(ii) -- verbatim
    reuse of model_rd.gram_deviation on the L2-normalized key/value
    populations (module docstring point 2). k/v_eff_items: (B,K,d)."""
    gd_k = gram_deviation(F.normalize(k_eff_items, dim=-1))
    gd_v = gram_deviation(F.normalize(v_eff_items, dim=-1))
    return {"premise_i_key_gram_deviation_mean": gd_k.mean().item(),
            "premise_ii_value_gram_deviation_mean": gd_v.mean().item()}


def premise_iii_iv(k_eff_items: torch.Tensor, v_eff_items: torch.Tensor,
                    q_eff_self: torch.Tensor, gen: torch.Generator) -> dict:
    """premise (iii): cos(q_eff_a, k_eff_a) same-entity vs. cos(q_eff_a,
    k_eff_{j!=a}) cross-entity null. premise (iv): cos(k_eff_i, v_eff_i)
    same-entity vs. cos(k_eff_i, v_eff_{j!=i}) cross-entity null.
    q_eff_self: (B,K,d) -- from a self-query pass (grammar_rd.self_query_tokens),
    ALIGNED to k_eff_items/v_eff_items's own slot indexing (same K, same order).
    Cross-entity null uses a fixed random derangement per row (a permutation
    of 0..K-1 with no fixed point) so j != i is guaranteed for every i."""
    B, K, d = k_eff_items.shape
    same_iii = F.cosine_similarity(q_eff_self, k_eff_items, dim=-1)  # (B,K)
    same_iv = F.cosine_similarity(k_eff_items, v_eff_items, dim=-1)  # (B,K)

    derangement = _random_derangement(B, K, gen, k_eff_items.device)
    k_eff_deranged = torch.gather(k_eff_items, 1, derangement.unsqueeze(-1).expand(-1, -1, d))
    v_eff_deranged = torch.gather(v_eff_items, 1, derangement.unsqueeze(-1).expand(-1, -1, d))
    null_iii = F.cosine_similarity(q_eff_self, k_eff_deranged, dim=-1)
    null_iv = F.cosine_similarity(k_eff_items, v_eff_deranged, dim=-1)

    return {"same_entity_cos_iii": same_iii, "null_cos_iii": null_iii,
            "same_entity_cos_iv": same_iv, "null_cos_iv": null_iv}


def _random_derangement(B: int, K: int, gen: torch.Generator, device) -> torch.Tensor:
    """A permutation of 0..K-1 per row with NO fixed point (i != derangement[i]
    for every i) -- the exact "entity i's key vs. a DIFFERENT, randomly
    chosen entity j!=i" construction sec 4.4's MAJOR-1 fix specifies.
    Rejection-sampled per row (cheap: K<=96, derangement probability ~1/e
    per draw, expected <3 retries)."""
    result = torch.empty(B, K, dtype=torch.int64, device=device)
    # LAUNCH FIX 2 (2026-07-07, first Stage-0 box run): torch.randperm with a
    # generator requires the output device to MATCH the generator's device --
    # on the box `gen` is a CUDA generator and the bare call raised "Expected
    # a 'cpu' device type for generator but found 'cuda'". Pass gen.device
    # explicitly (CPU locally, CUDA on box; same draws either way for a given
    # seed on a given device -- the null is device-local, never compared
    # across devices).
    gen_device = gen.device if gen is not None else torch.device("cpu")
    for b in range(B):
        while True:
            perm = torch.randperm(K, generator=gen, device=gen_device).to(device)
            if bool((perm == torch.arange(K, device=device)).any().item()) is False:
                result[b] = perm
                break
    return result


def premise_action_rule(same_iii: torch.Tensor, null_iii: torch.Tensor,
                         same_iv: torch.Tensor, null_iv: torch.Tensor) -> dict:
    """sec 4.4 MAJOR-1's unified action-rule table: median same-entity cosine
    must exceed the 95th percentile of the cross-entity null. No 0.9 magic
    number anywhere in this rule."""
    iii_null_p95 = torch.quantile(null_iii.flatten().float(), 0.95).item()
    iii_median = same_iii.flatten().float().median().item()
    iv_null_p95 = torch.quantile(null_iv.flatten().float(), 0.95).item()
    iv_median = same_iv.flatten().float().median().item()
    iii_pass = iii_median > iii_null_p95
    iv_pass = iv_median > iv_null_p95
    return {
        "premise_iii_median": iii_median, "premise_iii_null_p95": iii_null_p95, "premise_iii_pass": iii_pass,
        "premise_iv_median": iv_median, "premise_iv_null_p95": iv_null_p95, "premise_iv_pass": iv_pass,
        # sec 4.4's table: (iii) fail -> h=1 revoked/exploratory-only; (iv) fail -> h>=2
        # exploratory-only (h=1 retains confirmatory status if (iii) alone passes).
        "h1_confirmatory": iii_pass,
        "h_ge2_confirmatory": iii_pass and iv_pass,
    }


# ---------------------------------------------------------------------------
# sec 8.4 -- h1 sanity floor (null-relative + absolute backstop).
# ---------------------------------------------------------------------------

def bootstrap_ci_95(values: torch.Tensor, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    g = torch.Generator().manual_seed(seed)
    # LAUNCH FIX 2 companion: force CPU before indexing with the CPU-generator
    # draws below -- callers pass CUDA tensors on the box, and CUDA-tensor[
    # CPU-index-tensor] advanced indexing is version-fragile. Bootstrap is
    # tiny; CPU is free.
    values = values.detach().cpu().float().flatten()
    n = values.numel()
    assert n > 0
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    boot_means = values[idx].mean(dim=1)
    lower = torch.quantile(boot_means, 0.025).item()
    upper = torch.quantile(boot_means, 0.975).item()
    return lower, upper


def h1_sanity_floor(recovered_frac_h1_real: float, null_recovered_h1: torch.Tensor,
                     absolute_floor: float = H1_ABSOLUTE_FLOOR, boot_seed: int = 0) -> dict:
    """sec 8.4's registered pass condition: real h=1 recovery must exceed the
    null band's upper edge by a margin AT LEAST as large as the null band's
    own width, AND (MAJOR-2) exceed an absolute floor (0.10) independent of
    where the null band sits. BOTH required; either failing routes to
    probe-invalid."""
    null_lo, null_hi = bootstrap_ci_95(null_recovered_h1.float(), seed=boot_seed)
    null_width = null_hi - null_lo
    null_relative_pass = recovered_frac_h1_real > (null_hi + null_width)
    absolute_pass = recovered_frac_h1_real >= absolute_floor
    return {
        "null_ci_lower": null_lo, "null_ci_upper": null_hi, "null_width": null_width,
        "recovered_frac_h1_real": recovered_frac_h1_real,
        "null_relative_pass": null_relative_pass, "absolute_pass": absolute_pass,
        "probe_valid": null_relative_pass and absolute_pass,
    }


# ---------------------------------------------------------------------------
# sec 8.3 -- zero-cost covariates.
# ---------------------------------------------------------------------------

def state_condition_number(S_T: torch.Tensor) -> torch.Tensor:
    """S_T: (B,d,d) -> (B,) sigma_max/sigma_min (eigenvalue-spread covariate,
    sec 8.3 M1)."""
    sv = torch.linalg.svdvals(S_T.float())
    return sv[:, 0] / sv[:, -1].clamp(min=1e-12)


def cross_a_cosine_convergence(pred_all_a: torch.Tensor) -> torch.Tensor:
    """sec 8.3 M1's power-iteration-degeneracy signature: pred_all_a is
    (B,K,d) -- pred(a,h) for EVERY source slot a in a fixed episode at the
    SAME h. Returns (B,) mean pairwise cosine across the K predictions (high
    -> S_T has one dominant eigenvector, everything converges to the same
    answer regardless of which entity was actually queried)."""
    normed = F.normalize(pred_all_a, dim=-1)
    B, K, d = normed.shape
    gram = torch.bmm(normed, normed.transpose(1, 2))          # (B,K,K) pairwise cosines
    mask = ~torch.eye(K, dtype=torch.bool, device=gram.device).unsqueeze(0)
    off_diag = gram.masked_select(mask.expand(B, K, K)).view(B, K * (K - 1))
    return off_diag.mean(dim=-1)


def entity_gram_covariate(entity_embeddings: torch.Tensor) -> dict:
    """sec 7.7's standing covariate: raw pairwise-cosine Gram deviation over
    the entity-embedding rows actually used in an episode (identical
    instrument to key_anchoring.py::raw_table_conditioning's own Gram-
    deviation reading, realized here via the SAME gram_deviation function
    already imported). entity_embeddings: (B,K,d_model)."""
    gd = gram_deviation(F.normalize(entity_embeddings, dim=-1))
    return {"entity_gram_deviation_mean": gd.mean().item()}


# ---------------------------------------------------------------------------
# Option 2 -- fully natural next-token logit margin, computed ONLY at the
# query position, over ONLY the true-answer + K-1 distractor vocab rows
# (never the full (T,vocab) logits tensor -- sec 4.4's explicit registration,
# CLAUDE.md's own hard rule).
# ---------------------------------------------------------------------------

def option2_margin(hidden_at_query: torch.Tensor, embed_weight: torch.Tensor,
                    true_token_id: torch.Tensor, distractor_ids: torch.Tensor) -> torch.Tensor:
    """hidden_at_query: (B,d_model) post-norm_f hidden at the query's own
    final position. true_token_id: (B,) int64. distractor_ids: (B,K-1)
    int64. Returns margin (B,) = logit(true) - max_j logit(distractor_j),
    a continuous real-valued score (only the DISTRACTOR SET is enumerated,
    never the metric itself)."""
    true_vec = embed_weight[true_token_id]                       # (B,d_model)
    true_logit = (hidden_at_query * true_vec).sum(dim=-1)         # (B,)
    distractor_vecs = embed_weight[distractor_ids]                # (B,K-1,d_model)
    distractor_logits = torch.einsum('bd,bkd->bk', hidden_at_query, distractor_vecs)
    margin = true_logit - distractor_logits.max(dim=-1).values
    return margin


# ---------------------------------------------------------------------------
# CI utility (n=3 seeds, pinned t(2,.975)=4.303) + outcome-routing gates
# (sec 12 -- the mechanical gates, not the prose; these are pure functions
# a later harvest script calls once real multi-seed cell numbers exist).
# ---------------------------------------------------------------------------

def delta_ci_n3(values_a: list, values_b: list) -> dict:
    """Pinned CI over n=3 seeds' paired delta = a - b, this project's
    standing formula (t(2,.975)=4.303) wherever n=3 seeds."""
    assert len(values_a) == 3 and len(values_b) == 3, "delta_ci_n3 requires exactly 3 paired seeds"
    deltas = [a - b for a, b in zip(values_a, values_b)]
    mean = sum(deltas) / 3
    var = sum((d - mean) ** 2 for d in deltas) / 2  # n-1=2
    se = math.sqrt(var / 3)
    half_width = CI_T_975_DF2 * se
    return {"deltas": deltas, "mean": mean, "ci_low": mean - half_width, "ci_high": mean + half_width}


def option_agreement(option1_ci: dict, option2_ci_or_point) -> str:
    """sec 5.2/M1's agreement rule, restated mechanically: disagreement =
    Option 2's estimate/CI positively excludes zero on the NEGATIVE side
    while Option 1's CI excludes zero on the POSITIVE side. `option2_ci_or_point`
    is either a delta_ci_n3()-shaped dict (n=3) or a plain float point
    estimate (n=1, e.g. rung-3's descriptive-only reading, sec 6.1)."""
    opt1_excludes_positive = option1_ci["ci_low"] > 0
    if isinstance(option2_ci_or_point, dict):
        opt2_excludes_negative = option2_ci_or_point["ci_high"] < 0
    else:
        opt2_excludes_negative = option2_ci_or_point < 0
    if opt1_excludes_positive and opt2_excludes_negative:
        return "disagree"
    return "agree"


def killer_prediction_verdict(delta_k32: dict, delta_k20: dict, option1_option2_agreement: str) -> str:
    """sec 5.3's committed killer-prediction pass condition: |delta(K=32)| >
    |delta(K=20)|, K=32's CI excludes zero (positive) while K=20's does not,
    AND (sec 5.2/M1) Option 1/Option 2 agreement at K=32. Returns
    'CONFIRM' / 'REFUTE' / 'READOUT-DIVERGENCE'."""
    ci32_excludes_zero_positive = delta_k32["ci_low"] > 0
    ci20_excludes_zero = (delta_k20["ci_low"] > 0) or (delta_k20["ci_high"] < 0)
    k_dependence_pass = (abs(delta_k32["mean"]) > abs(delta_k20["mean"])
                          and ci32_excludes_zero_positive and not ci20_excludes_zero)
    if not k_dependence_pass:
        return "REFUTE"
    if option1_option2_agreement == "disagree":
        return "READOUT-DIVERGENCE"
    return "CONFIRM"


def leg_b_scale_gate(per_rung_agreement: list) -> str:
    """sec 6.2 Rev 4's mandatory Leg-B agreement gate: a monotone-trend
    CONFIRM may be claimed only over rungs where Option 1/Option 2 agree,
    requiring >=3 of the 4 ladder rungs. Returns 'ELIGIBLE_FOR_TREND_READ'
    (the caller then separately checks monotonicity/Spearman on the
    agreeing subset) or 'AMBIGUOUS'."""
    assert len(per_rung_agreement) == 4
    n_agree = sum(1 for a in per_rung_agreement if a == "agree")
    return "ELIGIBLE_FOR_TREND_READ" if n_agree >= 3 else "AMBIGUOUS"


def readout_form_invalid(h1_probe_valid: bool, h_ge2_uniform_floor_every_arm_rung: bool) -> bool:
    """sec 12's READOUT-FORM-INVALID trigger: h=1 clears its chance floor
    (probe itself valid) but h>=2 sits at floor UNIFORMLY across every
    arm/rung at every tested K."""
    return bool(h1_probe_valid and h_ge2_uniform_floor_every_arm_rung)


def outcome_precedence(is_readout_form_invalid: bool, is_ambiguous: bool) -> str | None:
    """sec 12 Rev 5's precedence rule: READOUT-FORM-INVALID wins when both
    triggers fire simultaneously (the more specific diagnosis)."""
    if is_readout_form_invalid:
        return "READOUT-FORM-INVALID"
    if is_ambiguous:
        return "AMBIGUOUS"
    return None


# ---------------------------------------------------------------------------
# Checkpoint path resolution.
# ---------------------------------------------------------------------------

def leg_a_cell_name(arm: str, lam: float, corpus: str, seed: int,
                     d_model: int = 256, d_state: int = 64, n_layers: int = 2) -> str:
    """Verbatim reproduction of frozen_bias_lm_sweep.py::cell_name -- verified
    directly against that file this build session (produces exactly
    'frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0' for the
    Stage-0 pinned control-arm checkpoint, sec 9)."""
    lam_tag = f"{lam:.2f}".replace(".", "p")
    return f"frozenbias_lm_{arm}_lam{lam_tag}_{corpus}_dm{d_model}_ds{d_state}_L{n_layers}_s{seed}"


def leg_a_ckpt_path(ckpt_base_dir: str, arm: str, lam: float, corpus: str, seed: int, step: int = 20000,
                     d_model: int = 256, d_state: int = 64, n_layers: int = 2) -> str:
    """<ckpt_base_dir>/<cell_name>/<run_name>_step<n>.pt -- verified directly
    against frozen_bias_lm_sweep.py::build_cmd_cell (ckpt_dir = ckpt_base_dir/
    spec['name']) and lm_pretrain_rd.py's own run_name auto-derivation
    (f'lmC_{corpus}_dm{d_model}_ds{d_state}_L{n_layers}_s{seed}', which does
    NOT include arm/lambda -- disambiguated by the cell-name subdirectory)."""
    cell = leg_a_cell_name(arm, lam, corpus, seed, d_model, d_state, n_layers)
    run_name = f"lmC_{corpus}_dm{d_model}_ds{d_state}_L{n_layers}_s{seed}"
    return os.path.join(ckpt_base_dir, cell, f"{run_name}_step{step}.pt")


def leg_b_ckpt_path_bestguess(ckpt_root: str, rung: int, corpus: str, seed: int, step: int) -> str:
    """BEST-EFFORT resolver, NOT authoritative. REASONING_LINK_DESIGN.md sec
    0 explicitly discloses Track C's checkpoint naming as "per-wave, not
    uniform" -- this is inferred from run_lm_rd_trackc_sweep.py's own
    cell_name()/_ckpt_dir()/gate_and_run_* functions (verified by direct
    code read this build session): <ckpt_root>/<wave_tag>/checkpoints/
    lmC_<corpus>_dm<d>_ds<ds>_L<L>_s<seed>_step<N>.pt. If this path does not
    exist on the real box, pass --ckpt directly instead of trusting this
    guess (the chain script does exactly that for its own pinned cells)."""
    cfg = LEG_B_RUNG_CFG[rung]
    run_name = f"lmC_{corpus}_dm{cfg['d_model']}_ds{cfg['d_state']}_L{cfg['n_layers']}_s{seed}"
    # LAUNCH FIX 3 (2026-07-07, verified against the REAL box layout, not
    # inferred from code): wave checkpoint files live FLAT in
    # <ckpt_root>/<wave_tag>/ -- there is NO "checkpoints/" subdirectory
    # (confirmed: /data/lm_rd_trackc_ckpts/wave3/lmC_..._step*.pt). Keep the
    # old guess as a fallback probe for robustness across wave layouts.
    flat = os.path.join(ckpt_root, cfg["tag"], f"{run_name}_step{step}.pt")
    nested = os.path.join(ckpt_root, cfg["tag"], "checkpoints", f"{run_name}_step{step}.pt")
    return flat if os.path.exists(flat) or not os.path.exists(nested) else nested


# ---------------------------------------------------------------------------
# REASONING-LINK's own result-schema wrap. Sec 12.3.1's mech_schema.
# wrap_exploratory() is a FROZEN_BIAS_LM_DESIGN.md-specific "mechanism-wave"
# tier convention (sec 12) -- REASONING_LINK_DESIGN.md never references
# mech_schema or wrap_exploratory anywhere (grepped, zero hits) and is its
# own program with its own pre-registered CONFIRM/REFUTE/READOUT-DIVERGENCE/
# READOUT-FORM-INVALID/AMBIGUOUS/probe-invalid outcome vocabulary (sec 12) --
# reusing FROZEN_BIAS's "exploratory-mechanism-wave -- NOT a confirmatory
# bar" tier label here would misrepresent a DIFFERENT program's own
# confirmatory-with-registered-thresholds design as something it explicitly
# is not. This wrap stamps a REASONING-LINK-specific, honest tier instead.
# ---------------------------------------------------------------------------

def wrap_reasoning_link(payload: dict, stage: str) -> dict:
    assert isinstance(payload, dict)
    assert stage in ("stage-minus1-selftest", "stage0-calibration", "stage0.5-cost-calibration", "stage1-cell")
    wrapped = {"design_ref": DESIGN_REF, "program": "REASONING-LINK Phase 1", "stage": stage,
               "fla_stub_installed": FLA_STUB_INSTALLED, "timestamp": time.time()}
    wrapped.update(payload)
    return wrapped


# ---------------------------------------------------------------------------
# Top-level cell orchestration -- runs ONE (checkpoint, K) cell, ALL tested
# h's, end to end.
# ---------------------------------------------------------------------------

def assert_forward_call_counts(forward_counts: dict, context: str = "") -> None:
    """FATAL-1 fix, this audit round -- a guard WITH TEETH against a
    regression back to per-h forward calls. sec 10 prices every (checkpoint,
    K) cell as exactly ONE forward-A + ONE forward-B ('anchor x 8' already
    folds in the two-forward protocol as a single x2 factor, sec 10's own
    multiplier derivation -- not x2 per h). The combined main-query +
    self-query batching below (measure_cell_all_h) achieves forward_b==1;
    if that batching is ever judged unsafe and a caller falls back to two
    separate forward-B calls (main query, self-query), forward_b==2 is the
    disclosed, still-registered fallback (sec 10's own pricing already
    treats forward-B as the query-extraction pass regardless of how many
    distinct query PURPOSES ride inside its batch dimension). Anything else
    (0, or >=3, or forward_a != 1) means a caller silently reintroduced a
    per-h forward loop -- exactly FATAL-1's own 6x-cost-inflation bug --
    and must fail LOUDLY, not slip through as a quiet cost regression."""
    assert forward_counts["forward_a"] == 1, (
        f"FATAL-1 guard{(' (' + context + ')') if context else ''}: expected exactly 1 forward-A "
        f"call per (checkpoint,K) cell (all tested h's reuse the SAME S_T/k_eff/v_eff), got "
        f"forward_a={forward_counts['forward_a']} -- a regression to per-h forward-A calls "
        f"silently re-inflates cost len(hops)x, exactly the bug this guard exists to catch.")
    assert forward_counts["forward_b"] in (1, 2), (
        f"FATAL-1 guard{(' (' + context + ')') if context else ''}: expected 1 forward-B call "
        f"(main-query + self-query batched together) or 2 (disclosed unbatched fallback) per "
        f"(checkpoint,K) cell, got forward_b={forward_counts['forward_b']}.")


def measure_cell_all_h(model: DeltaNetLM, episode_cfg: grammar_rd.DeltaNetRDTaskConfig,
                        pools: grammar_rd.EntityPools, readout_layer: int, K: int, hops: tuple,
                        batch_size: int, seed: int, surgery: str, device: str,
                        max_rows_per_forward: int = 4096, compute_option2: bool = True,
                        compute_premises: bool = True, null_seed: int | None = None
                        ) -> tuple[dict, dict]:
    """FATAL-1 fix (this audit round) -- the per-cell measurement body SHARED
    by `run_cell` (Stage 1) and `run_stage0` (calibration), restructured to
    draw ONE episode batch and run ONE forward-A + ONE (batched) forward-B
    for ALL of `hops`, instead of the pre-fix code's per-h loop (which
    redrew a batch and reran forward-A + 2xforward-B on EVERY h -- 4
    forward-A + 8 forward-B per 4-h cell, a 6x-8x cost inflation vs sec 10's
    own 'all 4 h scored from the SAME captured passes' pricing, sec 10/4.2).

    Why one draw suffices for every h (not an approximation): the hop count
    `h` NEVER enters the surface form (sec 4.2) -- `q_eff` (extracted via
    forward-B) and `S_T`/`k_eff`/`v_eff` (via forward-A) do not depend on
    `h` AT ALL. `h` only parameterizes two things, both cheap, no-new-
    forward-pass operations external to the model: (i) how many times
    `apply_state_power` matrix-powers `S_T` against `q_eff`, and (ii) which
    slot `compute_prev_slot_and_target` grades against. So the SAME
    forward-A/forward-B outputs are scored once per h in `hops`, exactly
    the design's own 'same shuffled-pairing episodes, scored at each h
    separately' language (sec 9's Stage-0 prose, now actually implemented
    that way rather than merely claimed).

    Numerical equivalence with the pre-fix per-h code path (verified this
    audit-fix session on a synthetic checkpoint, see the BUILD/FIX report):
    the pre-fix code called `sample_batch_rd(..., hop_set=(h,), ...)`
    separately per h, always with the SAME `seed` -- since `torch.randint`'s
    state advance depends only on output shape (not on the `high` value
    passed in), `entity_ids`/`succ`/`rel_id`/`a_slot` were already
    bit-identical across those 4 separate draws; only the (wasteful) repeat
    forward passes differed. Passing `hop_set=hops` here (rather than a
    single-h tuple) preserves that same state-advance property, so this
    restructure changes NOTHING about which numbers get computed -- only
    how many times the model is asked to compute them.

    Marker-batching (FATAL-1's forward-B fix): main-query rows (one query
    per (episode, Q-slot), Q=`cfg.queries`) and self-query rows (sec 4.4's
    premise-(iii)/(iv) self-query pass, one per (episode, K-slot)) are
    concatenated along the BATCH dimension into ONE forward-B call, main
    rows first ([0, B*Q)), self-query rows second ([B*Q, B*Q+B*K)) --
    documented layout, sliced back apart below. This is safe by
    construction, verified directly against `DeltaNetLMMixer.forward`
    (`lm_pretrain_rd.py`): nothing in that forward body mixes information
    ACROSS the batch dimension anywhere -- `RMSNorm`/`o_norm` normalize
    per-token (last-dim only), `ShortConvolution` is a per-channel,
    per-row-independent DEPTHWISE causal conv (`groups=hidden_size`), and
    `chunk_delta_rule`'s recurrent state is a per-(batch,head) matrix, never
    shared or attended across rows -- no BatchNorm, no cross-row attention,
    no batch-level reduction exists anywhere in the mixer. Batching main +
    self-query rows into one call is therefore exactly as safe as running
    two ordinary batches through the model back to back; it changes the
    wall-clock cost, not either half's own computed output.

    Returns (per_h: dict[h -> result], forward_counts: dict) --
    `forward_counts` is meant to be checked by the caller via
    `assert_forward_call_counts` (the FATAL-1 regression guard)."""
    forward_counts = {"forward_a": 0, "forward_b": 0}
    surgery_off = (surgery == "off")

    gen = torch.Generator(device=device).manual_seed(seed)
    batch = grammar_rd.sample_batch_rd(episode_cfg, batch_size, gen, hop_set=tuple(hops), pools=pools,
                                        device=device)

    # --- forward-A: ONCE, for every h (S_T/k_eff/v_eff never depend on h). ---
    final_states, k_raw, v_raw = run_forward_a(model, batch["token_ids"], surgery_off=surgery_off)
    forward_counts["forward_a"] += 1
    k_eff_items = gather_at_positions(k_raw[readout_layer], batch["item_pos"])
    v_eff_items = gather_at_positions(v_raw[readout_layer], batch["item_pos"])
    S_T = squeeze_state_head(final_states[readout_layer])

    B, Q = batch["hops"].shape
    T_bind = batch["token_ids"].shape[1]
    query_len = batch["query_tokens"].shape[-1]
    bind_expanded = batch["token_ids"].unsqueeze(1).expand(B, Q, T_bind).reshape(B * Q, T_bind)
    query_flat = batch["query_tokens"].reshape(B * Q, query_len)
    main_concat = torch.cat([bind_expanded, query_flat], dim=1)
    n_main_rows = main_concat.shape[0]

    self_concat = None
    Bk = Kk = None
    if compute_premises:
        self_q_windows = grammar_rd.self_query_tokens(episode_cfg, pools, batch["key_ids"], batch["rel_id"])
        Bk, Kk, qlen = self_q_windows.shape
        assert qlen == query_len, (
            "main-query and self-query windows must share one query_len to batch them together "
            f"(main={query_len}, self={qlen})")
        self_bind_expanded = batch["token_ids"].unsqueeze(1).expand(Bk, Kk, T_bind).reshape(Bk * Kk, T_bind)
        self_concat = torch.cat([self_bind_expanded, self_q_windows.reshape(Bk * Kk, qlen)], dim=1)
        combined = torch.cat([main_concat, self_concat], dim=0)   # main rows [0,B*Q), self rows after
    else:
        combined = main_concat

    # --- forward-B: ONE logical pass over the combined (main+self) batch, for every h. May be
    # chunked into several sub-calls purely for VRAM (max_rows_per_forward) -- that is still ONE
    # logical forward-B pass (sec 10 prices whole episode batches, not internal VRAM chunking),
    # so forward_counts["forward_b"] is incremented ONCE here, not once per chunk. ---
    q_eff_chunks, hidden_chunks = [], []
    for start in range(0, combined.shape[0], max_rows_per_forward):
        chunk = combined[start:start + max_rows_per_forward]
        hidden, _, q_raw = run_forward_b(model, chunk, need_option2_hidden=compute_option2,
                                          surgery_off=surgery_off)
        q_eff_chunks.append(q_raw[readout_layer][:, -1, :])
        if compute_option2:
            hidden_chunks.append(hidden[:, -1, :])
    forward_counts["forward_b"] += 1
    q_eff_all = torch.cat(q_eff_chunks, dim=0)
    q_eff = q_eff_all[:n_main_rows].view(B, Q, -1)
    q_eff_self = q_eff_all[n_main_rows:].view(Bk, Kk, -1) if compute_premises else None
    hidden_at_query = torch.cat(hidden_chunks, dim=0)[:n_main_rows].view(B, Q, -1) if compute_option2 else None

    # --- premises (iii)/(iv): cell-level facts, computed ONCE (they depend only on
    # k_eff_items/v_eff_items/q_eff_self, never on h -- sec 4.4's own premise definitions), copied
    # verbatim into every h's result dict below. The pre-fix per-h loop recomputed these len(hops)
    # times redundantly; since the underlying tensors never changed across h, the recomputed
    # numbers were always identical -- this is the SAME computation done once, not a behavior
    # change. ---
    premise_result = {}
    if compute_premises:
        prem = premise_iii_iv(k_eff_items, v_eff_items, q_eff_self, gen)
        action = premise_action_rule(prem["same_entity_cos_iii"], prem["null_cos_iii"],
                                      prem["same_entity_cos_iv"], prem["null_cos_iv"])
        premise_result.update(action)
        premise_result.update(premise_i_ii(k_eff_items, v_eff_items))

    succ_null = None
    if null_seed is not None:
        gen_null = torch.Generator(device=device).manual_seed(null_seed)
        succ_null = build_null_labeling(batch["succ"], gen_null)

    per_h = {}
    for h in hops:
        # Override the batch's own (randomly-drawn, hop_set-sampled) `hops` tensor with a literal
        # h -- scores THIS SAME batch's a_slot/succ (identical episodes for every h) at exactly h,
        # rather than relying on batch["hops"] to have happened to draw h uniformly (it does not,
        # with len(hops)>1 -- see module docstring's own equivalence note above).
        hops_h = torch.full_like(batch["a_slot"], h)
        prev_slot, v_eff_target = compute_prev_slot_and_target(batch["succ"], batch["a_slot"], hops_h,
                                                                 v_eff_items)
        pred = apply_state_power(S_T, q_eff, h)
        cos, recovered = cosine_and_recovered(pred, v_eff_target)
        recovered_frac = recovered.float().mean().item()

        result = {
            "h": h, "K": K, "surgery": surgery, "batch_size": batch_size,
            "recovered_frac": recovered_frac, "n_scored": int(recovered.numel()),
            "cos_mean": cos.mean().item(), "cos_std": cos.std(unbiased=False).item(),
            "state_condition_number_mean": state_condition_number(S_T).mean().item(),
        }

        if compute_option2:
            tgt_slot = grammar_rd._iterate_permutation(batch["succ"], batch["a_slot"], hops_h)
            true_token_id = torch.gather(batch["entity_ids"], 1, tgt_slot)
            all_entities = batch["entity_ids"].unsqueeze(1).expand(B, Q, K)
            true_expand = true_token_id.unsqueeze(-1).expand(B, Q, K)
            is_true = all_entities == true_expand
            distractor_ids = torch.stack([
                all_entities[b, q][~is_true[b, q]][:K - 1] for b in range(B) for q in range(Q)
            ], dim=0).view(B, Q, K - 1)
            margins = option2_margin(hidden_at_query.reshape(B * Q, -1), model.embed.weight,
                                      true_token_id.reshape(B * Q), distractor_ids.reshape(B * Q, K - 1))
            result["option2_margin_mean"] = margins.mean().item()
            result["option2_margin_std"] = margins.std(unbiased=False).item()

        if compute_premises:
            result.update(premise_result)

        if null_seed is not None:
            _, v_eff_target_null = compute_prev_slot_and_target(succ_null, batch["a_slot"], hops_h, v_eff_items)
            _, recovered_null = cosine_and_recovered(pred, v_eff_target_null)
            result["null_recovered_frac_mean"] = recovered_null.float().mean().item()
            if h == 1:
                result.update(h1_sanity_floor(recovered_frac, recovered_null))
                result["recovered_frac_h1_real"] = recovered_frac

        per_h[h] = result

    return per_h, forward_counts


def run_cell(ckpt_path: str, K: int, hops: tuple, surgery: str = "native", batch_size: int = BATCH_SIZE_DEFAULT,
             device: str = "cpu", max_rows_per_forward: int = 4096, marker_word: str | None = None,
             compute_option2: bool = True, leg: str | None = None, condition_idx: int = 0,
             corpus_idx: int = 0, ckpt_seed_idx: int = 0) -> dict:
    """Runs the full two-forward Option-1 (+ Option-2) readout for ONE cell:
    one checkpoint, one K, one or more h's, one surgery mode ('native' = as
    loaded; 'off' = sec 5.2a's forced-off surgery). Returns a dict with
    per-h recovered_frac, premises, covariates, and Option-2 margins -- the
    unit a later harvest script pools across seeds/corpora/arms/rungs.

    `leg`/`condition_idx`/`corpus_idx`/`ckpt_seed_idx` feed sec 4.6's
    episode-seed formula directly (the same non-collision guarantee Stage
    -1 item 10 verifies) -- `leg=None` (the CLI's ad-hoc default) falls
    back to a fixed literal seed for quick manual/exploratory runs
    OUTSIDE the registered grid, never silently reusing seed 0 for a REAL
    committed-grid cell without the caller noticing (the CLI always passes
    a real leg when --family is used, see main()).
    All 4 tested h's SHARE one episode_seed (the sec 4.6 formula has no
    h-axis at all) -- "same shuffled-pairing episodes, scored at each h
    separately" (sec 9's own Stage-0 language). FATAL-1 fix (this audit
    round): this is now ALSO true of the forward-pass COST, not merely the
    seed -- `measure_cell_all_h` below draws ONE episode batch and runs
    ONE forward-A + ONE forward-B for every h in `hops`, asserted by
    `assert_forward_call_counts` immediately after (the regression guard)."""
    assert surgery in ("native", "off"), f"surgery={surgery!r} not in ('native','off')"
    model, ckpt = load_checkpoint(ckpt_path, device)
    conv_size = ckpt["config"]["conv_size"]
    episode_cfg = episode_config_for_checkpoint(conv_size, K)
    assert ckpt["config"]["conv_size"] == episode_cfg.conv_size, (
        "Stage -1 item 7's own gate, re-checked live: checkpoint conv_size mismatch")

    if leg is not None:
        seed = episode_seed("eval", leg, condition_idx, corpus_idx, ckpt_seed_idx, K_INDEX[K])
    else:
        seed = 0  # ad-hoc/exploratory usage outside the registered grid (no --family given)

    pools, pool_report = build_reasoning_link_pools(seed=0, marker_word=marker_word)
    readout_layer = readout_layer_index(model)

    # sec 4.4/Rev 6: premises (iii)/(iv) are re-derived "AT EVERY RUNG'S OWN PRIMARY CELL (first
    # pass per rung)", not borrowed wholesale from Stage 0's single 14M-scale calibration point --
    # computed once here (compute_premises=True), inside the SAME forward-A/forward-B pair every
    # h in `hops` reuses (FATAL-1 fix), so a harvest script can read whichever cell it wants as
    # that rung's "first pass" reading.
    per_h_results, forward_counts = measure_cell_all_h(
        model, episode_cfg, pools, readout_layer, K, hops, batch_size, seed, surgery, device,
        max_rows_per_forward=max_rows_per_forward, compute_option2=compute_option2, compute_premises=True)
    assert_forward_call_counts(forward_counts, context="run_cell")

    return wrap_reasoning_link({
        "ckpt_path": ckpt_path, "ckpt_config": ckpt["config"], "K": K, "hops": list(hops),
        "surgery": surgery, "readout_layer": readout_layer, "n_layers": model.n_layers,
        "pool_report_marker": pool_report["marker_word"], "per_h": per_h_results,
        "forward_counts": forward_counts,
    }, stage="stage1-cell")


# sec 8.5 item 4: no exact numeric tolerance is pinned anywhere in the design
# for "the two-query-marker robustness replicate disagrees beyond a
# registered tolerance" at BUILD time -- this file's own module docstring
# point 4 (era: pre-audit) flagged 0.15 as build-time-invented, pending
# audit/reviewer confirmation. MINOR-4 fix (this audit round): the design
# has SINCE been updated to close that gap -- REASONING_LINK_DESIGN.md sec
# 8.5 item 4, "Registered tolerance (Rev 6.1) ... 0.15 in recovered_frac
# units" -- so this constant now REPRODUCES a design-registered number, not
# an invented one; the value itself is unchanged (0.15), only its
# provenance. Every constant in this file quotes or directly re-derives a
# design-stated number.
MARKER_DISAGREEMENT_TOLERANCE = 0.15


def blinded_console_summary(result: dict) -> dict:
    """sec 9 Stage 0/0.5's "blinded from any headline decision" requirement,
    realized mechanically: strip every recovery-bearing number
    (recovered_frac, cosine stats, option2 margins, premise medians) from
    what gets PRINTED to the chain's main log, keeping only structural
    pass/fail gate outcomes, timing, and metadata. The full numbers always
    still go to the JSON `--out` file -- this function only governs what
    `main()` prints to stdout/the tee'd chain log."""
    BLIND_KEYS = {"recovered_frac", "cos_mean", "cos_std", "option2_margin_mean", "option2_margin_std",
                  "premise_iii_median", "premise_iv_median", "null_ci_lower", "null_ci_upper",
                  "recovered_frac_h1_real", "premise_i_key_gram_deviation_mean",
                  "premise_ii_value_gram_deviation_mean", "null_recovered_frac_mean", "null_width"}

    def _strip(obj):
        if isinstance(obj, dict):
            return {k: ("<BLINDED -- see --out JSON>" if k in BLIND_KEYS else _strip(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    return _strip(result)


def run_stage0(ckpt_path: str, batch_size: int = BATCH_SIZE_DEFAULT, device: str = "cpu",
               seed: int = 0, K: int = 32) -> dict:
    """sec 9 Stage 0 -- the Leg-A control-arm calibration cell. Purposes
    (a)-(e), all computed here: (a) causality assertion on a REAL checkpoint;
    (b) per-cell wall-clock; (c) per-h label-shuffle null bands + the h1
    sanity floor; (d) premises (i)-(iv); (e) the action-rule gates applied
    BEFORE any Stage 1 grid launches. Runs sec 7.6's two-marker robustness
    replicate and flags disagreement (sec 8.5 item 4) beyond
    MARKER_DISAGREEMENT_TOLERANCE. Recovery numbers are written to the
    returned dict (and thus the JSON this caller writes to disk) but the
    CLI's console print is blinded via `blinded_console_summary` -- see
    that function's own docstring."""
    model, ckpt = load_checkpoint(ckpt_path, device)
    conv_size = ckpt["config"]["conv_size"]
    episode_cfg = episode_config_for_checkpoint(conv_size, K)
    readout_layer = readout_layer_index(model)
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    base_pools_probe, _ = build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
    two_markers = choose_two_markers(tokenizer, base_pools_probe)[:2]

    t0 = time.time()
    gen_c = torch.Generator(device=device).manual_seed(seed)
    causality_batch = grammar_rd.sample_batch_rd(episode_cfg, batch_size, gen_c, hop_set=(1,),
                                                  pools=base_pools_probe, device=device)
    causality_result = causality_check(model, causality_batch["token_ids"],
                                        causality_batch["query_tokens"][:, 0, :])

    calibration_seed = episode_seed("calibration", "leg_a", 0,
                                     CORPUS_INDEX.get(ckpt.get("corpus", "openr1-mix-ext"), 0), 0, K_INDEX[K])
    null_shuffle_seed = episode_seed("null_shuffle", "leg_a", 0, 0, 0, K_INDEX[32])
    per_marker = {}
    forward_counts_per_marker = {}
    for marker_entry in two_markers:
        marker_word = marker_entry["candidate"]
        pools, _ = build_reasoning_link_pools(tokenizer=tokenizer, seed=0, marker_word=marker_word)
        # sec 4.6's seed formula has NO h-axis: every h SHARES `calibration_seed`. FATAL-1 fix
        # (this audit round): `measure_cell_all_h` draws ONE episode batch and runs ONE forward-A
        # + ONE forward-B for ALL of H_TEST here, per MARKER -- "1 cell x 2 markers, all 4 h scored
        # from the SAME captured passes" (sec 10's own Stage-0 pricing), not the pre-fix code's 2
        # markers x 4 h = 8 separate (redraw + reforward) calls. Both markers also share this same
        # seed, so the marker-disagreement check (below) compares the IDENTICAL underlying
        # episodes, differing only in the query-marker token. Option 2 is not part of Stage 0's own
        # registered purposes (a)-(e) -- skipped here (compute_option2=False), unlike run_cell's
        # real Stage-1 cells.
        per_h, fcounts = measure_cell_all_h(model, episode_cfg, pools, readout_layer, K, H_TEST, batch_size,
                                             calibration_seed, "native", device, compute_option2=False,
                                             compute_premises=True, null_seed=null_shuffle_seed)
        assert_forward_call_counts(fcounts, context=f"run_stage0 marker={marker_word!r}")
        per_marker[marker_word] = per_h
        forward_counts_per_marker[marker_word] = fcounts

    marker_words = list(per_marker.keys())
    disagreement = {}
    if len(marker_words) == 2:
        for h in H_TEST:
            diff = abs(per_marker[marker_words[0]][h]["recovered_frac"] - per_marker[marker_words[1]][h]["recovered_frac"])
            disagreement[h] = {"abs_diff": diff, "exceeds_tolerance": diff > MARKER_DISAGREEMENT_TOLERANCE}
    marker_disagreement_flag = any(v["exceeds_tolerance"] for v in disagreement.values())

    wall_s = time.time() - t0
    return wrap_reasoning_link({
        "ckpt_path": ckpt_path, "K": K, "batch_size": batch_size, "wall_s": wall_s,
        "causality_check": causality_result,
        "per_marker": per_marker, "marker_disagreement_tolerance": MARKER_DISAGREEMENT_TOLERANCE,
        "marker_disagreement": disagreement, "marker_disagreement_flag": marker_disagreement_flag,
        "gate_result_h1_probe_valid": per_marker[marker_words[0]][1]["probe_valid"],
        "forward_counts_per_marker": forward_counts_per_marker,
    }, stage="stage0-calibration")


def run_stage05(ckpt_path: str, K: int = 64, batch_size: int = BATCH_SIZE_DEFAULT, device: str = "cpu",
                baseline_gpu_h_per_pass: float = 0.09) -> dict:
    """sec 9 Stage 0.5 -- rung-3 pass-cost calibration, BLINDED to any h4-
    style recovery readout (the recovery numbers `measure_cell_all_h`
    computes are discarded below, never surfaced in this function's own
    returned dict or printed anywhere -- only wall-clock for the real
    per-cell cost). Two-tier abort: >2x baseline revokes the K-extension
    eligibility (caller's job to act on); >4x baseline means halt Leg B's
    rung-3 rows entirely (caller's job). OOM fallback: retry ONCE at
    batch_size=8, then halt-with-disclosure (never an open-ended retry
    loop, sec 9's own MINOR fix).

    FATAL-1 fix (this audit round) -- times the REAL per-cell measurement
    path, not a hand-duplicated bare-forward proxy. The pre-fix
    `_time_one_pass` reimplemented a simplified forward-A + forward-B
    sequence inline, which OMITTED the premise self-query forward-B pass
    and the Option-2 hidden computation every REAL Stage-1 cell actually
    pays for (`run_cell`'s own `compute_premises=True`/`compute_option2=True`
    defaults) -- silently UNDER-pricing what the committed grid costs and
    defeating the whole point of a calibration cell. Calling
    `measure_cell_all_h` directly (the SAME shared function `run_cell`
    calls) instead of a parallel reimplementation means this calibration
    can never again silently drift out of sync with what a real cell
    executes. `H_TEST` (all 4 h) is passed to mirror the real grid's own
    `--hops 1,2,3,4` invocation exactly -- though per that function's own
    docstring, forward-pass cost does not depend on `len(hops)` at all, so
    this is a faithfulness choice, not a cost difference from `(1,)`."""
    def _time_one_pass(bs: int) -> float:
        model, ckpt = load_checkpoint(ckpt_path, device)
        conv_size = ckpt["config"]["conv_size"]
        episode_cfg = episode_config_for_checkpoint(conv_size, K)
        pools, _ = build_reasoning_link_pools(seed=0)
        readout_layer = readout_layer_index(model)
        seed = episode_seed("calibration", "leg_b", 3, 0, 0, K_INDEX[K])
        t0 = time.time()
        _per_h, fcounts = measure_cell_all_h(model, episode_cfg, pools, readout_layer, K, H_TEST, bs, seed,
                                              "native", device, compute_option2=True, compute_premises=True)
        wall_s = time.time() - t0
        assert_forward_call_counts(fcounts, context="run_stage05")
        del _per_h  # BLINDED: recovery numbers are computed (to price the real cost honestly) but
                    # never retained, returned, or printed by this function.
        return wall_s / bs   # per-episode wall time; caller converts to GPU-h/pass at the box's own rate

    oom_fallback_used = False
    try:
        per_episode_s = _time_one_pass(batch_size)
        used_batch_size = batch_size
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        oom_fallback_used = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            per_episode_s = _time_one_pass(8)
            used_batch_size = 8
        except RuntimeError as e2:
            return wrap_reasoning_link({
                "ckpt_path": ckpt_path, "K": K, "status": "HALT_OOM_AT_BATCH_8",
                "error": str(e2), "action": "halt Leg-B rung-3 rows entirely, disclose (sec 9 MINOR fix)",
            }, stage="stage0.5-cost-calibration")

    ratio_to_baseline = per_episode_s / (baseline_gpu_h_per_pass * 3600.0 / batch_size) if per_episode_s else 0.0
    if ratio_to_baseline > 4.0:
        action = "HALT: Leg-B rung-3 rows entirely, disclose unevaluated within Phase-1 budget"
    elif ratio_to_baseline > 2.0:
        action = "REVOKE: rung-3's d=128 K-extension eligibility this phase; keep committed K=64 cell"
    else:
        action = "OK: within budget, proceed"
    return wrap_reasoning_link({
        "ckpt_path": ckpt_path, "K": K, "used_batch_size": used_batch_size,
        "oom_fallback_used": oom_fallback_used, "per_episode_wall_s": per_episode_s,
        "baseline_gpu_h_per_pass": baseline_gpu_h_per_pass, "ratio_to_baseline": ratio_to_baseline,
        "action": action,
    }, stage="stage0.5-cost-calibration")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["selftest", "stage0", "stage05", "cell"], required=True)
    ap.add_argument("--ckpt", type=str, default=None, help="direct checkpoint .pt path")
    ap.add_argument("--family", choices=["leg_a", "leg_b"], default=None)
    ap.add_argument("--arm", choices=list(FROZEN_BIAS_ARM_MODES), default="off")
    ap.add_argument("--lam", type=float, default=0.58)
    ap.add_argument("--rung", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--corpus", type=str, default="openr1-mix-ext")
    ap.add_argument("--ckpt-seed", type=int, default=0)
    ap.add_argument("--ckpt-base-dir", type=str, default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--hops", type=str, default="1,2,3,4")
    ap.add_argument("--arms", type=str, default=None, help="comma-separated, Leg-A multi-arm convenience")
    ap.add_argument("--surgery", choices=["native", "off"], default="native")
    ap.add_argument("--seed-purpose", type=str, default="eval", choices=list(PURPOSE_BASE))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.mode == "selftest":
        import reasoning_link_stage_minus1
        ok = reasoning_link_stage_minus1.run_all_selftests()
        sys.exit(0 if ok else 1)

    ckpt_path = args.ckpt
    if ckpt_path is None and args.family == "leg_a":
        base = args.ckpt_base_dir or FROZEN_BIAS_CKPT_ROOT_DEFAULT
        lam = 0.0 if args.arm == "off" else args.lam
        ckpt_path = leg_a_ckpt_path(base, args.arm, lam, args.corpus, args.ckpt_seed)
    elif ckpt_path is None and args.family == "leg_b":
        base = args.ckpt_base_dir or TRACKC_CKPT_ROOT_DEFAULT
        ckpt_path = leg_b_ckpt_path_bestguess(base, args.rung, args.corpus, args.ckpt_seed, step=20000)

    if args.mode == "cell":
        assert ckpt_path is not None, "--mode cell requires --ckpt or --family(+args)"
        hops = tuple(int(h) for h in args.hops.split(","))
        # sec 4.6's seed-allocation formula, wired from the family args: condition_idx = arm index
        # (Leg A) or rung index (Leg B); corpus_idx from the corpus string; ckpt_seed_idx = the
        # checkpoint's own training seed. `leg=None` (no --family) falls back to run_cell's own
        # ad-hoc literal-seed path for quick manual/exploratory use outside the registered grid.
        leg = args.family
        if leg == "leg_a":
            condition_idx = LEG_A_ARM_INDEX[args.arm]
        elif leg == "leg_b":
            condition_idx = args.rung
        else:
            condition_idx = 0
        corpus_idx = CORPUS_INDEX.get(args.corpus, 0)
        result = run_cell(ckpt_path, args.k, hops, surgery=args.surgery, batch_size=args.batch_size,
                           device=args.device, leg=leg, condition_idx=condition_idx, corpus_idx=corpus_idx,
                           ckpt_seed_idx=args.ckpt_seed)
        print(json.dumps({k: v for k, v in result.items() if k != "per_h"}, indent=2, default=str))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"wrote {args.out}")
        return

    if args.mode == "stage0":
        assert ckpt_path is not None, "--mode stage0 requires --ckpt or --family leg_a(+args)"
        result = run_stage0(ckpt_path, batch_size=args.batch_size, device=args.device, K=args.k)
        # sec 9 Stage 0's "blinded from any headline decision" requirement: the console/chain-log
        # print never shows recovery numbers, only structural gate outcomes + timing/metadata.
        print(json.dumps(blinded_console_summary(result), indent=2, default=str))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"wrote {args.out} (full, UNBLINDED numbers -- read only at a deliberate unblinding step)")
        return

    if args.mode == "stage05":
        assert ckpt_path is not None, "--mode stage05 requires --ckpt or --family leg_b --rung 3"
        result = run_stage05(ckpt_path, K=args.k, batch_size=args.batch_size, device=args.device)
        print(json.dumps(result, indent=2, default=str))   # timing-only, nothing to blind (sec 9's own text)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"wrote {args.out}")
        return


if __name__ == "__main__":
    main()
