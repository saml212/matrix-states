"""phase2_trajectory_analysis.py -- REASONING_LINK_DESIGN.md sec 16.16
(Rev 2.2, DESIGN-CLEARED-FOR-BUILD) Phase-2b vocab-space behavioral-contrast
analysis: for a given corpus, re-scores EVERY trajectory checkpoint of EVERY
familiarized cell (3 arms x 3 ckpt_seeds x 2 K's per checkpoint) via the
NEW frozen-checkpoint eval-L_query readout (sec 16.16.3's readout (B):
`eval_query_loss_heldout`, held-out entities, Q=K), pools the 3 ckpt_seeds
into a `delta_ci_n3` 3-seed CI per (arm, checkpoint), computes
`det`/`holds`/`agree` via `phase2_hexachotomy`'s own primitives, and
classifies the resulting trajectory via `phase2_hexachotomy.classify_
trajectory`.

**MAJOR-1 rewrite (sec 16.16.3, attack-round-1 on sec 16.16, the highest-
value finding that round found).** The Phase-2 (sec 16.2) version of this
module sourced `off_vals`/`arm_vals` from the DEAD `d_state`-space
`per_h[h]['recovered_frac']` quantity (0.0 in 30/30 sec 16.15.1 readings)
and `stage05_pass_by_c` from the permanently-FAILED Stage-0.5 gate JSONs --
left as-is, a clean run would silently force any real monotone-holds-true
trajectory into UNRESOLVED-GATE before a single (B)-readout number was ever
computed. This rewrite: (1) `killer_prediction_readout` now returns a plain
`L_query` FLOAT, sourced from `eval_query_loss_heldout` -- the OFF half
reads a precomputed cache (`off_lquery_cache-Phase2b.json`, sec 16.16.8's
chain-step-3 cache builder; MAJOR-R2-3's own duplication fix), the non-off
half loads the frozen checkpoint live via the single-seam
`phase2b_load_eval_model` helper (Rev 2.1 MAJOR-R3-1). (2)
`stage05_pass_by_c` is now UNCONDITIONALLY `{c: True for c in CHECKPOINTS}`
(mirrors `phase2_hexachotomy.totality_check`'s own `always_pass_gate`
verbatim) -- the per-checkpoint Stage-0.5 gate is RETIRED for Phase-2b,
replaced by the single upfront OFF-floor gate (sec 16.16.6,
`phase2b_floor_gate_enforce.py`). The dead `gate_json_path_for` helper and
its `phase2_gate_enforce.gate_verdict` read are DELETED, not bypassed.

**Per-arm scoping fix (sec 16.18.3/16.18.9's own registered follow-up,
closed here -- NOT a design change, the classification RULES are sec
16.16.5's/`phase2_hexachotomy`'s, unchanged; this completes the built
pipeline to the scoping the design always specced).** `analyze_corpus`
classifies BOTH non-off arms INDEPENDENTLY per corpus (4 verdicts total
across the 2 corpora: each arm's own `holds_by_c`/`stage05_pass_by_c`
drives its OWN `classify_trajectory` call, via the new `classify_arms`
helper). Pre-fix, the corpus-level `classification` field was computed
from ONLY the `global` arm's own `holds_by_c` pattern (`per_token` folded
in solely via its terminal-checkpoint `det_arm` value, feeding outcomes
#4/#5) -- a silent proxy that masked a real, independently-re-derived
`wikitext-mix-ext x per_token` TRANSIENT signal (sec 16.18.3's own
hand-derivation table). The single top-level `classification` field is
KEPT, byte-identical to its pre-fix value (`classification_by_arm
["global"]`), for backward compatibility with `phase2b_chain.sh`'s (and
the historical `phase2_chain.sh`'s) own summary step, which reads
`d["classification"]` verbatim -- `classification_by_arm` is the
registered, complete 2-arm answer new callers should prefer.

**Secondary readout (sec 16.16.7):** `secondary_ood_readout` reuses the
SAME `eval_query_loss_heldout` function at `hop_set=(3,4)` (the held-out
hop depths never trained on) and reports a SIMPLER standalone `det(K,c)`
table per arm -- NOT folded into the primary hexachotomy classification.

Run (per corpus, after both OFF (reused) and the two intervention arms' 6
cells each have completed, AND `off_lquery_cache-Phase2b.json` has been
built by phase2b_off_cache.py, sec 16.16.8's chain-step-3):
    python phase2_trajectory_analysis.py --corpus openr1-mix-ext \\
        --ckpt-dir results/phase2/ckpts \\
        --off-cache results/phase2/off_lquery_cache-Phase2b.json \\
        --out results/phase2/traj_openr1_phase2b.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reasoning_link_probe as rlp  # noqa: E402  (installs the CPU fla stub as an import side effect)
import grammar_rd  # noqa: E402
import lm_pretrain_rd as lpr  # noqa: E402  (same direct-import convention phase2_stage_minus1.py already uses)
import phase2_familiarization_train as pft  # noqa: E402
import phase2_hexachotomy as phx  # noqa: E402

CKPT_SEEDS = (0, 1, 2)
ARMS_NON_OFF = ("global", "per_token")


def ckpt_path_for(ckpt_dir: str, arm: str, corpus: str, ckpt_seed: int, checkpoint_step: int) -> str:
    """NO K in this path (build-time correction, sec 16.2.3's own cost arithmetic: 18 training
    cells, not 36 -- see phase2_familiarization_train.K_TRAIN_DEFAULT's own comment). The SAME
    checkpoint file is re-scored at BOTH K=20 and K=32 by `killer_prediction_readout` below --
    the eval-time `K` controls readout episode construction only, independent of what K the
    checkpoint trained under."""
    return os.path.join(ckpt_dir, f"phase2fam_{arm}_{corpus}_s{ckpt_seed}_step{checkpoint_step}.pt")


# ---------------------------------------------------------------------------
# Loader -- the single seam (sec 16.16.3's "Loader" bullet, Rev 2.1 fix, round-3 verify MAJOR-R3-1).
# TWO PRODUCTION callers (Rev 2.2, round-4 MINOR-R4-2 harmonization): the rewritten
# `killer_prediction_readout` below (non-off arms only) and phase2b_off_cache.py's own OFF-eval
# cache builder (chain step 3, `build_off_lquery_cache`). Build-time disclosure (this Rev's own
# resolution, not a silent extension of the round-4-pinned "exactly two" invariant): sec 16.16.11
# item 1's own mandatory pre-launch timing pilot (a SEPARATE, later-registered obligation) is a
# THIRD, narrow caller (`phase2b_off_cache.time_one_eval_pass`) -- it needs a real loaded checkpoint
# to time one real scoring pass, and reusing this SAME seam (rather than a duplicate loader) is the
# DRY choice; disclosed explicitly here rather than silently widening the "exactly two" wording.
# Reproduces the SAME phase2_familiarization_train.py L408->L421
# double-defense sequence (torch.load(...)['config'] -> DeltaNetLM(**config) ->
# load_init_checkpoint_strict, config-equality assert + strict=True load_state_dict) -- production
# keeps the STRICT double-defense path unconditionally, never the laxer
# reasoning_link_probe.load_checkpoint (a single-layer load_state_dict with no explicit
# config-equality assert), which is explicitly REJECTED here.
# ---------------------------------------------------------------------------

def phase2b_load_eval_model(ckpt_path: str, device: str) -> lpr.DeltaNetLM:
    config = torch.load(ckpt_path, map_location=device)["config"]  # mirrors
                                     # phase2_familiarization_train.py L408
    model = lpr.DeltaNetLM(**config).to(device)
    lpr.load_init_checkpoint_strict(model, ckpt_path, device)      # mirrors
                                     # L421; lm_pretrain_rd.py L1803's own
                                     # config-equality assert + strict=True
                                     # load_state_dict double-defense
    model.eval()
    return model


# ---------------------------------------------------------------------------
# eval_query_loss_heldout -- readout (B), sec 16.16.3's "Build delta". Wraps query_loss_forward
# with use_heldout_entities=True and a CALLER-supplied hop_set (parameterized at the source,
# phase2_familiarization_train.query_loss_forward, sec 16.16.3's own disclosed parameterization
# task). NO surgery override (sec 16.16.3's "Surgery-mode scope, pinned" paragraph): eval-(B) runs
# the frozen-bias blend NATIVELY, required by sec 16.16.1's whole-causal-package framing --
# query_loss_forward's own forward pass has no frozen_bias_surgery wrapper on this path, and this
# function adds none of its own.
# ---------------------------------------------------------------------------

def _phase2b_seed_kind(hop_set: tuple) -> str:
    """Maps a hop_set to its own registered phase2_seed kind (sec 16.16.3's two new kinds)."""
    if tuple(hop_set) == tuple(pft.H_TRAIN):
        return "eval_lquery_heldout"
    if tuple(hop_set) == tuple(pft.H_TEST_HELD_OUT):
        return "eval_lquery_ood"
    raise ValueError(f"hop_set={hop_set!r} is neither the registered primary {pft.H_TRAIN} (sec "
                      f"16.16.3) nor secondary {pft.H_TEST_HELD_OUT} (sec 16.16.7) hop set")


def eval_query_loss_heldout(model: lpr.DeltaNetLM, K: int, hop_set: tuple, corpus: str,
                             ckpt_seed: int, checkpoint_step: int, batch_size: int = 16,
                             device: str = "cpu") -> float:
    """ONE frozen-checkpoint eval-L_query reading, held-out entities, Q=K (sec 16.16.3's readout
    (B)). `batch_size` pinned to a default of `16` (Rev 2, attack-round-2 MINOR-R2-1) -- matching
    `killer_prediction_readout`'s/`build_holds_and_gate_by_checkpoint`'s own existing `=16` default.

    **Pairing device, structural (sec 16.16.2 item 3, sec 16.16.3's own "Pairing device"
    paragraph): this function has NO `arm` parameter at all** -- the literal string `"off"` is
    baked into the seed formula's `arm` digit HERE, unconditionally, rather than left as a
    caller-supplied convention a future call site could violate. Every (arm-or-off) checkpoint
    scored at the SAME (corpus, ckpt_seed, K, checkpoint_step, hop_set) therefore draws the
    IDENTICAL held-out episode (same seed -> same torch.Generator stream -> same
    grammar_rd.sample_batch_rd draw) by construction -- a real paired comparison on shared
    episodes, never three independent draws. This is a build-time strengthening of sec 16.16.3's
    own illustrative signature (which named the caller-convention version) -- disclosed here per
    this project's "resolved during BUILD" convention, flagged for the independent audit.

    Pools/tokenizer are rebuilt per call (`rlp.build_reasoning_link_pools(seed=0)`, deterministic,
    marker template -- sec 16.8.3's own fallback pin), mirroring `reasoning_link_probe.run_cell`'s
    own existing per-call pool-rebuild convention exactly -- not a new inefficiency this build
    introduces."""
    kind = _phase2b_seed_kind(hop_set)
    seed = pft.phase2_seed(kind, "off", corpus, ckpt_seed, K, checkpoint_step)
    gen = torch.Generator(device=device).manual_seed(seed)
    episode_cfg = pft.familiarization_gate_episode_config(model.conv_size, K)  # Q=K, sec 16.16.3
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    pools, _ = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
    L_query, _, _ = pft.query_loss_forward(model, episode_cfg, pools, batch_size, gen, device,
                                            use_heldout_entities=True, step=checkpoint_step,
                                            hop_set=hop_set)
    return L_query.item()


# ---------------------------------------------------------------------------
# OFF-eval cache key -- shared format between this module (reader, killer_prediction_readout) and
# phase2b_off_cache.py (writer, chain step 3) -- defined ONCE here so the two can never drift apart.
# ---------------------------------------------------------------------------

def off_cache_key(corpus: str, ckpt_seed: int, K: int, checkpoint_step: int, hop_set: tuple) -> str:
    return f"{corpus}|{ckpt_seed}|{K}|{checkpoint_step}|{hop_set[0]}-{hop_set[1]}"


def killer_prediction_readout(ckpt_dir: str, arm: str, corpus: str, ckpt_seed: int, K: int,
                               checkpoint_step: int, off_cache: dict, hop_set: tuple = pft.H_TRAIN,
                               batch_size: int = 16, device: str = "cpu") -> float:
    """ONE (arm-or-off, corpus, ckpt_seed, K, checkpoint_step, hop_set) L_query reading (sec
    16.16.3's MAJOR-1 rewrite -- REPLACES the Rev-0/Phase-2 version, which returned a dict sourced
    from the dead `recovered_frac` quantity; this returns a plain float, `off_vals`/`arm_vals` are
    now populated directly from these).

    `arm == "off"`: reads from the precomputed `off_cache` (sec 16.16.3 item 1's own Rev-2 OFF-eval
    cache fix, MAJOR-R2-3) -- NEVER loads a model or calls `eval_query_loss_heldout` for this
    branch. This is one of `phase2b_load_eval_model`'s two PRODUCTION callers (Rev 2.2, round-4
    MINOR-R4-2 -- see that function's own docstring for the disclosed third, timing-pilot-only
    caller): this function's own non-off branch below, and the cache builder
    (phase2b_off_cache.py) that populates `off_cache` in the first place -- never this off branch.

    `arm != "off"`: loads the frozen checkpoint via `phase2b_load_eval_model` (the single seam) and
    scores it live via `eval_query_loss_heldout`."""
    if arm == "off":
        key = off_cache_key(corpus, ckpt_seed, K, checkpoint_step, hop_set)
        if key not in off_cache:
            raise KeyError(f"off_cache is missing key {key!r} -- the OFF-eval cache "
                            f"(off_lquery_cache-Phase2b.json, sec 16.16.8 chain step 3) must be "
                            f"built BEFORE any hexachotomy/secondary-readout pass runs")
        return off_cache[key]
    ckpt_path = ckpt_path_for(ckpt_dir, arm, corpus, ckpt_seed, checkpoint_step)
    model = phase2b_load_eval_model(ckpt_path, device)
    return eval_query_loss_heldout(model, K, hop_set, corpus, ckpt_seed, checkpoint_step,
                                    batch_size, device)


def build_holds_and_gate_by_checkpoint(ckpt_dir: str, arm: str, corpus: str, off_cache: dict,
                                        K_pair=(32, 20), hop_set: tuple = pft.H_TRAIN,
                                        batch_size: int = 16, device: str = "cpu") -> dict:
    """For ONE (arm, corpus): at each of the 5 trajectory checkpoints, pools the 3 ckpt_seeds' own
    L_query readings (via killer_prediction_readout, arm AND off) into a 3-seed delta_ci_n3 CI at
    BOTH K's, then computes det(32,c)/det(20,c)/holds(c) via phase2_hexachotomy's own primitives.
    Returns {"holds_by_c": {...}, "stage05_pass_by_c": {...}, "det_arm_by_c": {...}, "raw": {...}}
    (raw = every intermediate CI, for disclosure/debugging).

    **stage05_pass_by_c is now UNCONDITIONALLY True at every checkpoint (sec 16.16.3 item 2) --
    the per-checkpoint Stage-0.5 gate is RETIRED for Phase-2b**, mirroring
    `phase2_hexachotomy.totality_check`'s own `always_pass_gate = {c: True for c in CHECKPOINTS}`
    verbatim (L204). Its replacement, the single upfront OFF-floor gate (sec 16.16.6,
    `phase2b_floor_gate_enforce.py`), is evaluated ONCE per corpus BEFORE the 12-cell launch, not
    per-checkpoint inside this classifier -- the dead `gate_json_path_for`/`pge.gate_verdict` read
    is DELETED, not bypassed (verified dead by `phase2_stage_minus1._references`)."""
    K32, K20 = K_pair
    holds_by_c, det_arm_by_c, raw = {}, {}, {}
    for c in phx.CHECKPOINTS:
        per_k_delta = {}
        for K in (K32, K20):
            off_vals, arm_vals = [], []
            for s in CKPT_SEEDS:
                off_val = killer_prediction_readout(ckpt_dir, "off", corpus, s, K, c, off_cache,
                                                      hop_set, batch_size, device)
                arm_val = killer_prediction_readout(ckpt_dir, arm, corpus, s, K, c, off_cache,
                                                      hop_set, batch_size, device)
                off_vals.append(off_val)
                arm_vals.append(arm_val)
            # sec 16.16.5's Delta redefinition: Delta_Lquery(arm,K,c) := L_query(off,K,c) -
            # L_query(arm,K,c) -- positive = arm's loss is LOWER than off's = arm helps. An
            # INTENTIONAL, disclosed argument-order reversal from the old recovered_frac-based
            # delta_ci_n3(arm_vals, off_vals) convention (correct there for a higher-is-better metric).
            per_k_delta[K] = rlp.delta_ci_n3(off_vals, arm_vals)
        det32 = phx.det(per_k_delta[K32]["ci_low"], per_k_delta[K32]["ci_high"])
        det20 = phx.det(per_k_delta[K20]["ci_low"], per_k_delta[K20]["ci_high"])
        holds_by_c[c] = phx.holds(det32, det20, abs(per_k_delta[K32]["mean"]), abs(per_k_delta[K20]["mean"]))
        det_arm_by_c[c] = det32   # det_arm(arm,c) reuses the SAME K=32 delta (sec 16.2.1's own citation)
        raw[c] = {"delta_k32": per_k_delta[K32], "delta_k20": per_k_delta[K20], "det32": det32, "det20": det20}

    stage05_pass_by_c = {c: True for c in phx.CHECKPOINTS}
    return {"holds_by_c": holds_by_c, "stage05_pass_by_c": stage05_pass_by_c,
            "det_arm_by_c": det_arm_by_c, "raw": raw}


def secondary_ood_readout(ckpt_dir: str, arm: str, corpus: str, off_cache: dict, K_pair=(32, 20),
                           batch_size: int = 16, device: str = "cpu") -> dict:
    """sec 16.16.7's held-out-hop (h in {3,4}) generalization readout: the SAME
    eval_query_loss_heldout function at `hop_set=H_TEST_HELD_OUT`, reported as a SIMPLER standalone
    `det(K,c)` table -- NOT folded into a second parallel hexachotomy classification
    (disproportionate for a secondary readout)."""
    K32, K20 = K_pair
    table = {}
    for c in phx.CHECKPOINTS:
        per_k_delta = {}
        for K in (K32, K20):
            off_vals, arm_vals = [], []
            for s in CKPT_SEEDS:
                off_val = killer_prediction_readout(ckpt_dir, "off", corpus, s, K, c, off_cache,
                                                      pft.H_TEST_HELD_OUT, batch_size, device)
                arm_val = killer_prediction_readout(ckpt_dir, arm, corpus, s, K, c, off_cache,
                                                      pft.H_TEST_HELD_OUT, batch_size, device)
                off_vals.append(off_val)
                arm_vals.append(arm_val)
            per_k_delta[K] = rlp.delta_ci_n3(off_vals, arm_vals)
        table[c] = {
            "det32": phx.det(per_k_delta[K32]["ci_low"], per_k_delta[K32]["ci_high"]),
            "det20": phx.det(per_k_delta[K20]["ci_low"], per_k_delta[K20]["ci_high"]),
            "delta_k32": per_k_delta[K32], "delta_k20": per_k_delta[K20],
        }
    return table


def classify_arms(per_arm: dict, agree_by_c: dict) -> dict:
    """sec 16.18.3/16.18.9's registered follow-up fix: classifies BOTH non-off arms
    INDEPENDENTLY -- each arm's OWN `holds_by_c`/`stage05_pass_by_c` (already fully
    computed per-arm by `build_holds_and_gate_by_checkpoint`, one call per arm in
    `analyze_corpus`'s own `per_arm` dict) drives its OWN `classify_trajectory` call,
    never a `global`-as-silent-proxy-for-`per_token` substitution (the exact scoping
    bug sec 16.18.3's hand-derivation table caught: a real `wikitext-mix-ext x
    per_token` TRANSIENT signal, absorbed into `wikitext`'s corpus-level UNRESOLVED
    because only `global`'s `holds_by_c` was ever consulted).

    The terminal-checkpoint `det_arm_global_5000`/`det_arm_per_token_5000`/`agree_5000`
    trio is SHARED across both calls (unchanged from before this fix) -- outcomes #4/#5
    (CONVERGED-EQUIVALENT/UNRESOLVED) are definitionally an equivalence BETWEEN arms,
    not a single-arm quantity (`classify_trajectory`'s own signature takes both arms'
    det_arm values as separate parameters for exactly this reason) -- only the
    `holds_by_c`/`stage05_pass_by_c` inputs that drive outcomes #1-3 (PERSISTENT/
    TRANSIENT/LATE-EMERGENT) were ever silently single-arm-sourced pre-fix.

    Returns `{"global": {...}, "per_token": {...}}`, each value a `classify_trajectory`
    result dict (`{"outcome", "c1", "detail"}`)."""
    det_arm_global_5000 = per_arm["global"]["det_arm_by_c"][phx.TERMINAL_CHECKPOINT]
    det_arm_per_token_5000 = per_arm["per_token"]["det_arm_by_c"][phx.TERMINAL_CHECKPOINT]
    agree_5000 = agree_by_c[phx.TERMINAL_CHECKPOINT]
    return {arm: phx.classify_trajectory(
                holds_by_c=per_arm[arm]["holds_by_c"],
                stage05_pass_by_c=per_arm[arm]["stage05_pass_by_c"],
                det_arm_global_5000=det_arm_global_5000,
                det_arm_per_token_5000=det_arm_per_token_5000,
                agree_5000=agree_5000)
            for arm in ARMS_NON_OFF}


def analyze_corpus(ckpt_dir: str, corpus: str, off_cache_path: str, floor_pinned_path: str,
                    batch_size: int = 16, device: str = "cpu") -> dict:
    """Build-audit FATAL fix: this used to read `off_cache_path` with a bare `json.load`, which
    returns `write_off_lquery_cache`'s own writer ENVELOPE ({design_ref, n_entries, cached_at,
    cached_at_iso, cache: {...}}), never unwrapped -- every real run hit a KeyError inside
    `killer_prediction_readout`'s off branch (the flat `{off_cache_key(...): float}` dict it
    expects was never what `off_cache` actually held). Fixed: reads via
    `phase2b_off_cache.load_off_lquery_cache_verified`, the production reader that (a) unwraps the
    envelope correctly (mirrors `phase2b_off_cache.py`'s own `load_off_lquery_cache`) and (b),
    build-audit MAJOR fix, hard-aborts (raises `phase2b_off_cache.CacheIntegrityFailure`, no
    verdict) if `off_cache_path`'s sha256 does not match the hash pinned in
    `floor_pinned_path`/FLOOR_PINNED-Phase2b.json at pin time -- closing the gap where this
    function (and `killer_prediction_readout`, which it feeds `off_cache` into) consumed the cache
    with zero integrity check, unlike the floor-gate's own already-tamper-evident read path.

    Imports `phase2b_off_cache` LOCALLY (not at module top level): that module already does
    `import phase2_trajectory_analysis as pta` at ITS top level (it calls this module's
    `off_cache_key`/`ckpt_path_for`/`phase2b_load_eval_model`/`eval_query_loss_heldout`) -- a
    top-level `import phase2b_off_cache` here would be a circular module import; deferring to this
    function's own single call site breaks the cycle cleanly without touching either module's
    existing import graph."""
    import phase2b_off_cache as poc
    off_cache = poc.load_off_lquery_cache_verified(off_cache_path, floor_pinned_path)

    per_arm = {arm: build_holds_and_gate_by_checkpoint(ckpt_dir, arm, corpus, off_cache,
                                                          batch_size=batch_size, device=device)
               for arm in ARMS_NON_OFF}
    # agree(c): global's vs per_token's own Delta(K=32,c) CIs overlap.
    agree_by_c = {}
    for c in phx.CHECKPOINTS:
        g = per_arm["global"]["raw"][c]["delta_k32"]
        p = per_arm["per_token"]["raw"][c]["delta_k32"]
        agree_by_c[c] = phx.agree(g["ci_low"], g["ci_high"], p["ci_low"], p["ci_high"])

    # sec 16.18.3/16.18.9 follow-up fix: classify BOTH non-off arms independently (4 verdicts
    # total across the 2 corpora), never using `global` as a silent proxy for `per_token`'s own
    # holds_by_c pattern -- see classify_arms's own docstring for the full rationale and the
    # sec 16.18.3 hand-derivation table this closes.
    classification_by_arm = classify_arms(per_arm, agree_by_c)
    # Backward-compat alias, byte-identical to this field's pre-fix value (it was ALWAYS exactly
    # classify_trajectory() applied to the global arm's own holds_by_c -- classify_arms computes
    # the identical thing for "global", just now alongside "per_token"'s own independent verdict
    # rather than instead of it): `phase2b_chain.sh`'s (and the historical `phase2_chain.sh`'s)
    # own summary step reads `d["classification"]` verbatim -- kept so neither script needs to
    # change to keep working. New callers should prefer `classification_by_arm`.
    classification = classification_by_arm["global"]

    secondary_ood = {arm: secondary_ood_readout(ckpt_dir, arm, corpus, off_cache,
                                                  batch_size=batch_size, device=device)
                      for arm in ARMS_NON_OFF}

    return {"corpus": corpus, "per_arm": per_arm, "agree_by_c": agree_by_c,
            "classification": classification, "classification_by_arm": classification_by_arm,
            "secondary_ood": secondary_ood}


# ===========================================================================
# sec 16.19 (Rev 3, Phase-2b SEED EXTENSION) -- wave-specific n=12 harvest machinery.
# EVERYTHING below is ADDITIVE: analyze_corpus above stays byte-identical, still used verbatim by
# every other corpus/arm combination (sec 16.19.5 item 5's "fork, do not edit a shared production
# path in place" disclosure -- this wave's driver is its OWN function, NOT production analyze_corpus
# invoked blindly, which would attempt a global-arm branch this wave trains no new seeds for).
# ===========================================================================

SEEDEXT_CORPUS = "wikitext-mix-ext"                  # sec 16.19.4 Option B: ONE corpus
SEEDEXT_ARM = "per_token"                            # sec 16.19.4 Option B: the one live-signal arm
SEEDEXT_ARCHIVED_SEEDS = (0, 1, 2)                   # sourced via load_archived_arm_val, NEVER live
SEEDEXT_NEW_SEEDS = tuple(range(3, 12))              # sourced via the guarded live eval path
SEEDEXT_MIN_LIVE_CKPT_SEED = 3                       # the whole-harvest-runtime guard's threshold
SEEDEXT_ANCHOR_K = 32                                # sec 16.19.8: keyed FIRST on K=32, c=2500
SEEDEXT_ANCHOR_CHECKPOINT = 2500
# sec 16.19.8's registered archived n=3 point estimate for the anchor cell (the design's own
# literal, quoted throughout the partition table; the full-precision archived mean is
# -0.4999653498331706, trajectory_wikitext-mix-ext_phase2b.json per_arm.per_token.raw["2500"]
# .delta_k32.mean -- the ROUNDED registered literal is what the decision rules pin).
SEEDEXT_ARCHIVED_POINT = -0.4999
# sec 16.19.5 MINOR-1 (round-2) batch-effect gate thresholds, round-3 MINOR-1's pinned pooled_SE.
SEEDEXT_BATCH_MEAN_SHIFT_K = 2.0                     # |mean(new)-mean(old)| > 2 x pooled_SE
SEEDEXT_BATCH_VAR_RATIO_MAX = 4.0                    # larger sample variance / smaller > 4


def contains_point(ci_low: float, ci_high: float, point: float) -> bool:
    """sec 16.19.8's registered point-in-CI helper (round-3 MAJOR-A): NON-STRICT <= on BOTH sides
    -- an endpoint exactly at `point` counts as containing it (round-4 verify: pins the
    endpoint-exactly-at--0.4999 boundary to bucket (i), never (ii)). Pure, alongside
    phase2_hexachotomy.det (which stays untouched -- its strict/non-strict split at zero is a
    SEPARATE, already-registered convention this helper does not modify)."""
    return ci_low <= point <= ci_high


def classify_seedext_outcome(ci_low: float, ci_high: float,
                              point: float = SEEDEXT_ARCHIVED_POINT) -> dict:
    """sec 16.19.8's 4-outcome MECE partition of the anchor cell's pooled n=12 CI, precedence-
    ordered (i)-(iv) exactly as registered (the 4 conditions are pairwise disjoint by construction
    -- precedence is implementation clarity, not a tie-break; the totality walk is sec 16.19.8's
    own proof, re-enumerated mechanically by phase2b_seedext_stage_minus1.py)."""
    assert ci_low <= ci_high, f"malformed CI: ci_low={ci_low} > ci_high={ci_high}"
    excludes_zero = phx.det(ci_low, ci_high)
    if excludes_zero and ci_high < 0 and contains_point(ci_low, ci_high, point):
        return {"bucket": "(i)", "outcome": "TRANSIENT-CONFIRMED-AT-MAGNITUDE",
                "detail": f"CI excludes zero (negative side) AND contains the archived n=3 point "
                          f"estimate {point}"}
    if excludes_zero and ci_high < 0:
        return {"bucket": "(ii)", "outcome": "TRANSIENT-CONFIRMED-SMALLER",
                "detail": f"CI excludes zero (negative side) AND excludes the archived n=3 point "
                          f"estimate {point} -- a REAL negative causal effect at a DIFFERENT "
                          f"magnitude (attenuation is the a priori likelier sub-direction; a CI "
                          f"entirely MORE negative than {point} is the disclosed other "
                          f"sub-direction, flag explicitly at harvest)"}
    if not excludes_zero:
        return {"bucket": "(iii)", "outcome": "TRANSIENT-REFUTED",
                "detail": "pooled CI straddles/includes zero -- the n=3 window was noise around a "
                          "null (or CI-consistent-with-null) true effect"}
    return {"bucket": "(iv)", "outcome": "NEW-PATTERN(SIGN-FLIP)",
            "detail": "CI excludes zero on the POSITIVE side -- a sign-discipline violation, never "
                      "silently relabeled as 'arm helps' without its own dedicated investigation"}


def load_archived_arm_val(corpus: str, arm: str, ckpt_seed: int, K: int, checkpoint_step: int,
                           hop_set: tuple = pft.H_TRAIN, *, off_cache: dict,
                           trajectory_json: dict) -> float:
    """sec 16.19.5 item 5 (round-2 MAJOR, hop_set-generalized at round-3 MAJOR-B): reconstructs an
    ARCHIVED seed's own arm-side L_query from two already-archived, read-only artifacts -- NEVER a
    model load, NEVER eval_query_loss_heldout/phase2_seed, for EITHER hop_set:

        old_arm_val = off_cache[off_cache_key(...)] - trajectory_deltas[ckpt_seed]

    (deltas[i] = off_vals[i] - arm_vals[i], sec 16.16.5's Delta redefinition.) The trajectory
    sub-block is selected per the registered table: per_arm[arm]["raw"] for the primary hop_set
    H_TRAIN=(1,2), secondary_ood[arm] for H_TEST_HELD_OUT=(3,4) (one function serving both
    readouts, not a fork). Failure mode mirrors killer_prediction_readout's own off-cache branch:
    KeyError on ANY missing key in EITHER artifact, never a silent fallback to a live eval call.

    Build-time note (disclosed): `off_cache`/`trajectory_json` are keyword-only -- the design's
    registered positional order with a defaulted `hop_set` mid-signature is not otherwise
    expressible in Python; no Rev-2 call site existed in code (Rev 2 was design-only), so no
    existing caller changes."""
    if ckpt_seed not in SEEDEXT_ARCHIVED_SEEDS:
        raise KeyError(
            f"load_archived_arm_val serves ARCHIVED seeds {SEEDEXT_ARCHIVED_SEEDS} only, got "
            f"ckpt_seed={ckpt_seed} -- new seeds must go through the (guarded) live eval path, "
            f"never a loader that would silently index a 3-element archived deltas list")
    hop_t = tuple(hop_set)
    if hop_t == tuple(pft.H_TRAIN):
        try:
            traj_block = trajectory_json["per_arm"][arm]["raw"]
        except KeyError:
            raise KeyError(f"trajectory_json has no per_arm[{arm!r}].raw block")
    elif hop_t == tuple(pft.H_TEST_HELD_OUT):
        try:
            traj_block = trajectory_json["secondary_ood"][arm]
        except KeyError:
            raise KeyError(f"trajectory_json has no secondary_ood[{arm!r}] block")
    else:
        raise ValueError(f"hop_set={hop_set!r} is neither the registered primary {pft.H_TRAIN} nor "
                          f"secondary {pft.H_TEST_HELD_OUT} hop set")

    key = off_cache_key(corpus, ckpt_seed, K, checkpoint_step, hop_set)
    if key not in off_cache:
        raise KeyError(f"off_cache is missing archived key {key!r} -- refusing to reconstruct "
                        f"(never recomputes live)")
    c_key, delta_key = str(checkpoint_step), f"delta_k{K}"
    if c_key not in traj_block or delta_key not in traj_block[c_key]:
        raise KeyError(f"trajectory block is missing [{c_key!r}][{delta_key!r}] -- refusing to "
                        f"reconstruct (never recomputes live)")
    return off_cache[key] - traj_block[c_key][delta_key]["deltas"][ckpt_seed]


def install_seedext_live_eval_guard() -> None:
    """sec 16.19.5 item 5's WHOLE-HARVEST-RUNTIME no-live-recompute guard (scope re-pinned at
    round-3 MAJOR-B): installed ONCE, at this wave's own harvest driver's entry point, active for
    the process's ENTIRE remaining lifetime -- never call-site-local. Rebinds this module's own
    `eval_query_loss_heldout` global to a wrapper asserting `ckpt_seed >= 3` on EVERY call. This
    ONE seam mechanically covers BOTH readout loops by construction: `eval_query_loss_heldout` has
    no `arm` parameter of its own and is the single function both `killer_prediction_readout`'s
    non-off branch (primary) and `secondary_ood_readout` (OOD) route through. Idempotent (a second
    install is a no-op, never a double-wrap). Production NEVER uninstalls this guard -- the Stage
    -1 suite restores the module global after its own negative tests purely as test hygiene,
    disclosed there."""
    global eval_query_loss_heldout
    if getattr(eval_query_loss_heldout, "_seedext_whole_runtime_guard", False):
        return
    inner = eval_query_loss_heldout

    def _seedext_guarded_eval_query_loss_heldout(model, K, hop_set, corpus, ckpt_seed,
                                                   checkpoint_step, batch_size=16, device="cpu"):
        assert ckpt_seed >= SEEDEXT_MIN_LIVE_CKPT_SEED, (
            f"SEEDEXT LIVE-EVAL GUARD (sec 16.19.5 item 5, whole-harvest-runtime): live "
            f"eval_query_loss_heldout called with ARCHIVED ckpt_seed={ckpt_seed} (< "
            f"{SEEDEXT_MIN_LIVE_CKPT_SEED}) at hop_set={hop_set} -- archived seeds must be "
            f"sourced via load_archived_arm_val/the OFF-eval cache, NEVER re-scored live (the "
            f"_MAX_CKPT_SEED 10->12 bump changed every EVAL-kind seed, so a live call would "
            f"silently draw a DIFFERENT held-out episode than produced the archived values)")
        return inner(model, K, hop_set, corpus, ckpt_seed, checkpoint_step, batch_size, device)

    _seedext_guarded_eval_query_loss_heldout._seedext_whole_runtime_guard = True
    _seedext_guarded_eval_query_loss_heldout._seedext_guard_inner = inner
    eval_query_loss_heldout = _seedext_guarded_eval_query_loss_heldout


def _sample_mean_sd(values: list) -> tuple:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def batch_effect_gate(old_off_vals: list, new_off_vals: list) -> dict:
    """sec 16.19.5 MINOR-1 (round-2) pre-pooling gate with the round-3-MINOR-1-pinned pooled_SE:
    compares the 3 archived vs 9 new OFF seeds' own mean/spread BEFORE concatenation. Flags iff
    |mean(new) - mean(old)| > 2 x pooled_SE, where pooled_SE := sqrt(SE_old^2 + SE_new^2) (the
    standard-error-OF-A-DIFFERENCE / Welch form -- correct for unequal n, NOT the equal-variance
    "pooled SD" two-sample formula), OR variance_ratio > 4 (larger sample variance over the
    smaller, either direction; a zero-variance cohort against a nonzero one counts as a flag).
    Scope: OFF-arm L_query cohorts ONLY (per the registered single-risk scoping)."""
    mean_old, sd_old = _sample_mean_sd(old_off_vals)
    mean_new, sd_new = _sample_mean_sd(new_off_vals)
    se_old = sd_old / math.sqrt(len(old_off_vals))
    se_new = sd_new / math.sqrt(len(new_off_vals))
    pooled_se = math.sqrt(se_old ** 2 + se_new ** 2)
    mean_shift = abs(mean_new - mean_old)
    mean_shift_flag = mean_shift > SEEDEXT_BATCH_MEAN_SHIFT_K * pooled_se
    var_old, var_new = sd_old ** 2, sd_new ** 2
    lo, hi = min(var_old, var_new), max(var_old, var_new)
    if hi == 0.0:
        var_ratio = 1.0
    elif lo == 0.0:
        var_ratio = float("inf")
    else:
        var_ratio = hi / lo
    var_ratio_flag = var_ratio > SEEDEXT_BATCH_VAR_RATIO_MAX
    return {"mean_old": mean_old, "mean_new": mean_new, "sd_old": sd_old, "sd_new": sd_new,
            "se_old": se_old, "se_new": se_new, "pooled_se": pooled_se, "mean_shift": mean_shift,
            "mean_shift_flag": mean_shift_flag, "var_ratio": var_ratio,
            "var_ratio_flag": var_ratio_flag, "flagged": bool(mean_shift_flag or var_ratio_flag)}


def _seedext_table_for_hop_set(ckpt_dir: str, off_cache: dict, trajectory_json: dict,
                                hop_set: tuple, batch_size: int, device: str) -> dict:
    """One n=12 table (all 5 checkpoints x both K's) for ONE hop_set -- the SAME concatenation
    pattern applied identically to the primary (H_TRAIN) and OOD (H_TEST_HELD_OUT) readouts
    (round-3 MAJOR-B's symmetry). Per cell: 12 OFF values (3 archived + 9 new, ALL via the
    OFF-eval cache through killer_prediction_readout's cache-only off branch), 12 arm values (3
    archived via load_archived_arm_val + 9 new via the guarded live path), the batch-effect gate
    on the OFF cohorts, then EITHER one pooled delta_ci_n (gate clear) OR the two cohorts' own
    separate CIs (gate flagged -- never silently pooled; the sec 16.19.8 partition is NOT applied
    to a flagged cell, its precondition -- a validly-pooled CI -- fails)."""
    corpus, arm = SEEDEXT_CORPUS, SEEDEXT_ARM
    all_seeds = SEEDEXT_ARCHIVED_SEEDS + SEEDEXT_NEW_SEEDS
    table = {}
    for c in phx.CHECKPOINTS:
        per_k = {}
        for K in (32, 20):
            off_vals = [killer_prediction_readout(ckpt_dir, "off", corpus, s, K, c, off_cache,
                                                    hop_set, batch_size, device)
                        for s in all_seeds]
            old_arm_vals = [load_archived_arm_val(corpus, arm, s, K, c, hop_set,
                                                    off_cache=off_cache,
                                                    trajectory_json=trajectory_json)
                            for s in SEEDEXT_ARCHIVED_SEEDS]
            new_arm_vals = [killer_prediction_readout(ckpt_dir, arm, corpus, s, K, c, off_cache,
                                                        hop_set, batch_size, device)
                            for s in SEEDEXT_NEW_SEEDS]
            arm_vals = old_arm_vals + new_arm_vals
            gate = batch_effect_gate(off_vals[:3], off_vals[3:])
            entry = {"batch_gate": gate, "flagged": gate["flagged"]}
            if gate["flagged"]:
                entry["pooled"] = None
                entry["old_cohort"] = rlp.delta_ci_n(off_vals[:3], arm_vals[:3])
                entry["new_cohort"] = rlp.delta_ci_n(off_vals[3:], arm_vals[3:])
            else:
                entry["pooled"] = rlp.delta_ci_n(off_vals, arm_vals)
            per_k[K] = entry
        cell = {"delta_k32": per_k[32], "delta_k20": per_k[20]}
        if not per_k[32]["flagged"] and not per_k[20]["flagged"]:
            det32 = phx.det(per_k[32]["pooled"]["ci_low"], per_k[32]["pooled"]["ci_high"])
            det20 = phx.det(per_k[20]["pooled"]["ci_low"], per_k[20]["pooled"]["ci_high"])
            cell.update({"det32": det32, "det20": det20,
                          "holds": phx.holds(det32, det20, abs(per_k[32]["pooled"]["mean"]),
                                              abs(per_k[20]["pooled"]["mean"]))})
        else:
            cell.update({"det32": None, "det20": None, "holds": None})
        table[c] = cell
    return table


def analyze_corpus_seedext(ckpt_dir: str, off_cache_path: str, floor_pinned_path: str,
                            archived_trajectory_path: str, batch_size: int = 16,
                            device: str = "cpu") -> dict:
    """sec 16.19.5 item 5's wave-specific n=12 harvest driver -- NOT production analyze_corpus:
    (a) NEVER touches arm="global" (zero new seeds this wave; that arm stays at its own n=3 sec
    16.18 verdict); (b) wikitext-mix-ext x per_token ONLY; (c) archived seeds 0-2 via
    load_archived_arm_val / the OFF-eval cache, new seeds 3-11 via the whole-runtime-guarded live
    path -- the two paths partition {0..11} completely, for the primary AND OOD readouts
    independently. The guard installs HERE, first, for the process's whole lifetime.

    `floor_pinned_path` is the WAVE'S OWN pin file (FLOOR_PINNED-Phase2b-n12-wikitext.json), whose
    off_cache_sha256 records the EXTENDED cache -- the same verified-read discipline
    analyze_corpus already uses, pointed at this wave's own pin."""
    install_seedext_live_eval_guard()

    import phase2b_off_cache as poc   # local import -- same circular-import note as analyze_corpus
    off_cache = poc.load_off_lquery_cache_verified(off_cache_path, floor_pinned_path)
    with open(archived_trajectory_path) as f:
        trajectory_json = json.load(f)

    primary = _seedext_table_for_hop_set(ckpt_dir, off_cache, trajectory_json, pft.H_TRAIN,
                                          batch_size, device)
    secondary = _seedext_table_for_hop_set(ckpt_dir, off_cache, trajectory_json,
                                            pft.H_TEST_HELD_OUT, batch_size, device)

    anchor_entry = primary[SEEDEXT_ANCHOR_CHECKPOINT][f"delta_k{SEEDEXT_ANCHOR_K}"]
    if anchor_entry["flagged"]:
        # Fixed sec 16.19.5 routing (round-4 MINOR): a flagged anchor cell sits OUTSIDE the sec
        # 16.19.8 partition entirely -- no bucket (i)-(iv) is declared; the two cohorts' own
        # separate CIs and the flag itself are the reported outcome.
        anchor = {"bucket": None, "outcome": "BATCH-EFFECT-FLAGGED",
                  "detail": "the batch-effect pre-pooling gate flagged the anchor cell's OFF "
                            "cohorts -- no pooled n=12 CI is decision-grade here; the two cohorts "
                            "are reported separately (never silently pooled)",
                  "ci_low": None, "ci_high": None}
    else:
        pooled = anchor_entry["pooled"]
        anchor = dict(classify_seedext_outcome(pooled["ci_low"], pooled["ci_high"]))
        anchor.update({"ci_low": pooled["ci_low"], "ci_high": pooled["ci_high"]})
    anchor.update({"K": SEEDEXT_ANCHOR_K, "checkpoint_step": SEEDEXT_ANCHOR_CHECKPOINT,
                    "hop_set": f"{pft.H_TRAIN[0]}-{pft.H_TRAIN[1]}",
                    "archived_point": SEEDEXT_ARCHIVED_POINT})

    return {"design_ref": "REASONING_LINK_DESIGN.md sec 16.19 (Rev 3, DESIGN-CLEARED-FOR-BUILD, "
                           "round-4 verify)",
            "corpus": SEEDEXT_CORPUS, "arm": SEEDEXT_ARM,
            "seeds": {"archived": list(SEEDEXT_ARCHIVED_SEEDS), "new": list(SEEDEXT_NEW_SEEDS)},
            "anchor": anchor, "primary": primary, "secondary_ood": secondary}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["openr1-mix-ext", "wikitext-mix-ext"], required=True)
    ap.add_argument("--ckpt-dir", type=str, required=True)
    ap.add_argument("--off-cache", type=str, required=True,
                     help="off_lquery_cache-Phase2b.json path (sec 16.16.8 chain step 3)")
    ap.add_argument("--floor-pinned", type=str, required=True,
                     help="FLOOR_PINNED-Phase2b.json path (sec 16.16.6 chain step 4) -- build-audit "
                          "MAJOR fix: --off-cache's sha256 is verified against the hash pinned here "
                          "before any verdict is computed (phase2b_off_cache."
                          "load_off_lquery_cache_verified)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seedext", action="store_true",
                     help="sec 16.19: run the wave-specific n=12 seed-extension harvest driver "
                          "(analyze_corpus_seedext -- wikitext-mix-ext x per_token ONLY) instead "
                          "of production analyze_corpus. Requires --archived-trajectory, and "
                          "--floor-pinned must be the wave's OWN FLOOR_PINNED-Phase2b-n12-"
                          "wikitext.json (its off_cache_sha256 records the EXTENDED cache).")
    ap.add_argument("--archived-trajectory", type=str, default=None,
                     help="sec 16.19.5 item 5: path to the ARCHIVED, read-only "
                          "trajectory_wikitext-mix-ext_phase2b.json the archived-values loader "
                          "reconstructs seeds 0-2 from (required with --seedext)")
    args = ap.parse_args()

    if args.seedext:
        assert args.corpus == SEEDEXT_CORPUS, (
            f"--seedext is registered for {SEEDEXT_CORPUS} ONLY (sec 16.19.4 Option B), got "
            f"--corpus {args.corpus}")
        assert args.archived_trajectory, "--seedext requires --archived-trajectory"
        result = analyze_corpus_seedext(args.ckpt_dir, args.off_cache, args.floor_pinned,
                                          args.archived_trajectory, batch_size=args.batch_size,
                                          device=args.device)
        print(json.dumps({"corpus": result["corpus"], "arm": result["arm"],
                           "anchor": result["anchor"]}, indent=2, default=str))
    else:
        result = analyze_corpus(args.ckpt_dir, args.corpus, args.off_cache, args.floor_pinned,
                                 batch_size=args.batch_size, device=args.device)
        print(json.dumps({"corpus": result["corpus"], "classification": result["classification"],
                           "classification_by_arm": result["classification_by_arm"],
                           "agree_by_c": result["agree_by_c"]}, indent=2, default=str))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
