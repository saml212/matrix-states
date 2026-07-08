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

**Scoping decision carried forward unchanged from Phase-2 (sec 16.2.1's own
"resolved during BUILD" convention, flagged for the independent audit's
attention):** this module classifies ONE trajectory PER CORPUS (2 total:
openr1-mix-ext, wikitext-mix-ext), each built from a 3-seed-pooled
`delta_ci_n3` CI at every checkpoint, for BOTH arms (global, per_token).

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

    # holds(c) at the CORPUS level is the SAME condition regardless of which non-off arm is being
    # read -- build-time resolution (disclosed, carried forward from Phase-2 unchanged): use the
    # GLOBAL arm's own holds_by_c as the trajectory's holds() input (sec 16.2.1's H_LINK-A causal
    # claim is itself global-arm-centric).
    classification = phx.classify_trajectory(
        holds_by_c=per_arm["global"]["holds_by_c"],
        stage05_pass_by_c=per_arm["global"]["stage05_pass_by_c"],
        det_arm_global_5000=per_arm["global"]["det_arm_by_c"][phx.TERMINAL_CHECKPOINT],
        det_arm_per_token_5000=per_arm["per_token"]["det_arm_by_c"][phx.TERMINAL_CHECKPOINT],
        agree_5000=agree_by_c[phx.TERMINAL_CHECKPOINT])

    secondary_ood = {arm: secondary_ood_readout(ckpt_dir, arm, corpus, off_cache,
                                                  batch_size=batch_size, device=device)
                      for arm in ARMS_NON_OFF}

    return {"corpus": corpus, "per_arm": per_arm, "agree_by_c": agree_by_c,
            "classification": classification, "secondary_ood": secondary_ood}


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
    args = ap.parse_args()

    result = analyze_corpus(args.ckpt_dir, args.corpus, args.off_cache, args.floor_pinned,
                             batch_size=args.batch_size, device=args.device)
    print(json.dumps({"corpus": result["corpus"], "classification": result["classification"],
                       "agree_by_c": result["agree_by_c"]}, indent=2, default=str))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
