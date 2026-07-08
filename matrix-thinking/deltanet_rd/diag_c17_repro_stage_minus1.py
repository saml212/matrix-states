"""diag_c17_repro_stage_minus1.py -- KEY_ANCHORING_SCALING_DRAFT.md sec
15.24.5's BLOCKING Stage -1 self-test suite for the C17 eval-admission
repro instrument (Rev 4, DESIGN-CLEARED-FOR-BUILD). All 12 registered
items, run to completion per this project's own standing "assertion has
teeth" rule -- not merely written.

    DRY_RUN_BYPASS=1 python3 diag_c17_repro_stage_minus1.py

(the repo's pre-train-gate hook pattern-matches any `python *.py`
invocation; DRY_RUN_BYPASS=1 is the correct, sanctioned bypass here --
every item this script actually RUNS is pure-Python/CPU-only, no GPU.)

BOX-DEFERRED items (registered here, NOT run this session): items 1, 2, 3,
4, 9, and PART of item 5 genuinely need either (a) CUDA + `fla` (flash-
linear-attention) to import run_deltanet_rd.py/model_rd.py at all -- an
ALREADY-ESTABLISHED constraint in this codebase, not new to this build
(see smoke_keyanchor_scaling.py's own "model_rd.py cannot be imported in
this fla-less CPU sandbox" note, re-verified this session: `import
model_rd` raises `ModuleNotFoundError: No module named 'fla'` in this
repo's own `.venv`), or (b) a REAL trained checkpoint / archived result
JSON this dev machine does not have (those live on the H100 box). Each
BOX-DEFERRED item registers its own box-side run requirement below (see
`_defer()` calls) rather than being silently skipped.

Items 6, 7, 8, 10, 11, and 12 are genuinely CPU-only logic tests against
diag_c17_repro_analysis.py's own step functions -- called DIRECTLY (never
through main()/the CLI), per that module's own registered unit-test-
isolation design decision -- and DO run to completion this session,
including item 11's own tokenizer-backed two-seeds-trap fixture (GPT-2
tokenizer load confirmed working, network/cache available in this
sandbox).
"""
from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import diag_c17_repro_analysis as A                    # noqa: E402 (fla-free)
import grammar_rd as grd                                # noqa: E402 (fla-free)

RESULTS_DIR = os.path.join(HERE, "results", "keyanchor_scaling_c17repro")

FAILURES: list[str] = []
BOX_DEFERRED: list[dict] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def _defer(item: str, requirement: str) -> None:
    print(f"[{item}] BOX-DEFERRED -- {requirement}", flush=True)
    BOX_DEFERRED.append({"item": item, "requirement": requirement})


# ---------------------------------------------------------------------------
# Shared fixture kit -- REAL, tokenizer-backed pools (no CUDA/fla needed;
# `grammar_rd.build_entity_pools` only needs `transformers` + `torch`).
# ---------------------------------------------------------------------------

_TOKENIZER = None
_POOLS = None


def _pools():
    global _TOKENIZER, _POOLS
    if _POOLS is None:
        _TOKENIZER = grd.load_gpt2_tokenizer()
        _POOLS = A.reconstruct_pools_offline(_TOKENIZER)
    return _POOLS


def _make_synthetic_event(pools, seed: int, step: int, hop: int, batch_idx: int,
                           k: int = 8, b: int = 4, d: int = 16, bad_rows=()) -> dict:
    """ONE synthetic C17 fallback event dict with REAL heldout entity ids
    (passes Step 0b cleanly) unless `bad_rows` names row indices to poison
    with a TRAIN id instead (the Step 0b violation fixture)."""
    heldout_ids = pools.heldout_name_ids[:k]
    key_ids = heldout_ids.unsqueeze(0).repeat(b, 1).clone()
    for row in bad_rows:
        key_ids[row, 0] = pools.train_name_ids[0]
    gen = torch.Generator().manual_seed(seed * 1_000_000 + step * 1000 + hop * 100 + batch_idx)
    k_eff_raw = torch.nn.functional.normalize(torch.randn(b, k, d, generator=gen), dim=-1)
    resid = torch.rand(b, generator=gen) * 0.005    # well below GATE2_RESID_TOL (0.01) by default
    return {"seed": seed, "step": step, "hop": hop, "batch_idx": batch_idx,
            "key_ids": key_ids, "k_eff_raw": k_eff_raw, "k_blend_raw": k_eff_raw.clone(),
            "resid": resid}


def _make_launch(pools, seed: int, events: list, anchor_train_ids=None,
                  tf32_matmul: bool = False, tf32_cudnn: bool = False) -> dict:
    return {"seed": seed, "ckpt_dir": f"<synthetic:seed{seed}>", "events": events,
            "tf32_matmul": tf32_matmul, "tf32_cudnn": tf32_cudnn,
            "anchor_train_ids": (anchor_train_ids if anchor_train_ids is not None
                                  else pools.train_name_ids.clone())}


# ===========================================================================
# Item 1 -- Bitwise-identical-trajectory smoke. BOX-DEFERRED.
# ===========================================================================

def item_1_bitwise_identical_trajectory_smoke():
    _defer("item 1: bitwise-identical-trajectory smoke",
           "run_deltanet_rd.py's train()/smoke() needs `model_rd` (imports "
           "`fla.modules.ShortConvolution` at module scope, CUDA-oriented) to even import -- "
           "genuinely box-only. Box-side requirement: launch a 50-step, fixed-seed mini run "
           "TWICE (--c17-repro-telemetry ON vs OFF, otherwise identical CLI), assert bitwise-"
           "identical trajectory/loss curve (sec 15.24.2's own required Wave -1 smoke).")


# ===========================================================================
# Item 2 -- Dump-fires-exactly-at-fallback smoke. BOX-DEFERRED.
# ===========================================================================

def item_2_dump_fires_exactly_at_fallback():
    _defer("item 2: dump-fires-exactly-at-fallback smoke",
           "_dump_c17_fallback_event (run_deltanet_rd.py) is exercised via a stand-in `model` "
           "object whose geo3_last_* attributes come from model_rd.geo3_orthogonalize_logged -- "
           "importing model_rd needs `fla`, genuinely box-only. Box-side requirement: (a) a "
           "hand-built near-duplicate K-row batch forcing fallback_triggered=True -> assert the "
           "dump sink receives EXACTLY 1 entry; (b) a well-conditioned/orthonormal-ish batch -> "
           "assert EXACTLY 0 entries.")


# ===========================================================================
# Item 3 -- Tensor-shape assertions. BOX-DEFERRED.
# ===========================================================================

def item_3_tensor_shape_assertions():
    _defer("item 3: tensor-shape assertions",
           "needs a REAL dumped event (produced by the same box-only path as item 2). Box-side "
           "requirement: k_eff_raw/k_blend_raw dumps == (B_batch, 84, 96), resid dump == "
           "(B_batch,), all CPU, fp32, detached (requires_grad=False).")


# ===========================================================================
# Item 4 -- Determinism cross-check, per-launch. BOX-DEFERRED.
# ===========================================================================

def item_4_determinism_cross_check():
    _defer("item 4: determinism cross-check, per-launch",
           "needs a REAL full_step<N>.pt state_dict snapshot + a REAL live C17 dump, both "
           "produced by an actual training launch (CUDA + fla). Box-side requirement: run ONCE "
           "PER LAUNCH (primary seed 1940, and again for each contingency seed if fired) -- load "
           "the snapshot into a fresh model instance, rebuild eval_gen = seed + 10_000 + step, "
           "replay the m2/m3/c17/c19 pool-call order, assert BYTE-IDENTICAL key_ids/k_eff_raw "
           "vs. that launch's own live dump.")


# ===========================================================================
# Item 5 -- sha256/config cross-check. PARTIAL: build_cmd()-mechanics half
# runs locally (fla-free, uses run_deltanet_rd_exactness_sweep.py directly);
# the cross-check against the REAL archived K84/seed=1940 result JSON is
# BOX-DEFERRED (that JSON lives on the H100 box, not in this checkout).
# ===========================================================================

_EXPECTED_ARCH_FLAGS = {
    "--K": "84", "--seed": "1940", "--use-geo3": None, "--geo3-n-iter": "20",
    "--geo3-resid-tol": "0.01", "--anchor-active": None, "--anchor-lambda-mode": "learned",
    "--drift-probe": None, "--rev7-engagement": None, "--d-state": "96",
}


def _cmd_flag_map(cmd: list) -> dict:
    m, i = {}, 0
    while i < len(cmd):
        tok = cmd[i]
        if tok.startswith("--"):
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                m[tok] = cmd[i + 1]
                i += 2
            else:
                m[tok] = None
                i += 1
        else:
            i += 1
    return m


def item_5_sha256_config_cross_check():
    import run_deltanet_rd_exactness_sweep as rdx
    spec = rdx._keyanchor_scaling_spec(84, 1940, 96)
    cmd = rdx.build_cmd(spec, "results/deltanet_rd_exactness/wavekeyanchor-scaling-wide", 7200)
    flags = _cmd_flag_map(cmd)
    ok = all(flags.get(k) == v for k, v in _EXPECTED_ARCH_FLAGS.items())
    detail = f"build_cmd(K=84,seed=1940,d_state=96) carries every sec 15.24.2-pinned arch flag: {ok}"
    _report("item 5 (LOCAL HALF): build_cmd()'s own K84/seed=1940 command carries every "
            "sec 15.24.2-pinned architecture flag", ok, detail)

    # Field-diff vs. an already-archived SIBLING seed for the SAME K=84 cell family
    # (1941, KEYANCHOR_SCALING_SEEDS_BY_D_K[96][84]'s own second seed) -- mirrors
    # run_k69_s1733_contingency.py's own token-diff-vs-archived-seed pattern: proves
    # build_cmd() is deterministic and the ONLY sanctioned delta across two calls at
    # the same (K, d_state) is the seed-derived tokens (--seed, --out, --ckpt-dir).
    sib_spec = rdx._keyanchor_scaling_spec(84, 1941, 96)
    sib_cmd = rdx.build_cmd(sib_spec, "results/deltanet_rd_exactness/wavekeyanchor-scaling-wide", 7200)

    def _normalize(c, seed):
        return [tok.replace(str(seed), "SEED") for tok in c]

    diff_ok = _normalize(cmd, 1940) == _normalize(sib_cmd, 1941)
    _report("item 5 (LOCAL HALF): build_cmd(seed=1940) vs build_cmd(seed=1941) token-diff -- "
            "ONLY seed-derived tokens differ", diff_ok,
            f"normalized command lists equal: {diff_ok}")

    _defer("item 5 (BOX HALF): sha256/config cross-check vs the REAL archived JSON",
           "the archived K84/seed=1940 result JSON (its own exactness_config block) lives on "
           "the H100 box, not in this checkout. Box-side requirement: diff build_cmd()'s "
           "architecture-relevant fields (K, d_state, seed, geo3_n_iter, geo3_resid_tol, "
           "anchor_active, anchor_lambda_mode, drift_probe, H_extra) against that JSON's own "
           "exactness_config block, by hand, recorded in the launch log.")


# ===========================================================================
# Item 6 -- Step -1 NO-REPRO-guard negative test. RUNS.
# ===========================================================================

def item_6_step_neg1_no_repro_guard():
    pools = _pools()

    # Raw-function teeth: the <3 gate itself.
    empty_guard = A.step_neg1_event_guard([])
    _report("item 6a: step_neg1_event_guard([]) refuses (empty sink)", not empty_guard["cleared"],
            f"n_distinct_events={empty_guard['n_distinct_events']}")

    two_events = [
        _make_synthetic_event(pools, seed=1940, step=2000, hop=1, batch_idx=0),
        _make_synthetic_event(pools, seed=1940, step=4000, hop=1, batch_idx=0),
    ]
    two_guard = A.step_neg1_event_guard(two_events)
    _report("item 6b: step_neg1_event_guard(<2 events>) refuses (below <3 minimum)",
            not two_guard["cleared"], f"n_distinct_events={two_guard['n_distinct_events']}")

    # Full-precedence teeth: BOTH cases emit NO-REPRO and refuse every other verdict.
    for label, events in (("empty", []), ("2-event", two_events)):
        launch = _make_launch(pools, seed=1940, events=events)
        result = A.run_full_precedence([launch], _TOKENIZER, k=8)
        ok = (result.get("verdict") == A.NO_REPRO
              and "step_0a" not in result and "step_1" not in result and "step_2" not in result)
        _report(f"item 6c ({label} sink): run_full_precedence emits NO-REPRO, refuses "
                f"REAL-CAPACITY-BOUNDARY/INSTRUMENT-BUG/TOLERANCE-MISCALIBRATION",
                ok, f"verdict={result.get('verdict')!r}")


# ===========================================================================
# Item 7 -- Step 1's two-level floor boundary test. RUNS (pure logic, no NS
# recompute -- exercises _apply_two_level_floor directly).
# ===========================================================================

def item_7_step1_floor_boundary():
    E1, E2, E3, E4 = (1940, 2000, 1, 0), (1940, 4000, 1, 0), (1940, 6000, 1, 0), (1940, 8000, 1, 0)

    # (a) exactly 1 anomalous episode among >=3 total events -> excluded, disclosed, NOT dispositive.
    fx_a = [{"event": E1, "row_idx": 3}]
    r_a = A._apply_two_level_floor(fx_a)
    ok_a = (not r_a["floor_met"]) and r_a["excluded_episodes"] == [E1 + (3,)]
    _report("item 7a: exactly 1 anomalous episode -> flagged, EXCLUDED, named "
            "(seed,step,hop,batch_idx,row_idx), NOT dispositive", ok_a, str(r_a))

    # (b) E2: 4-event sink, 3 anomalous episodes ALL within ONE event -> floor NOT met.
    fx_b = [{"event": E1, "row_idx": 0}, {"event": E1, "row_idx": 1}, {"event": E1, "row_idx": 2}]
    r_b = A._apply_two_level_floor(fx_b)
    ok_b = (not r_b["floor_met"]) and r_b["n_distinct_events"] == 1 and len(r_b["excluded_episodes"]) == 3
    _report("item 7b (E2): 3 anomalous episodes, 1 distinct event -> floor NOT met, all 3 excluded",
            ok_b, str(r_b))

    # (c) E3: 3-event sink, 3 anomalous episodes, ONE per event -> floor MET, dispositive.
    fx_c = [{"event": E1, "row_idx": 0}, {"event": E2, "row_idx": 0}, {"event": E3, "row_idx": 0}]
    r_c = A._apply_two_level_floor(fx_c)
    ok_c = r_c["floor_met"] and r_c["excluded_episodes"] == []
    _report("item 7c (E3): 3 anomalous episodes across 3 distinct events -> floor MET, dispositive",
            ok_c, str(r_c))

    # (d) exactly 2 anomalous episodes in 2 distinct events (general 2-and-2 case) -> floor MET.
    fx_d = [{"event": E1, "row_idx": 0}, {"event": E4, "row_idx": 1}]
    r_d = A._apply_two_level_floor(fx_d)
    ok_d = r_d["floor_met"]
    _report("item 7d: exactly 2 anomalous episodes in 2 distinct events -> floor MET", ok_d, str(r_d))


# ===========================================================================
# Item 8 -- 0b-precedes-Step-1 enforced-abort negative test (+ Rev3 E4 sub-
# case). RUNS.
# ===========================================================================

def item_8_0b_precedes_step_neg1_enforced_abort():
    pools = _pools()

    # E1: a 2-EVENT sink (below Step -1's <3 minimum), 2 pool-membership-mismatch
    # episodes, one per event -> INSTRUMENT-BUG (not NO-REPRO).
    e1_events = [
        _make_synthetic_event(pools, seed=1940, step=2000, hop=1, batch_idx=0, bad_rows=(0,)),
        _make_synthetic_event(pools, seed=1940, step=4000, hop=1, batch_idx=0, bad_rows=(1,)),
    ]
    r1 = A.run_full_precedence([_make_launch(pools, 1940, e1_events)], _TOKENIZER, k=8)
    ok1 = r1.get("verdict") == A.INSTRUMENT_BUG and r1.get("verdict_source") == "step_0b"
    _report("item 8a (E1): 2-event sink, 2 pool-mismatch episodes -> INSTRUMENT-BUG (0b runs "
            "BEFORE, independent of, Step -1's <3 gate)", ok1, f"verdict={r1.get('verdict')!r}")

    # E4/"state 3": 5-event sink, EXACTLY 1 pool-mismatch episode (4 clean + 1 mismatch)
    # -> INSTRUMENT-BUG, NOT REAL-CAPACITY-BOUNDARY/TOLERANCE-MISCALIBRATION on the 4 clean.
    e4_events = [_make_synthetic_event(pools, seed=1940, step=2000 * (i + 1), hop=1, batch_idx=0)
                 for i in range(4)]
    e4_events.append(_make_synthetic_event(pools, seed=1940, step=10000, hop=1, batch_idx=0,
                                            bad_rows=(2,)))
    r4 = A.run_full_precedence([_make_launch(pools, 1940, e4_events)], _TOKENIZER, k=8)
    ok4 = r4.get("verdict") == A.INSTRUMENT_BUG and r4.get("verdict_source") == "step_0b"
    _report("item 8b (E4/'state 3'): 5-event sink, EXACTLY 1 pool-mismatch episode (4 clean + 1) "
            "-> INSTRUMENT-BUG, refuses REAL-CAPACITY-BOUNDARY/TOLERANCE-MISCALIBRATION on the "
            "4 clean remainder", ok4, f"verdict={r4.get('verdict')!r}")


# ===========================================================================
# Item 9 -- Batched-vs-singleton offline recompute comparison. BOX-DEFERRED.
# ===========================================================================

def item_9_batched_vs_singleton_recompute():
    _defer("item 9: batched-vs-singleton offline recompute comparison",
           "needs model_rd.newton_schulz_orthogonalize (fla-dependent import), genuinely "
           "box-only. Box-side requirement: on a hand-built (128,84,96) tensor with one row "
           "engineered near the resid=0.01 threshold, run newton_schulz_orthogonalize once as "
           "ONE BATCHED call on the full tensor (Step 1's own pinned method) and once as 128 "
           "singleton (1,84,96) calls; assert the batched call's resid_offline[row_idx] is what "
           "diag_c17_repro_analysis.py's Step 1 actually uses, and DISCLOSE any row where the "
           "two methods' resid values differ.")


# ===========================================================================
# Item 10 -- Cross-marker negative test (E5/"state 6"). RUNS.
# ===========================================================================

def item_10_cross_marker_negative_test():
    pools = _pools()
    EA, EB, EC = (1940, 2000, 1, 0), (1940, 4000, 1, 0), (1940, 6000, 1, 0)

    # 0b's own dispositiveness on event A alone (a lone pool-mismatch episode),
    # tested in isolation via step_0b_pool_membership.
    ev_a = _make_synthetic_event(pools, seed=1940, step=2000, hop=1, batch_idx=0, bad_rows=(0,))
    ev_b_clean = _make_synthetic_event(pools, seed=1940, step=4000, hop=1, batch_idx=0)
    ev_c_clean = _make_synthetic_event(pools, seed=1940, step=6000, hop=1, batch_idx=0)
    b0 = A.step_0b_pool_membership([ev_a, ev_b_clean, ev_c_clean], pools, k=8)
    ok_0b = b0["dispositive"] and len(b0["violations"]) == 1 and b0["violations"][0]["event"] == EA
    _report("item 10a: 0b's own single-violation rule fires on event A alone, independent of "
            "any Step-1 marker elsewhere in the sink", ok_0b, str(b0))

    # Step 1's own floor does NOT clear on a SINGLE disagreement in a DIFFERENT event
    # (event B) -- tested in isolation via _apply_two_level_floor (never summed with 0b's
    # own episode count under either marker's own floor).
    step1_single_disagreement = [{"event": EB, "row_idx": 0}]
    floor_b = A._apply_two_level_floor(step1_single_disagreement)
    ok_floor = (not floor_b["floor_met"]) and floor_b["excluded_episodes"] == [EB + (0,)]
    _report("item 10b: Step 1's own single disagreement (event B) alone does NOT clear Step "
            "1's own floor -- excluded, disclosed, non-dispositive, NEVER summed with 0b's "
            "episode toward a shared count", ok_floor, str(floor_b))

    # Net verdict via the real orchestrator: attributed to 0b ALONE.
    result = A.run_full_precedence([_make_launch(pools, 1940, [ev_a, ev_b_clean, ev_c_clean])],
                                    _TOKENIZER, k=8)
    ok_net = result.get("verdict") == A.INSTRUMENT_BUG and result.get("verdict_source") == "step_0b"
    _report("item 10c: net verdict via run_full_precedence is INSTRUMENT-BUG, attributed to 0b "
            "alone (event A), on a sink where event B's own key_ids are clean (no 0b violation "
            "there) -- disclosed note: this build's own run_full_precedence short-circuits at "
            "0b's dispositive return and does not ALSO compute/attach Step 1's own diagnostic "
            "to the SAME result dict (see this script's module docstring / the build report's "
            "own disclosed deviation); items 10a/10b above independently prove the underlying "
            "0b-vs-Step-1 per-marker-type counting machinery is correct and would compose "
            "correctly if a future revision chooses to disclose both markers on one run.",
            ok_net, f"verdict={result.get('verdict')!r} source={result.get('verdict_source')!r}")


# ===========================================================================
# Item 11 -- Pool-reconstruction cross-check, the two-seeds-trap fixture.
# RUNS (real GPT-2 tokenizer, no CUDA/fla needed).
# ===========================================================================

def item_11_pool_reconstruction_two_seeds_trap():
    pools = _pools()

    # Positive fixture: the correctly-seeded (seed=0) reconstruction matches.
    chk_pos = A.check_reconstruction_gate(pools, pools.train_name_ids)
    _report("item 11a (positive): seed=0 reconstruction SET-EQUAL to its own archived "
            "anchor_train_ids -- cross-check PASSES", chk_pos["passed"], str(chk_pos))

    # Negative fixture, the trap itself: reconstruct with seed=1940 (the LAUNCH seed) and
    # cross-check against the SAME real (seed=0) anchor_train_ids tensor.
    trap_pools, _ = grd.build_entity_pools(_TOKENIZER, heldout_frac=0.5, seed=1940)
    chk_neg = A.check_reconstruction_gate(trap_pools, pools.train_name_ids)
    _report("item 11b (negative, the trap): seed=1940 reconstruction is NOT set-equal to the "
            "seed=0 archived anchor_train_ids -- cross-check FAILS as required",
            not chk_neg["passed"], str(chk_neg))

    # Full-precedence teeth: the trap launch HARD-ABORTS before Step 0b (or any other step).
    ev = _make_synthetic_event(pools, seed=1940, step=2000, hop=1, batch_idx=0)
    trap_launch = _make_launch(pools, 1940, [ev], anchor_train_ids=pools.train_name_ids)
    # Simulate the trap at the ORCHESTRATOR level by handing it a launch whose OWN archived
    # anchor_train_ids is the WRONGLY (seed=1940)-reconstructed set -- exactly the failure
    # mode a real, honestly-corrupted checkpoint would present.
    trap_launch_bad = _make_launch(pools, 1940, [ev], anchor_train_ids=trap_pools.train_name_ids)
    result = A.run_full_precedence([trap_launch_bad], _TOKENIZER, k=8)
    ok_abort = (result.get("verdict") == A.RECONSTRUCTION_FAILURE
                and "step_0b" not in result and "step_neg1" not in result)
    _report("item 11c: run_full_precedence HARD-ABORTS with RECONSTRUCTION-FAILURE before Step "
            "0b or any other step runs, no verdict emitted", ok_abort,
            f"verdict={result.get('verdict')!r}")

    # And the CORRECT launch (archived_anchor_train_ids from the pinned seed=0 reconstruction,
    # matching real production behavior) clears the gate and proceeds.
    result_ok = A.run_full_precedence([trap_launch], _TOKENIZER, k=8)
    ok_proceeds = result_ok.get("verdict") != A.RECONSTRUCTION_FAILURE
    _report("item 11d: a launch with the CORRECTLY (seed=0)-derived archived anchor_train_ids "
            "clears the reconstruction gate and proceeds to Step 0b", ok_proceeds,
            f"verdict={result_ok.get('verdict')!r}")


# ===========================================================================
# Item 12 -- Single-event 0b minimal-boundary fixture. RUNS.
# ===========================================================================

def item_12_single_event_0b_minimal_boundary():
    pools = _pools()
    ev = _make_synthetic_event(pools, seed=1940, step=2000, hop=1, batch_idx=0, bad_rows=(0,))
    launch = _make_launch(pools, 1940, [ev])
    result = A.run_full_precedence([launch], _TOKENIZER, k=8)
    ok = (result.get("verdict") == A.INSTRUMENT_BUG and result.get("verdict_source") == "step_0b"
          and "step_neg1" not in result)
    _report("item 12: len(fallback_dump_sink)==1, 1 pool-mismatch episode -> INSTRUMENT-BUG, "
            "confirmed BEFORE Step -1's own <3-event gate logic is even reached (result dict "
            "has no 'step_neg1' key -- that step never ran)", ok,
            f"verdict={result.get('verdict')!r} keys={sorted(result.keys())}")


# ===========================================================================
# Driver.
# ===========================================================================

def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("diag_c17_repro_stage_minus1.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.24.5 "
          "Stage -1 (12 registered items)")
    print("=" * 70)

    item_1_bitwise_identical_trajectory_smoke()
    item_2_dump_fires_exactly_at_fallback()
    item_3_tensor_shape_assertions()
    item_4_determinism_cross_check()
    item_5_sha256_config_cross_check()
    item_6_step_neg1_no_repro_guard()
    item_7_step1_floor_boundary()
    item_8_0b_precedes_step_neg1_enforced_abort()
    item_9_batched_vs_singleton_recompute()
    item_10_cross_marker_negative_test()
    item_11_pool_reconstruction_two_seeds_trap()
    item_12_single_event_0b_minimal_boundary()

    print("=" * 70)
    if FAILURES:
        print(f"STAGE -1: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
    else:
        print("STAGE -1: every RUN item PASSED")
    print(f"STAGE -1: {len(BOX_DEFERRED)} item(s) BOX-DEFERRED (genuinely need CUDA/fla or a "
          f"real archived artifact off this dev machine): {[d['item'] for d in BOX_DEFERRED]}")
    print("=" * 70)

    payload = {
        "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.24.5 (Stage -1, C17 repro instrument)",
        "gate_passed_locally": not FAILURES,
        "n_failures": len(FAILURES), "failures": FAILURES,
        "n_box_deferred": len(BOX_DEFERRED), "box_deferred": BOX_DEFERRED,
    }
    out_path = os.path.join(RESULTS_DIR, "diag_c17_repro_stage_minus1_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")

    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
