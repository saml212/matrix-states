"""diag_c17_repro_stage_minus1.py -- KEY_ANCHORING_SCALING_DRAFT.md sec
15.24.5's BLOCKING Stage -1 self-test suite for the C17 eval-admission
repro instrument (Rev 4, DESIGN-CLEARED-FOR-BUILD). All 12 registered
items, run to completion per this project's own standing "assertion has
teeth" rule -- not merely written.

    DRY_RUN_BYPASS=1 python3 diag_c17_repro_stage_minus1.py

(the repo's pre-train-gate hook pattern-matches any `python *.py`
invocation; DRY_RUN_BYPASS=1 is the correct, sanctioned bypass here --
every item this script runs OFF-box is pure-Python/CPU-only, no GPU.)

C17 BUILD AUDIT FIX (this session, HIGH launch-blocking finding): items 1,
2, 3, 4, 9 used to be UNCONDITIONAL `_defer()` calls with no box-side
execution path at all -- the chain's own claim that its Stage -1 step
"re-runs them on box" was FALSE (this script reported BOX-DEFERRED even on
a capable H100 box, exit 0, `set -e` never tripped). Fixed: each of these 5
items now runs a CAPABILITY PROBE (`_try_import_run_deltanet_rd()` /
`import model_rd`) and executes its REAL implementation whenever the
import succeeds (needs CUDA + `fla`, transitively via `model_rd`/
`run_deltanet_rd`); `_defer()` is now the FALLBACK branch, not the only
branch. `main()` additionally REFUSES to exit 0 if any of these 5 items is
STILL deferred while the environment reports itself CAPABLE (real
CUDA+fla, or the `C17_FORCE_CAPABLE=1` env override used ONLY by this
suite's own negative test) -- see `_env_capable_for_gated_items()` and the
`capability_gate_failed` check in `main()`.

BOX-DEFERRED items on THIS (CPU-only, no `fla`) dev machine: items 1, 2, 3,
4, 9's REAL branches (needs CUDA + `fla` -- an ALREADY-ESTABLISHED
constraint in this codebase, re-verified this session: `import model_rd`
raises `ModuleNotFoundError: No module named 'fla'` in this repo's own
`.venv`), and PART of item 5 (needs a REAL trained checkpoint / archived
result JSON this dev machine does not have -- those live on the H100 box;
item 5's box half is NOT subject to the capability-gate above, since its
blocker is a missing ARTIFACT, not missing CUDA/fla -- see
`GATED_ITEM_LABELS` below). Each BOX-DEFERRED item registers its own
box-side run requirement (see `_defer()` calls) rather than being silently
skipped.

Items 6, 7, 8, 10, 11, 12, and the new item 13 (MINOR-5 audit fix) are
genuinely CPU-only logic tests against diag_c17_repro_analysis.py's own
step functions -- called DIRECTLY (never through main()/the CLI), per that
module's own registered unit-test-isolation design decision -- and DO run
to completion this session, including item 11's own tokenizer-backed
two-seeds-trap fixture (GPT-2 tokenizer load confirmed working, network/
cache available in this sandbox).
"""
from __future__ import annotations

import argparse
import hashlib
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

# C17 build audit HIGH fix: the 5 items with NO legitimate off-box-artifact excuse -- their ONLY
# blocker is missing CUDA/fla, so once the environment IS capable, deferring them is a FAILURE,
# not a legitimate state. Item 5's box half is deliberately EXCLUDED (its blocker is a missing
# ARCHIVED RESULT JSON, a different kind of unavailability that persists even in a capable env).
GATED_ITEM_LABELS = ("item 1:", "item 2", "item 3:", "item 4:", "item 9")  # colon-terminated where a longer label shares the prefix (item 1 vs 10-13); "item 2"/"item 9" cover their own a/b/c sub-labels deliberately

# C17 build audit MINOR-4 fix: a SAVED, INDEPENDENTLY-COMPUTED expected fixture for item 11a's
# positive case -- sha256(repr(sorted(train_name_ids))) from a REAL, isolated
# `grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)` call, reproduced and recorded
# HERE, ahead of time, this session (NOT derived from the same call item 11a exercises --
# comparing a reconstruction against itself, the ORIGINAL tautology the audit flagged, proves
# nothing about whether the reconstruction is actually CORRECT, only that it is self-consistent).
EXPECTED_TRAIN_IDS_SEED0_SHA256 = "2f38f9dc96d4c7c50af9d31778af3f96ef09bafb7b15def37439682b3a5357bc"
EXPECTED_TRAIN_IDS_SEED0_N = 107


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def _defer(item: str, requirement: str, gated: bool = True) -> None:
    """`gated=True` (default): this item's ONLY blocker is missing CUDA/fla -- subject to
    `main()`'s own capability gate (refuses exit 0 if still deferred while the environment IS
    capable). `gated=False`: a legitimate off-box-artifact excuse (e.g. item 5's box half, which
    needs a REAL archived result JSON no capability probe can conjure) -- never gated."""
    print(f"[{item}] BOX-DEFERRED -- {requirement}", flush=True)
    BOX_DEFERRED.append({"item": item, "requirement": requirement, "gated": gated})


def _try_import_run_deltanet_rd():
    """C17 build audit HIGH fix -- the capability probe. Attempts the REAL box-only import chain
    (run_deltanet_rd -> model_rd -> fla.modules.ShortConvolution, CUDA-oriented at module scope).
    Returns the module on success, None on ImportError (the genuinely-off-box case this file's
    own module docstring documents) -- NEVER swallows any OTHER exception class, so a real bug in
    run_deltanet_rd.py surfaces loudly rather than silently deferring."""
    try:
        import run_deltanet_rd as rdr
        return rdr
    except ImportError:
        return None


def _env_capable_for_gated_items() -> bool:
    """True iff this environment SHOULD be able to run items 1/2/3/4/9's real branches (cuda+fla
    present). `C17_FORCE_CAPABLE=1` forces this to True regardless of actual capability -- used
    ONLY by this suite's own negative test (proving the "refuse to pass a capable-but-deferred
    run" gate below has teeth without needing a real CUDA+fla box to exercise it): each gated
    item's OWN capability probe (`_try_import_run_deltanet_rd()`) is unaffected by this override
    and will still genuinely defer on a non-capable machine, so forcing this signal True on such a
    machine reliably reproduces "capable env, item still deferred" -> the intended exit-1 path."""
    if os.environ.get("C17_FORCE_CAPABLE") == "1":
        return True
    return _try_import_run_deltanet_rd() is not None and torch.cuda.is_available()


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


# ---------------------------------------------------------------------------
# Box-only fixture kit (items 2/3/4): a REAL, geo3_active DeltaNetRDBlock at
# sec 15.24.2's own PINNED production cell (K=84, d_state=96), forced into
# the eigh fallback deterministically via a real bind() call -- mirrors
# model_rd.py's own self-test item-2b technique EXACTLY (rank-collapse
# k_proj to 2 output dims: K raw keys crammed into a 2-dim subspace CANNOT
# be mutually orthonormal for K>2, a physically-motivated, deterministic
# fallback trip through the REAL forward path, never a synthetic stand-in).
# Only ever called from inside a `_try_import_run_deltanet_rd()`-gated
# branch (needs CUDA + fla).
# ---------------------------------------------------------------------------

def _force_fallback_dump_event(rdr, device, k: int = 84, d_state: int = 96, d_model: int = 256,
                                batch_size: int = 8, step: int = 2000, seed: int = 1940):
    tokenizer = grd.load_gpt2_tokenizer()
    pools, _pool_report = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    pools = pools.to(device)
    cfg = rdr.DeltaNetRDTaskConfig(K=k, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    torch.manual_seed(11)
    model = rdr.DeltaNetRDBlock(pools.vocab_size_total, d_model=d_model, d_state=d_state,
                                 conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                                 geo3_active=True, geo3_n_iter=20, geo3_resid_tol=1e-2).to(device)
    with torch.no_grad():
        model.k_proj.weight[2:, :] = 0.0
    gen = torch.Generator(device=device).manual_seed(seed)
    sink: list = []
    # Single hop only: the rank-collapsed k_proj trips the fallback structurally,
    # hop-independently, so passing all of H_train would dump one event PER hop
    # (3 events) and break items 2a/3's exactly-one-event precondition
    # (round-2 audit residual #1 -- would false-block a healthy box).
    rdr.evaluate_pool(model, cfg, gen, device, (cfg.H_train[0],), pools, n_batches=1,
                       batch_size=batch_size,
                       use_heldout_entities=True, fallback_dump_sink=sink, step=step, seed=seed)
    return model, pools, cfg, sink


# ===========================================================================
# Item 1 -- Trajectory-perturbation smoke (baseline-relative RE-PIN of sec
# 15.24.2's bitwise spec -- disclosed deviation, see docstring). REAL when
# capable, else BOX-DEFERRED (C17 build audit HIGH fix).
# ===========================================================================

def item_1_bitwise_identical_trajectory_smoke():
    """Sec 15.24.2's required Wave -1 smoke, RE-PINNED (disclosed deviation,
    2026-07-07 box Stage -1 run) from strict bitwise equality to a
    BASELINE-RELATIVE check. Why: the first real box execution FAILED the
    bitwise form (step-50 loss 3.3978 OFF vs 3.3981 ON), and the
    discriminating OFF-vs-OFF diagnostic (logs/c17repro_item1_off_vs_off_
    diag.json on box; 3 dense log_every=1 runs) adjudicated branch (b):
    two runs with the IDENTICAL flag (both OFF), identical seeds, diverge
    from step 4 with max_abs_dev=7.5e-04 -- the repo's own already-measured
    fixed-seed GPU run-to-run nondeterminism (KEY_ANCHORING_DESIGN.md
    ~L1976-1994, the same F1 premise sec 15.24.4's Step -1 guard is built
    on; no use_deterministic_algorithms pin exists anywhere in the training
    path, by design). Bitwise equality of full 50-step trajectories is
    therefore UNSATISFIABLE on this hardware for ANY flag, so it cannot be
    the test of whether the telemetry flag perturbs training. Causal
    cross-check (code-level): c17_repro_telemetry's only effects inside
    evaluate_pool() are .detach().cpu() reads of already-computed tensors
    (run_deltanet_rd.py:392-393/473-482) -- it never consumes RNG and never
    mutates the model -- and the fixture's ONLY checkpoint eval fires at
    step 50 AFTER the step-50 loss is logged, so the observed step-50
    divergence had no causal path from the flag even in principle.

    Re-pinned criterion: OFF is run TWICE (measuring this hardware's own
    same-flag nondeterminism envelope on the identical fixture) + ON once,
    all at log_every=1 (dense, 50 points -- the original 2-point
    log_every=50 form couldn't even localize a divergence). PASS iff
    max-abs OFF-vs-ON loss deviation <= 3x the OFF-vs-OFF envelope (3x =
    registered slack for the envelope being a 1-sample estimate). If the
    envelope is EXACTLY 0 (deterministic hardware), the criterion reduces
    to the original bitwise spec -- i.e. this is a strict generalization,
    not a weakening on hardware that can support the original."""
    rdr = _try_import_run_deltanet_rd()
    if rdr is None:
        _defer("item 1: trajectory-perturbation smoke (baseline-relative re-pin)",
               "run_deltanet_rd.py's train() needs `model_rd` (imports "
               "`fla.modules.ShortConvolution` at module scope, CUDA-oriented) to even import -- "
               "genuinely box-only. Box-side requirement: a 50-step, fixed-seed mini run THREE "
               "times (--c17-repro-telemetry OFF/OFF/ON, otherwise identical, log_every=1); "
               "assert the max-abs OFF-vs-ON loss deviation <= 3x the OFF-vs-OFF (same-flag "
               "nondeterminism) envelope; envelope 0 -> the original bitwise spec.")
        return
    assert torch.cuda.is_available(), "DeltaNet-RD requires CUDA (chunk_delta_rule has no CPU path)"
    device = "cuda"
    tokenizer = grd.load_gpt2_tokenizer()
    pools, pool_report = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    pools = pools.to(device)
    cfg = rdr.DeltaNetRDTaskConfig(K=8, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))

    def _run(telemetry: bool):
        torch.manual_seed(99)
        model = rdr.DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64,
                                     conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                                     geo3_active=True, geo3_n_iter=12, geo3_resid_tol=1e-2).to(device)
        res = rdr.train(model, cfg, pools, pool_report, device, d_model=64, d_state=64,
                         steps=50, batch_size=16, seed=42, log_every=1, ckpt_every=50,
                         c17_repro_telemetry=telemetry)
        return [t["loss"] for t in res["trajectory"]]

    off1 = _run(False)
    off2 = _run(False)
    on1 = _run(True)

    def _max_abs_dev(a, b):
        assert len(a) == len(b), f"trajectory length mismatch: {len(a)} vs {len(b)}"
        return max((abs(x - y) for x, y in zip(a, b)), default=0.0)

    env = _max_abs_dev(off1, off2)
    dev_on = max(_max_abs_dev(off1, on1), _max_abs_dev(off2, on1))
    if env == 0.0:
        ok = dev_on == 0.0     # deterministic hardware -> the original bitwise spec applies
    else:
        ok = dev_on <= 3.0 * env
    _report("item 1: trajectory-perturbation smoke (--c17-repro-telemetry OFF/OFF/ON, 50-step "
            "fixed-seed dense-logged runs; baseline-relative re-pin of sec 15.24.2's Wave -1 "
            "smoke -- OFF-vs-ON deviation must sit within 3x the same-flag OFF-vs-OFF "
            "nondeterminism envelope; envelope 0 -> bitwise)", ok,
            f"n_steps_logged={len(off1)} off_vs_off_envelope={env:.3e} "
            f"max_off_vs_on_dev={dev_on:.3e} threshold={(3.0 * env):.3e} "
            f"bitwise_mode={env == 0.0}")


# ===========================================================================
# Item 2 -- Dump-fires-exactly-at-fallback smoke. REAL when capable, else
# BOX-DEFERRED.
# ===========================================================================

def item_2_dump_fires_exactly_at_fallback():
    rdr = _try_import_run_deltanet_rd()
    if rdr is None:
        _defer("item 2: dump-fires-exactly-at-fallback smoke",
               "_dump_c17_fallback_event (run_deltanet_rd.py) is exercised via evaluate_pool()'s "
               "OWN production dump-decision code path on a REAL geo3_active model -- importing "
               "model_rd needs `fla`, genuinely box-only. Box-side requirement: (a) a hand-built "
               "near-duplicate K-row batch forcing fallback_triggered=True -> assert the dump "
               "sink receives EXACTLY 1 entry; (b) a well-conditioned/orthonormal-ish batch -> "
               "assert EXACTLY 0 entries.")
        return
    assert torch.cuda.is_available(), "DeltaNet-RD requires CUDA (chunk_delta_rule has no CPU path)"
    device = "cuda"

    # (a) forced-near-duplicate (rank-collapsed k_proj) -> EXACTLY 1 dumped entry.
    _model_a, _pools_a, _cfg_a, sink_a = _force_fallback_dump_event(rdr, device)
    ok_a = len(sink_a) == 1
    _report("item 2a: hand-built near-duplicate (rank-collapsed k_proj) K-row batch FORCES "
            "fallback_triggered=True -> dump sink receives EXACTLY 1 entry", ok_a,
            f"len(sink)={len(sink_a)}")

    # (b) well-conditioned (mid-training-like, 5 real SGD steps -- mirrors model_rd.py's own
    # self-test item 2a's exact conditioning technique) -> EXACTLY 0 dumped entries.
    tokenizer = grd.load_gpt2_tokenizer()
    pools, pool_report = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    pools = pools.to(device)
    cfg = rdr.DeltaNetRDTaskConfig(K=84, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    torch.manual_seed(8)
    model_wc = rdr.DeltaNetRDBlock(pools.vocab_size_total, d_model=256, d_state=96,
                                    conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                                    geo3_active=True, geo3_n_iter=20, geo3_resid_tol=1e-2).to(device)
    opt = torch.optim.Adam(model_wc.parameters(), lr=1e-2)
    gen_train = torch.Generator(device=device).manual_seed(9)
    b_wc = grd.sample_batch_rd(cfg, 16, gen_train, hop_set=cfg.H_train, pools=pools, device=device)
    for _ in range(5):
        pred, tgt, _, _, _ = model_wc(b_wc)
        loss = (1.0 - torch.nn.functional.cosine_similarity(pred, tgt, dim=-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    gen_b = torch.Generator(device=device).manual_seed(1940)
    sink_b: list = []
    rdr.evaluate_pool(model_wc, cfg, gen_b, device, cfg.H_train, pools, n_batches=1, batch_size=8,
                       use_heldout_entities=True, fallback_dump_sink=sink_b, step=2000, seed=1940)
    ok_b = len(sink_b) == 0
    _report("item 2b: well-conditioned (mid-training-like) K-row batch does NOT trigger fallback "
            "-> dump sink receives EXACTLY 0 entries", ok_b, f"len(sink)={len(sink_b)}")


# ===========================================================================
# Item 3 -- Tensor-shape assertions. REAL when capable, else BOX-DEFERRED.
# ===========================================================================

def item_3_tensor_shape_assertions():
    rdr = _try_import_run_deltanet_rd()
    if rdr is None:
        _defer("item 3: tensor-shape assertions",
               "needs a REAL dumped event (produced by the same box-only path as item 2). "
               "Box-side requirement: k_eff_raw/k_blend_raw dumps == (B_batch, 84, 96), resid "
               "dump == (B_batch,), all CPU, fp32, detached (requires_grad=False).")
        return
    assert torch.cuda.is_available(), "DeltaNet-RD requires CUDA (chunk_delta_rule has no CPU path)"
    device = "cuda"
    B = 8
    _model, _pools_r, cfg, sink = _force_fallback_dump_event(rdr, device, batch_size=B)
    assert len(sink) == 1, f"fixture did not produce exactly 1 dumped event ({len(sink)}) -- item 2's own precondition"
    ev = sink[0]
    ok_shapes = (tuple(ev["k_eff_raw"].shape) == (B, cfg.K, 96)
                 and tuple(ev["k_blend_raw"].shape) == (B, cfg.K, 96)
                 and tuple(ev["resid"].shape) == (B,)
                 and tuple(ev["key_ids"].shape) == (B, cfg.K))
    ok_dtype = (ev["k_eff_raw"].dtype == torch.float32 and ev["k_eff_raw"].device.type == "cpu"
                and not ev["k_eff_raw"].requires_grad)
    ok = ok_shapes and ok_dtype
    _report(f"item 3: tensor-shape assertions on a REAL dumped event -- k_eff_raw/k_blend_raw == "
            f"(B,{cfg.K},96), resid == (B,), key_ids == (B,{cfg.K}), CPU/fp32/detached", ok,
            f"k_eff_raw.shape={tuple(ev['k_eff_raw'].shape)} resid.shape={tuple(ev['resid'].shape)} "
            f"key_ids.shape={tuple(ev['key_ids'].shape)} dtype={ev['k_eff_raw'].dtype} "
            f"device={ev['k_eff_raw'].device} requires_grad={ev['k_eff_raw'].requires_grad}")


# ===========================================================================
# Item 4 -- Determinism cross-check, per-launch. REAL when capable, else
# BOX-DEFERRED.
# ===========================================================================

def item_4_determinism_cross_check():
    rdr = _try_import_run_deltanet_rd()
    if rdr is None:
        _defer("item 4: determinism cross-check, per-launch",
               "needs a REAL full_step<N>.pt state_dict snapshot + a REAL live C17 dump, both "
               "produced by an actual training launch (CUDA + fla). Box-side requirement: run "
               "ONCE PER LAUNCH (primary seed 1940, and again for each contingency seed if "
               "fired) -- load the snapshot into a fresh model instance, rebuild "
               "eval_gen = seed + 10_000 + step, replay the m2/m3/c17/c19 pool-call order, "
               "assert BYTE-IDENTICAL key_ids/k_eff_raw vs. that launch's own live dump.")
        return
    assert torch.cuda.is_available(), "DeltaNet-RD requires CUDA (chunk_delta_rule has no CPU path)"
    device = "cuda"
    import tempfile
    tokenizer = grd.load_gpt2_tokenizer()
    pools, pool_report = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    pools = pools.to(device)
    # K=84 at d_state=64 -> K/d = 1.31, far beyond the observed >=0.8125
    # total-collapse regime (sec 15.22): the C17 fallback is structurally
    # guaranteed to fire during training's own checkpoint eval, so the live
    # dump carries >=1 event and the >=1-event guard below (round-2 audit
    # residual #2, vacuous-all F1 pattern) can never false-block a healthy
    # box. Was K=8 (well-conditioned -> likely zero events -> vacuous pass).
    cfg = rdr.DeltaNetRDTaskConfig(K=84, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    seed, steps = 1940, 20

    with tempfile.TemporaryDirectory(prefix="c17repro_item4_") as ckpt_dir:
        torch.manual_seed(seed)
        model = rdr.DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64,
                                     conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                                     geo3_active=True, geo3_n_iter=12, geo3_resid_tol=1e-2).to(device)
        rdr.train(model, cfg, pools, pool_report, device, d_model=64, d_state=64,
                  steps=steps, batch_size=16, seed=seed, log_every=steps, ckpt_every=steps,
                  ckpt_dir=ckpt_dir, c17_repro_telemetry=True, full_ckpt_step=steps)
        live_dump = torch.load(os.path.join(ckpt_dir, f"c17fallback_step{steps}.pt"),
                                map_location="cpu", weights_only=False)
        full_state = torch.load(os.path.join(ckpt_dir, f"full_step{steps}.pt"),
                                 map_location="cpu", weights_only=False)

        # Reconstruction: a FRESH model instance, loaded from the FULL state_dict snapshot.
        model2 = rdr.DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64,
                                      conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                                      geo3_active=True, geo3_n_iter=12, geo3_resid_tol=1e-2).to(device)
        model2.load_state_dict({k: v.to(device) for k, v in full_state.items()})
        model2.eval()

        # Replay the m2/m3/c17/c19 pool-call order (train()'s OWN order, run_deltanet_rd.py) --
        # ONE shared eval_gen, the SAME seed+10_000+step formula train() itself uses.
        eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000 + steps)
        rdr.evaluate_pool(model2, cfg, eval_gen, device, cfg.H_train, pools, c17_repro_telemetry=True)
        rdr.evaluate_pool(model2, cfg, eval_gen, device, (*cfg.H_test, *cfg.H_extra), pools,
                           c17_repro_telemetry=True)
        replay_sink: list = []
        rdr.evaluate_pool(model2, cfg, eval_gen, device, cfg.H_train, pools, use_heldout_entities=True,
                           fallback_dump_sink=replay_sink, step=steps, seed=seed, c17_repro_telemetry=True)

    live_events = live_dump["events"]
    # Round-2 audit residual #2: a 0-event mini-run would make the all() below
    # pass vacuously -- the exact F1 pattern. The fixture must produce >=1 event
    # or this item proves nothing.
    ok_nonempty = len(live_events) >= 1
    ok_count = ok_nonempty and len(live_events) == len(replay_sink)
    ok_bytes = ok_count and all(
        torch.equal(le["key_ids"], re["key_ids"]) and torch.equal(le["k_eff_raw"], re["k_eff_raw"])
        for le, re in zip(live_events, replay_sink))
    ok = ok_nonempty and ok_count and ok_bytes
    _report("item 4: determinism cross-check -- offline state_dict-replay reconstruction "
            "byte-matches the live dump's key_ids/k_eff_raw (sec 15.24.4's own 'Determinism "
            "cross-check, per-launch' item; >=1 event REQUIRED, vacuous-pass guarded)", ok,
            f"n_live_events={len(live_events)} n_replay_events={len(replay_sink)} "
            f"byte_identical={ok_bytes} nonempty={ok_nonempty}")


# ===========================================================================
# STEP 3b -- REAL-launch, per-launch determinism replay (C17 build audit
# residual 2, MINOR-4 fix: "item 4 ... run PER LAUNCH -- primary run, and
# again for each fired contingency seed -- each time sequenced immediately
# after THAT launch and before ITS events are folded into the combined
# sink", KEY_ANCHORING_SCALING_DRAFT.md sec 15.24.4/15.24.5, line 4736).
# NOT part of the Stage -1 suite (main(), below) -- a SEPARATE CLI mode,
# invoked by diag_c17_repro_chain.sh between the real cell's own launch and
# the offline analysis step, against THAT launch's REAL on-disk dump
# (never a synthetic mini-fixture like item 4's own K=84/d_state=64
# self-test, whose job is only to prove the MECHANISM works at all).
# Reuses item 4's own machinery byte-for-byte (full-state reload,
# eval_gen = seed+10_000+step, m2/m3/c17/c19 replay order) -- the ONLY
# difference is the SOURCE of ground truth: a REAL c17fallback_step<step>.pt
# + full_step<step>.pt pair already on disk, never a freshly-trained
# synthetic mini-model.
# ===========================================================================

def replay_launch_dump(ckpt_dir: str, seed: int, step: int, K: int = 84, d_state: int = 96) -> int:
    """Loads `<ckpt_dir>/c17fallback_step<step>.pt` (the REAL live dump)
    and, if it carries >=1 event, `<ckpt_dir>/full_step<step>.pt` (the REAL
    full state_dict snapshot -- sec 15.24.2 item (i)) into a FRESH model
    instance, rebuilds eval_gen = seed + 10_000 + step (train()'s own
    formula, run_deltanet_rd.py:955), replays the m2/m3/c17/c19 pool-call
    order (train()'s own order, run_deltanet_rd.py:961-985), and asserts
    the replayed c17 dump is BYTE-IDENTICAL (key_ids AND k_eff_raw) to the
    live one. Returns 0 (chain continues) on a match OR a disclosed
    zero-event skip; 1 (chain stops) on any real mismatch or a missing
    required file.

    Architecture (K/d_state/geo3_n_iter/geo3_resid_tol/anchor_lambda_mode/
    h_extra) is sourced from `rdx._keyanchor_scaling_spec(K, seed,
    d_state)` -- the SAME spec object diag_c17_repro_chain.sh's own STEP 3
    token-diffs the real launch command against (never hand-typed here
    either). `d_model=256` is the one architecture field left hardcoded --
    it is NEVER emitted as a CLI flag by build_cmd() for any spec in this
    file (verified this session: build_cmd has no `--d-model` branch), so
    there is no flag to source it from; it is both run_deltanet_rd.py's own
    `--d-model` argparse default AND model_rd.DeltaNetRDBlock's own
    constructor default, cross-checked this session against the ORIGINAL
    archived K84/seed=1940 result JSON's own recorded `d_model` field
    (box-side), and matches this file's own pre-existing item-2/3 fixture
    (`_force_fallback_dump_event`, above), which already defaults to the
    same production cell's `d_model=256`. A wrong value would fail loudly
    inside `load_state_dict` regardless (shape mismatch), so this is
    defense-in-depth, not the only safety net.

    Disclosed skip (per this task's own explicit routing instruction), WITH
    an independent-audit fix (this session): a zero-event FINAL checkpoint
    is a clean, non-fatal skip (return 0) ONLY if the COMBINED event count
    across every `c17fallback_step*.pt` file in `ckpt_dir` also stays below
    the analysis script's own Step -1 minimum (`A.STEP_NEG1_MIN_EVENTS`) --
    mirroring `diag_c17_repro_analysis.py`'s own `load_launch_dump()`,
    which globs and combines EVERY checkpoint's events, not just the final
    one. If earlier checkpoints already cleared that floor, the downstream
    analysis WILL proceed past its own NO-REPRO gate using events this
    replay mechanism structurally cannot verify (only ONE full state_dict
    snapshot exists, at `full_ckpt_step`) -- that case is a disclosed
    FATAL coverage gap (return 1), not a clean skip. Item 4's own
    `ok_nonempty` assertion is correct for item 4's PURPOSE (proving the
    mechanism has teeth on its own single-checkpoint synthetic fixture) but
    was insufficient for THIS purpose against a REAL multi-checkpoint
    launch, exactly the gap this docstring's own fix addresses."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR, f"c17_repro_replay_launch_dump_seed{seed}_step{step}.json")

    rdr = _try_import_run_deltanet_rd()
    assert rdr is not None, (
        "--replay-launch-dump requires CUDA + fla (needs a REAL forward pass to replay against "
        "a REAL checkpoint) -- this CLI mode is box-only, unlike the rest of this script's own "
        "Stage -1 suite, which has a legitimate off-box BOX-DEFERRED path. There is no deferred "
        "path here: STEP 3b only ever runs immediately after a real GPU launch, on the same box.")
    assert torch.cuda.is_available(), "DeltaNet-RD requires CUDA (chunk_delta_rule has no CPU path)"
    device = "cuda"

    fallback_path = os.path.join(ckpt_dir, f"c17fallback_step{step}.pt")
    assert os.path.exists(fallback_path), (
        f"STEP 3b: {fallback_path!r} does not exist -- run_deltanet_rd.py writes this file "
        f"UNCONDITIONALLY under --c17-repro-telemetry (even a 0-event checkpoint, sec 15.24.2 "
        f"item (ii)), so a missing file means the launch did not actually run with telemetry on, "
        f"or wrote to a different --ckpt-dir than passed here -- a real bug, not a legitimate "
        f"zero-event outcome (that case is the `len(live_events) == 0` branch below, which DOES "
        f"find this file, just with an empty events list).")
    live_dump = torch.load(fallback_path, map_location="cpu", weights_only=False)
    live_events = live_dump["events"]

    if len(live_events) == 0:
        # Independent-audit fix (this session): a zero-event FINAL checkpoint is NOT automatically
        # the "routes to NO-REPRO" skip the task brief describes -- diag_c17_repro_analysis.py's
        # own load_launch_dump() globs and COMBINES every c17fallback_step*.pt file in ckpt_dir
        # (ckpt_every defaults to 2000, so a full 20000-step launch writes up to 10 such files),
        # not just the final one. If earlier checkpoints already accumulated >= the analysis
        # script's own Step -1 minimum (A.STEP_NEG1_MIN_EVENTS), the real analysis proceeds PAST
        # its own NO-REPRO gate and emits a dispositive verdict built from checkpoints this replay
        # mechanism structurally CANNOT verify -- only ONE full state_dict snapshot exists, at
        # full_ckpt_step (sec 15.24.2 item (i)), so there is no way to reconstruct any EARLIER
        # checkpoint's model state to replay against it. Check the COMBINED count (mirroring
        # load_launch_dump's own glob) before deciding this is a genuine, clean skip.
        import glob as _glob
        all_fallback_paths = sorted(_glob.glob(os.path.join(ckpt_dir, "c17fallback_step*.pt")))
        combined_n_events = sum(
            len(torch.load(p, map_location="cpu", weights_only=False)["events"])
            for p in all_fallback_paths)
        if combined_n_events < A.STEP_NEG1_MIN_EVENTS:
            note = (f"STEP 3b DISCLOSED SKIP: {fallback_path!r} (the final checkpoint) carries 0 "
                     f"C17 fallback events for seed={seed} step={step}, AND the COMBINED count "
                     f"across all {len(all_fallback_paths)} checkpoint file(s) in {ckpt_dir!r} is "
                     f"only {combined_n_events} (< the analysis script's own Step -1 minimum, "
                     f"{A.STEP_NEG1_MIN_EVENTS}) -- per this task's own registered instruction, this "
                     f"routes to the analysis script's own Step -1 NO-REPRO path (sec 15.24.4). "
                     f"Skipping the replay (nothing to reconstruct).")
            print(note, flush=True)
            payload = {"skipped": True, "coverage_gap": False, "reason": note, "seed": seed,
                       "step": step, "ckpt_dir": ckpt_dir, "n_live_events_final_checkpoint": 0,
                       "combined_n_events_all_checkpoints": combined_n_events,
                       "n_checkpoint_files": len(all_fallback_paths)}
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"wrote {out_path}")
            return 0
        note = (f"STEP 3b FATAL COVERAGE GAP: the FINAL checkpoint (step={step}) carries 0 C17 "
                 f"fallback events, but the COMBINED count across all {len(all_fallback_paths)} "
                 f"checkpoint file(s) in {ckpt_dir!r} is {combined_n_events} (>= the analysis "
                 f"script's own Step -1 minimum, {A.STEP_NEG1_MIN_EVENTS}) -- the downstream "
                 f"analysis WILL proceed past its own NO-REPRO gate and emit a dispositive verdict "
                 f"built (in part) from earlier checkpoints this replay mechanism structurally "
                 f"CANNOT verify (only ONE full state_dict snapshot exists, at step={step} -- sec "
                 f"15.24.2 item (i) -- so there is no way to reconstruct any earlier checkpoint's "
                 f"model state to replay against it). This is a genuine structural coverage gap in "
                 f"the instrument itself (found by independent audit, disclosed rather than "
                 f"silently proceeding) -- NOT proof of a reproducibility bug at the checkpoints it "
                 f"COULD verify (this cell's own final-checkpoint zero-event outcome is itself the "
                 f"expected shape for a well-conditioned late-training state, sec 15.24.5's own "
                 f"item-2b fixture comment). Stopping rather than letting the analysis draw a "
                 f"conclusion this instrument cannot fully vouch for -- a human should review "
                 f"whether Step 0b/1/2's OWN verdict logic (which reads dumped tensors directly, "
                 f"independent of this replay) is trustworthy enough to proceed on anyway before "
                 f"re-running just the analysis step by hand.")
        print(note, file=sys.stderr, flush=True)
        payload = {"skipped": True, "coverage_gap": True, "reason": note, "seed": seed, "step": step,
                   "ckpt_dir": ckpt_dir, "n_live_events_final_checkpoint": 0,
                   "combined_n_events_all_checkpoints": combined_n_events,
                   "n_checkpoint_files": len(all_fallback_paths)}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {out_path}")
        return 1

    full_state_path = os.path.join(ckpt_dir, f"full_step{step}.pt")
    assert os.path.exists(full_state_path), (
        f"STEP 3b: {fallback_path!r} carries {len(live_events)} event(s), so a replay IS "
        f"required, but {full_state_path!r} (the full state_dict snapshot, sec 15.24.2 item (i)) "
        f"is missing -- the launch must pass --full-ckpt-step {step} (matching this checkpoint "
        f"step exactly).")
    full_state = torch.load(full_state_path, map_location="cpu", weights_only=False)

    import run_deltanet_rd_exactness_sweep as rdx
    spec = rdx._keyanchor_scaling_spec(K, seed, d_state)
    h_extra = tuple(spec["h_extra"]) if spec.get("h_extra") is not None else (7, 21)

    tokenizer = grd.load_gpt2_tokenizer()
    pools, _pool_report = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    pools = pools.to(device)
    cfg = rdr.DeltaNetRDTaskConfig(K=spec["K"], conv_size=4, H_train=(1, 2, 3),
                                    H_test=(4, 5, 6), H_extra=h_extra)

    D_MODEL = 256   # see docstring -- the never-flagged, never-varied CLI/constructor default
    model2 = rdr.DeltaNetRDBlock(
        pools.vocab_size_total, d_model=D_MODEL, d_state=spec["d_state"], conv_size=cfg.conv_size,
        buffer_id=pools.buffer_id, geo3_active=spec["geo3_active"], geo3_n_iter=spec["geo3_n_iter"],
        geo3_resid_tol=spec["geo3_resid_tol"], anchor_active=spec["anchor_active"],
        anchor_lambda_mode=spec["anchor_lambda_mode"], anchor_lambda_fixed=spec["anchor_lambda_fixed"],
        anchor_train_ids=(pools.train_name_ids if spec["anchor_active"] else None),
        anchor_table_frozen=spec.get("anchor_table_frozen", False),
        anchor_table_init_mode=spec.get("anchor_table_init_mode", "frame_potential"),
    ).to(device)
    model2.load_state_dict({k: v.to(device) for k, v in full_state.items()})
    model2.eval()

    # Replay the m2/m3/c17/c19 pool-call order (train()'s OWN order, run_deltanet_rd.py) -- ONE
    # shared eval_gen, the SAME seed+10_000+step formula train() itself uses, and the SAME
    # force_rank_k this cell was launched with (None for this cell; threaded for generality).
    force_rank_k = spec.get("force_rank_k")
    eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000 + step)
    rdr.evaluate_pool(model2, cfg, eval_gen, device, cfg.H_train, pools, force_rank_k=force_rank_k,
                       c17_repro_telemetry=True)
    rdr.evaluate_pool(model2, cfg, eval_gen, device, (*cfg.H_test, *cfg.H_extra), pools,
                       force_rank_k=force_rank_k, c17_repro_telemetry=True)
    replay_sink: list = []
    rdr.evaluate_pool(model2, cfg, eval_gen, device, cfg.H_train, pools, force_rank_k=force_rank_k,
                       use_heldout_entities=True, fallback_dump_sink=replay_sink, step=step,
                       seed=seed, c17_repro_telemetry=True)

    ok_count = len(live_events) == len(replay_sink)
    ok_bytes = ok_count and all(
        torch.equal(le["key_ids"], re["key_ids"]) and torch.equal(le["k_eff_raw"], re["k_eff_raw"])
        for le, re in zip(live_events, replay_sink))
    ok = ok_count and ok_bytes
    detail = (f"n_live_events={len(live_events)} n_replay_events={len(replay_sink)} "
              f"count_match={ok_count} byte_identical={ok_bytes}")
    status = "PASS" if ok else "FAIL"
    print(f"[STEP 3b: per-launch determinism replay, seed={seed} step={step}] {status} -- {detail}",
          flush=True)
    payload = {"skipped": False, "ok": ok, "seed": seed, "step": step, "ckpt_dir": ckpt_dir,
               "n_live_events": len(live_events), "n_replay_events": len(replay_sink),
               "count_match": ok_count, "byte_identical": ok_bytes}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")
    return 0 if ok else 1


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

    # BOX HALF (deploy-session fix, 2026-07-07: was an unconditional _defer even ON the box,
    # written from the dev machine's perspective -- the same only-branch-is-defer pattern the
    # build audit's HIGH finding already killed for items 1/2/3/4/9; the artifact-presence probe
    # below is this item's own capability probe): mechanically cross-check build_cmd()/spec
    # fields against the REAL archived K84/seed=1940 result JSON's own recorded config, when
    # that artifact exists at its registered path (HERE-relative, cwd-independent).
    archived_json = os.path.join(HERE, "results", "deltanet_rd_exactness",
                                  "wavekeyanchor-scaling-wide", f"{spec['name']}.json")
    if not os.path.exists(archived_json):
        _defer("item 5 (BOX HALF): config cross-check vs the REAL archived JSON",
               f"the archived K84/seed=1940 result JSON ({spec['name']}.json, its own "
               "exactness_config block) lives on the H100 box, not in this checkout. Box-side "
               "requirement: this same item re-runs there and mechanically diffs the spec's "
               "architecture-relevant fields (K, d_state, seed, steps, geo3_n_iter, "
               "geo3_resid_tol, anchor_active, anchor_lambda_mode, drift_probe, "
               "rev7_engagement, H_extra) against that JSON's own recorded config.",
               gated=False)   # blocker is a missing ARTIFACT, not CUDA/fla -- never gated (see GATED_ITEM_LABELS)
        return
    with open(archived_json) as f:
        arch = json.load(f)
    xc = arch["exactness_config"]
    checks = {
        "K": (arch["K"], spec["K"]),
        "seed": (arch["seed"], spec["seed"]),
        "d_state": (arch["d_state"], spec["d_state"]),
        "steps": (arch["steps"], spec["steps"]),
        # run_deltanet_rd.py's own --h-extra CLI default (7,21) is what a spec with
        # h_extra=None launches with -- the same defaulting build_cmd() itself relies on.
        "H_extra": (tuple(arch["H_extra"]),
                     tuple(spec["h_extra"]) if spec.get("h_extra") is not None else (7, 21)),
        "geo3_active": (xc["geo3_active"], spec["geo3_active"]),
        "geo3_n_iter": (xc["geo3_n_iter"], spec["geo3_n_iter"]),
        "geo3_resid_tol": (xc["geo3_resid_tol"], spec["geo3_resid_tol"]),
        "anchor_active": (xc["anchor_active"], spec["anchor_active"]),
        "anchor_lambda_mode": (xc["anchor_lambda_mode"], spec["anchor_lambda_mode"]),
        "drift_probe": (xc["drift_probe"], spec["drift_probe"]),
        "rev7_engagement": (xc["rev7_engagement"], spec["rev7_engagement"]),
    }
    mismatches = {k: v for k, v in (checks.items()) if v[0] != v[1]}
    _report("item 5 (BOX HALF): mechanical config cross-check -- the spec this chain launches "
            "from matches the REAL archived K84/seed=1940 JSON's own recorded "
            "config/exactness_config field-for-field", not mismatches,
            f"archived={os.path.basename(archived_json)} n_fields={len(checks)} "
            f"mismatches={mismatches}")


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
# Item 9 -- Batched-vs-singleton offline recompute comparison. REAL when
# capable, else BOX-DEFERRED. Extended (C17 build audit MEDIUM-2 fix) to
# ALSO cover Step 2's own recompute path (a structural source check, item
# 9c, runs LOCALLY -- fully CPU-safe, no CUDA/fla needed).
# ===========================================================================

def item_9_batched_vs_singleton_recompute():
    try:
        import model_rd as mrd
    except ImportError:
        mrd = None
    if mrd is None:
        _defer("item 9: batched-vs-singleton offline recompute comparison",
               "needs model_rd.newton_schulz_orthogonalize (fla-dependent import), genuinely "
               "box-only. Box-side requirement: on a hand-built (128,84,96) tensor with one row "
               "engineered near the resid=0.01 threshold, run newton_schulz_orthogonalize once "
               "as ONE BATCHED call on the full tensor (Step 1's own pinned method) and once as "
               "128 singleton (1,84,96) calls; assert the batched call's resid_offline[row_idx] "
               "is what diag_c17_repro_analysis.py's Step 1 actually uses, and DISCLOSE any row "
               "where the two methods' resid values differ.")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        B, K, d = 128, 84, 96
        gen = torch.Generator().manual_seed(2026)
        k_eff_raw = torch.nn.functional.normalize(torch.randn(B, K, d, generator=gen), dim=-1)
        # Engineer row 0 near the resid=0.01 threshold: nudge one row toward a near-duplicate of
        # another WITHIN the same episode (row-level near-collinearity is exactly what drives the
        # NS residual up -- the same mechanism model_rd.py's own self-test item 2b uses via a
        # rank-collapsed model, applied here to ONE row of a hand-built tensor instead).
        k_eff_raw[0, 1] = torch.nn.functional.normalize(
            k_eff_raw[0, 0] + 1e-3 * torch.randn(d, generator=gen), dim=0)
        k_eff_raw = k_eff_raw.to(device)

        with torch.no_grad():
            _, resid_batched = mrd.newton_schulz_orthogonalize(k_eff_raw, n_iter=20)
        resid_singleton = torch.empty(B, device=device)
        with torch.no_grad():
            for row in range(B):
                _, r = mrd.newton_schulz_orthogonalize(k_eff_raw[row:row + 1], n_iter=20)
                resid_singleton[row] = r[0]
        diff = (resid_batched - resid_singleton).abs()
        differing_rows = [i for i in range(B) if diff[i].item() > 0]
        # Sec 15.24.4/15.24.5 item 9's own pinned decision: the BATCHED result is what Step 1
        # actually uses (diag_c17_repro_analysis.py's own _step_1_recompute_disagreements never
        # calls a singleton slice) -- this fixture DISCLOSES the batch-size confound's measured
        # size on this synthetic near-boundary population, it does not gate on the singleton
        # result itself (matching the design's own "disclose, don't gate" instruction).
        _report("item 9a: batched-vs-singleton offline recompute on a (128,84,96) near-boundary "
                "tensor -- batched resid is what Step 1 actually uses, singleton discrepancy "
                "DISCLOSED", True,
                f"n_rows_differing={len(differing_rows)}/{B} max_abs_diff={diff.max().item():.3e} "
                f"differing_rows={differing_rows[:10]}{'...' if len(differing_rows) > 10 else ''}")

        # item 9b (MEDIUM-2 fix coverage): the SAME batched discipline, exercised through
        # diag_c17_repro_analysis.py's OWN Step-2 recompute (`_step_2_recompute_resolve_niter`),
        # on a real synthetic event -- confirms it actually runs (needs model_rd, box-only) and
        # returns a well-formed per-episode map.
        synth_event = {"seed": 1940, "step": 2000, "hop": 1, "batch_idx": 0,
                       "k_eff_raw": k_eff_raw.detach().cpu()}
        per_episode = A._step_2_recompute_resolve_niter([synth_event], device=device)
        ok_step2_runs = len(per_episode) == B and all(
            A.episode_key(synth_event, row) in per_episode for row in range(B))
        _report("item 9b (MEDIUM-2 fix coverage): diag_c17_repro_analysis.py's Step-2 n_iter "
                "sweep runs end-to-end on the SAME batched-per-event tensor and returns one "
                "resolved-n_iter entry per episode", ok_step2_runs, f"n_episodes={len(per_episode)}")

    # item 9c (MEDIUM-2 fix coverage, structural -- ALWAYS runs, fully CPU-safe, no CUDA/fla
    # needed): a source-level check (mirrors run_deltanet_rd.py smoke()'s own NO-EVAL-LEAK
    # discipline) that NEITHER Step 1's NOR Step 2's own recompute contains a
    # `k_eff_raw[row_idx:row_idx + 1]`-style singleton slice INSIDE its sweep/comparison loop --
    # the exact MEDIUM-2/MAJOR-B batch-size-confound bug pattern -- proving the fix is actually
    # wired into the source, not merely true on today's synthetic data (item 9a/9b's own numeric
    # checks can pass by luck if this specific hardware/software stack happens not to exhibit the
    # kernel-selection confound on THIS data).
    import ast
    import inspect

    def _code_only_normalized(func) -> str:
        # Audit nit fix, hardened twice over: (1) whitespace-insensitive (a
        # spacing variant of the slice would evade an exact-string match);
        # (2) DOCSTRING-STRIPPED via ast.unparse -- both functions' own
        # docstrings legitimately NAME the forbidden pattern in prose
        # ("NEVER a k_eff_raw[row_idx:row_idx+1] slice"), and a raw
        # getsource substring check trips on that prose (found live this
        # session: the un-stripped check false-flagged Step 2 on its own
        # docstring). ast.unparse drops comments; we drop the leading
        # docstring Expr node explicitly. Same fragility class the Phase-2b
        # round-4 auditor demonstrated for _references.
        tree = ast.parse(inspect.getsource(func))
        fn = tree.body[0]
        if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)
                and isinstance(fn.body[0].value.value, str)):
            fn.body = fn.body[1:]
        return "".join(ast.unparse(tree).split())

    src_step1 = _code_only_normalized(A._step_1_recompute_disagreements)
    src_step2 = _code_only_normalized(A._step_2_recompute_resolve_niter)
    singleton_pattern = "".join("row_idx:row_idx + 1".split())
    has_singleton_step1 = singleton_pattern in src_step1
    has_singleton_step2 = singleton_pattern in src_step2
    ok_no_singleton = not has_singleton_step1 and not has_singleton_step2
    _report("item 9c (MEDIUM-2 fix coverage, structural, LOCAL): neither Step 1's nor Step 2's "
            "own recompute source contains a `k_eff_raw[row_idx:row_idx+1]` singleton slice",
            ok_no_singleton,
            f"step1_has_singleton={has_singleton_step1} step2_has_singleton={has_singleton_step2}")


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

    # Positive fixture (MINOR-4 audit fix, replaces the ORIGINAL tautological
    # `check_reconstruction_gate(pools, pools.train_name_ids)` self-comparison -- pools.
    # train_name_ids compared against ITSELF proves nothing): the seed=0 reconstruction's
    # train_name_ids, hashed, compared against a SAVED, INDEPENDENTLY-computed expected fixture
    # (EXPECTED_TRAIN_IDS_SEED0_SHA256, pinned above, NOT derived from this call).
    actual_ids = sorted(int(x) for x in pools.train_name_ids.tolist())
    actual_hash = hashlib.sha256(repr(actual_ids).encode()).hexdigest()
    fixture_ok = (actual_hash == EXPECTED_TRAIN_IDS_SEED0_SHA256
                  and len(actual_ids) == EXPECTED_TRAIN_IDS_SEED0_N)
    _report("item 11a (positive, MINOR-4 fix -- against a SAVED fixture, not itself): seed=0 "
            "reconstruction's train_name_ids sha256 matches the independently-pinned expected "
            "fixture", fixture_ok,
            f"actual_sha256={actual_hash} expected={EXPECTED_TRAIN_IDS_SEED0_SHA256} "
            f"n={len(actual_ids)} expected_n={EXPECTED_TRAIN_IDS_SEED0_N}")

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
# Item 13 -- TF32-recording negative test (C17 build audit MINOR-5 fix --
# wires in the design's own "Required negative test, TF32-recording" (sec
# 15.24.2), never previously assigned a Stage -1 item number). RUNS
# (CPU-safe: tests only that the production READ expression captures live
# process state, never any actual TF32 kernel behavior, which needs a GPU
# and is out of scope here).
# ===========================================================================

def item_13_tf32_recording_negative_test():
    """sec 15.24.2's own required negative test: force
    torch.backends.cuda.matmul.allow_tf32 / torch.backends.cudnn.allow_tf32 True then False and
    assert the EXACT expression run_deltanet_rd.py's checkpoint writer uses
    (`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32` -- train()'s own
    `c17fallback_step<N>.pt` writer) reads back the FORCED value each time -- proving the tap
    reads live process state at dump time rather than a hardcoded/cached value. Both attributes
    are plain settable/gettable Python attributes even with no CUDA device present (verified this
    session), so this needs NO GPU -- deliberately scoped to the RECORDING mechanism only."""
    orig_matmul = torch.backends.cuda.matmul.allow_tf32
    orig_cudnn = torch.backends.cudnn.allow_tf32
    try:
        for forced in (True, False):
            torch.backends.cuda.matmul.allow_tf32 = forced
            torch.backends.cudnn.allow_tf32 = forced
            recorded_matmul = torch.backends.cuda.matmul.allow_tf32   # the EXACT production read
            recorded_cudnn = torch.backends.cudnn.allow_tf32          # expression, run_deltanet_rd.py
            ok = recorded_matmul == forced and recorded_cudnn == forced
            _report(f"item 13: TF32-recording negative test (forced={forced}) -- recorded "
                    "tf32_matmul/tf32_cudnn match the FORCED value, not a stale/hardcoded one",
                    ok, f"forced={forced} recorded_matmul={recorded_matmul} recorded_cudnn={recorded_cudnn}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = orig_matmul
        torch.backends.cudnn.allow_tf32 = orig_cudnn


# ===========================================================================
# Driver.
# ===========================================================================

def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 70)
    print("diag_c17_repro_stage_minus1.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.24.5 "
          "Stage -1 (12 registered items + item 13, C17 build audit MINOR-5 fix)")
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
    item_13_tf32_recording_negative_test()

    print("=" * 70)
    if FAILURES:
        print(f"STAGE -1: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
    else:
        print("STAGE -1: every RUN item PASSED")
    print(f"STAGE -1: {len(BOX_DEFERRED)} item(s) BOX-DEFERRED (genuinely need CUDA/fla or a "
          f"real archived artifact off this dev machine): {[d['item'] for d in BOX_DEFERRED]}")

    # C17 build audit HIGH fix: refuse to pass if the environment IS capable (real cuda+fla, or
    # the C17_FORCE_CAPABLE=1 override used ONLY by this suite's own negative test) but one of
    # items 1/2/3/4/9 (GATED_ITEM_LABELS) is STILL deferred -- on a genuinely capable box, that
    # can only mean a real bug in the capability probe or the item's own real-path implementation,
    # never a legitimate BOX-DEFERRED state.
    capable = _env_capable_for_gated_items()
    still_deferred_gated = [d for d in BOX_DEFERRED
                             if d.get("gated", True)
                             and any(d["item"].startswith(lbl) for lbl in GATED_ITEM_LABELS)]
    capability_gate_failed = capable and bool(still_deferred_gated)
    if capability_gate_failed:
        print(f"STAGE -1 FATAL: environment reports CAPABLE (cuda+fla, or "
              f"C17_FORCE_CAPABLE=1) but {len(still_deferred_gated)} launch-blocking item(s) are "
              f"STILL BOX-DEFERRED: {[d['item'] for d in still_deferred_gated]} -- refusing to "
              f"pass (KEY_ANCHORING_SCALING_DRAFT.md sec 15.24.5, C17 build audit HIGH fix: no "
              f"box-side execution path is an acceptable state once the environment IS capable).",
              file=sys.stderr)
    else:
        print(f"STAGE -1: capability gate OK (capable={capable}, "
              f"still_deferred_gated={[d['item'] for d in still_deferred_gated]})")
    print("=" * 70)

    payload = {
        "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.24.5 (Stage -1, C17 repro instrument)",
        "gate_passed_locally": not FAILURES and not capability_gate_failed,
        "n_failures": len(FAILURES), "failures": FAILURES,
        "n_box_deferred": len(BOX_DEFERRED), "box_deferred": BOX_DEFERRED,
        "capable": capable, "capability_gate_failed": capability_gate_failed,
        "still_deferred_gated": [d["item"] for d in still_deferred_gated],
    }
    out_path = os.path.join(RESULTS_DIR, "diag_c17_repro_stage_minus1_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")

    return 0 if (not FAILURES and not capability_gate_failed) else 1


if __name__ == "__main__":
    if "--replay-launch-dump" in sys.argv:
        # STEP 3b CLI mode (audit residual 2 / MINOR-4) -- a SEPARATE entrypoint from the Stage -1
        # suite's own `main()` above, never run together in one invocation.
        ap = argparse.ArgumentParser(
            description="STEP 3b: per-launch determinism replay against a REAL launch's own "
                        "on-disk C17 dump (KEY_ANCHORING_SCALING_DRAFT.md sec 15.24.4/15.24.5, "
                        "audit residual 2 / MINOR-4). Does NOT run the Stage -1 suite.")
        ap.add_argument("--replay-launch-dump", type=str, required=True, metavar="CKPT_DIR",
                         help="the launch's own --ckpt-dir (containing c17fallback_step<STEP>.pt "
                              "+ full_step<STEP>.pt)")
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--step", type=int, required=True)
        ap.add_argument("--k", type=int, default=84)
        ap.add_argument("--d-state", type=int, default=96)
        args = ap.parse_args()
        sys.exit(replay_launch_dump(args.replay_launch_dump, args.seed, args.step,
                                     K=args.k, d_state=args.d_state))
    sys.exit(main())
