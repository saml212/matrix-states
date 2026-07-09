"""h2h_round4_driver_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.31.4 item 6: the ROUND-4
RE-METRIC driver. Rev 5's two-leg gate (sec 1.31) re-meters the 9-cell task1/task2
calibration set under the REPAIRED instrument -- Leg A (episode-restricted discrete recall,
the WIN metric) + Leg B (offline-ridge mechanism attribution at the §1.30-localized tap) +
S0-necessity + the relabeled rung 2 + both-direction planted/noise-null controls -- WITHOUT
retraining: 7 of the 9 round-3 checkpoints are REUSED (provenance-checked at load time, sec
1.31.4 item 4); the 2 cells whose round-3 weights never reached disk
(`transformer_task2_calib`, `transformer_task1_stress_K48`, sec 1.31.7 F2) train fresh via the
ALREADY-FIXED `train_grammar_cell` (item 4's save-before-rung2-fit + versioned filename apply
automatically there).

This module OWNS none of the fixes themselves (items 1-5 live in `h2h_cell_train_rd.py` /
`h2h_calibration_wrappers_rd.py` / `probe_head_rd.py`) -- it is the thing that DRIVES them
against real cells, per sec 1.31.4 item 6's own text: "This is the item fix items 1-5 above
feed INTO -- without it they are unwired checks with nothing driving them."

BUILD-STAGE SCOPE: no box access, no real checkpoints. `--selftest` exercises the FULL
per-cell driver loop (provenance-checked load, Leg A, relabeled rung 2, instrument health,
Leg B ridge + its own standing sanity control, S0-necessity) end-to-end on tiny FAKE cells
(freshly-trained miniature models, temp-dir checkpoints) -- proving the wiring, never the real
round-3/round-4 numbers, which are a disclosed box-only follow-up (the real manifest's md5/
mtime values do not exist on this dev box).

Run the CPU selftest: REASONING_LINK_FORCE_CPU_STUB=1 python h2h_round4_driver_rd.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_fla_stub_rd import ensure_fla_stub                                        # noqa: E402

_STUB_INSTALLED = ensure_fla_stub()
from h2h_cell_train_rd import (load_h2h_checkpoint, get_pools,                     # noqa: E402
                               task_cfg, make_eval_episodes, train_grammar_cell,
                               eval_diagnostic_rung1_and_tell, fit_rung2_identity_classifier,
                               check_instrument_health, _leg_b_tap, _atomic_dump,
                               answer_token_ids, calibration_cells,
                               _recurrent_continuation_answer_logits, _rung1_k_restricted_pred_slot,
                               GRAMMAR_BATCH, current_dial_round)
import probe_head_rd as ph                                                          # noqa: E402
from grammar_rd import sample_batch_rd                                               # noqa: E402
from h2h_tap_localization_rd import _ridge_table, DIAG_FIT_SEED                      # noqa: E402

N_FIT_BATCHES = 24     # matches h2h_tap_localization_rd.N_FIT_BATCHES_DEFAULT (24x32 episodes)
RUNG2_SEED_OFFSET = 7_919          # matches train_grammar_cell's own rung-2 seed derivation
INSTRUMENT_HEALTH_SEED_OFFSET = 8_231
LEG_B_SANITY_SEED_OFFSET = 9_001

# ---------------------------------------------------------------------------
# sec 1.31.7 -- the 9-cell round-4 manifest: 7 reused (round-3 checkpoints on disk) + 2 fresh
# (round-3 weights for these two never reached disk, sec 1.31.7 F2: torch.save executed AFTER
# the OOM'd rung-2 fit in the pre-Rev-5.1 code, so the K48-transformer file on disk predates
# round 3 entirely; transformer_task2_calib was simply never launched).
# ---------------------------------------------------------------------------

ROUND4_CELL_SPEC = (
    {"cell_id": "contender_task1_calib", "arch": "contender", "task": "task1_calib", "K": 32,
     "role": "primary", "fresh": False},
    {"cell_id": "ablation_task1_calib", "arch": "ablation", "task": "task1_calib", "K": 32,
     "role": "primary", "fresh": False},
    {"cell_id": "transformer_task1_calib", "arch": "transformer", "task": "task1_calib", "K": 32,
     "role": "primary", "fresh": False},
    {"cell_id": "contender_task2_calib", "arch": "contender", "task": "task2_calib", "K": 32,
     "role": "primary", "fresh": False},
    {"cell_id": "ablation_task2_calib", "arch": "ablation", "task": "task2_calib", "K": 32,
     "role": "primary", "fresh": False},
    {"cell_id": "contender_task1_stress_K48", "arch": "contender", "task": "task1_calib", "K": 48,
     "role": "stress_locate_only", "fresh": False},
    {"cell_id": "ablation_task1_stress_K48", "arch": "ablation", "task": "task1_calib", "K": 48,
     "role": "stress_locate_only", "fresh": False},
    {"cell_id": "transformer_task2_calib", "arch": "transformer", "task": "task2_calib", "K": 32,
     "role": "primary", "fresh": True},
    {"cell_id": "transformer_task1_stress_K48", "arch": "transformer", "task": "task1_calib",
     "K": 48, "role": "stress_locate_only", "fresh": True},
)
assert len(ROUND4_CELL_SPEC) == 9 and sum(1 for c in ROUND4_CELL_SPEC if not c["fresh"]) == 7 \
    and sum(1 for c in ROUND4_CELL_SPEC if c["fresh"]) == 2, \
    "sec 1.31.7: round 4 must be exactly 7 reused + 2 fresh (the corrected 7+2 arithmetic, F2)"


def _round3_ckpt_filename(arch: str, task: str, K: int, role: str) -> str:
    """Resolves a reused cell's EXPECTED on-disk filename by cross-referencing
    `calibration_cells()`'s own naming convention (`h2h_calib_{arch}_{task}_{role}[_K{K}]
    [_auxrev2].pt`) -- never re-derives the convention independently (DRY: a second, drifting
    copy of that naming logic would be exactly the kind of duplication that silently goes stale).
    NOTE: round-3 checkpoints predate sec 1.31.4 item 4's `_r{round}` filename-versioning fix,
    so this is deliberately the UNSUFFIXED name -- round 5+'s own re-metric would look for the
    suffixed name instead, once item 4 has had a chance to fire on a real round boundary."""
    for cell in calibration_cells():
        if (cell["arch"] == arch and cell["task"] == task and cell["role"] == role
                and cell.get("K") == K):
            return cell["name"] + ".pt"
    raise KeyError(f"no calibration_cells() entry matches arch={arch} task={task} K={K} role={role}")


# ---------------------------------------------------------------------------
# Provenance (sec 1.31.4 item 4 / sec 1.31.7's pre-flight + loader-side pinning)
# ---------------------------------------------------------------------------

class ProvenanceError(RuntimeError):
    """A checkpoint's on-disk md5 does not match the manifest's recorded value -- refuses to
    load rather than silently proceeding on a possibly stale/corrupted/swapped file."""


def _md5_of_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance_manifest(ckpt_dir: str, cells=ROUND4_CELL_SPEC) -> dict:
    """sec 1.31.7's mandatory pre-flight gate: BEFORE any reused checkpoint is loaded, compute
    {cell_id -> {path, md5, mtime, arch, task}} for all 7 reused cells. A missing file is a
    hard error here (never silently skipped) -- round 4 re-meters EXISTING weights, it does not
    invent them."""
    manifest = {}
    for cell in cells:
        if cell["fresh"]:
            continue
        path = os.path.join(ckpt_dir, _round3_ckpt_filename(cell["arch"], cell["task"],
                                                             cell["K"], cell["role"]))
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"round-4 pre-flight (sec 1.31.7): expected REUSED checkpoint missing for cell "
                f"{cell['cell_id']!r}: {path}")
        manifest[cell["cell_id"]] = {"path": path, "md5": _md5_of_file(path),
                                     "mtime": os.path.getmtime(path),
                                     "arch": cell["arch"], "task": cell["task"]}
    return manifest


def check_manifest_against_recorded(fresh_manifest: dict, recorded: dict) -> dict:
    """Cross-checks a freshly-recomputed manifest (e.g. rebuilt just before Stage-B3 launches)
    against a PREVIOUSLY recorded one (e.g. the pre-flight manifest written to disk earlier in
    the same round) -- sec 1.31.7: 'a mismatch on ANY of the 7 blocks that cell's re-metric
    pass with a hard error, not a warning.'"""
    mismatches = [cell_id for cell_id, rec in recorded.items()
                 if fresh_manifest.get(cell_id, {}).get("md5") != rec["md5"]]
    return {"ok": len(mismatches) == 0, "mismatches": mismatches}


def load_h2h_checkpoint_with_provenance(path: str, expected_md5: str, device: str):
    """sec 1.31.4 item 4's loader-side provenance pinning: asserts the file's md5 against the
    manifest's recorded value AT LOAD TIME (not only at pre-flight), so a checkpoint swapped or
    corrupted in the window between the pre-flight scan and the actual eval cannot silently
    pass through."""
    actual_md5 = _md5_of_file(path)
    if actual_md5 != expected_md5:
        raise ProvenanceError(
            f"checkpoint provenance mismatch for {path}: manifest recorded md5={expected_md5}, "
            f"file on disk now has md5={actual_md5} -- refusing to load (sec 1.31.7's "
            f"loader-side pinning)")
    return load_h2h_checkpoint(path, device)


# ---------------------------------------------------------------------------
# Leg B extraction + its own standing sanity control (sec 1.31.4 item 2's rung-3/Leg-B half)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _leg_b_extract(arch: str, model, rig, batches: list, pools) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts (Leg-B tap, T_val target) pairs over a batch list, via the SAME `_leg_b_tap`
    dispatcher `fit_rung2_identity_classifier` uses -- one source of truth for "what is the
    Leg-B tap" across rung 2 and Leg B's own ridge fit."""
    taps, targets = [], []
    for b in batches:
        tap = _leg_b_tap(arch, model, b, pools)
        Bn, Q, D = tap.shape
        taps.append(tap.reshape(Bn * Q, D).float())
        targets.append(rig.T_val[answer_token_ids(b).reshape(-1)].float())
    return torch.cat(taps), torch.cat(targets)


def _ridge_harness_sanity_control(tap_dim: int, value_dim: int, device: str, seed: int) -> dict:
    """sec 1.31.4 item 2's rung-3/Leg-B half: promotes `h2h_tap_localization_rd.py`'s own
    known-linear-mapping recovery check from a one-off selftest to a STANDING per-run control --
    every real Leg-B ridge read in this driver is accompanied by this synthetic `Y = X @ W_true`
    (no noise) sanity fit at the SAME tap/value dims, so a silently-broken ridge harness (not
    the model) can never be mistaken for a real rf@0.9=0 finding."""
    torch.manual_seed(seed)
    N = 2000
    X = torch.randn(N, tap_dim, device=device)
    W_true = torch.randn(tap_dim, value_dim, device=device)
    Y = X @ W_true
    Xf, Yf, Xe, Ye = X[:1500], Y[:1500], X[1500:], Y[1500:]
    table = _ridge_table(Xf, Yf, Xe, Ye, device)
    passed = table["cos_mean"] > 0.99 and table["gap_vs_shuffled_cos_mean"] > 0.5
    return {**table, "passed": passed}


# ---------------------------------------------------------------------------
# S0-necessity check with the sec 1.31.2 (F4) pinned numeric HARD-STOP
# ---------------------------------------------------------------------------

@torch.no_grad()
def _s0_s1_zeroing_rung1(arch: str, model, eval_batches: list, pools) -> dict:
    """Recurrent arms only (contender/ablation -- the transformer has no fast-weight state).
    Re-derives sec 1.30's own state-zeroing localization, generalized to an arbitrary cell's
    (task, K, eval_batches), via the SAME `_recurrent_continuation_answer_logits` /
    `_rung1_k_restricted_pred_slot` rung-1 accuracy machinery every other Leg-A read in this
    codebase uses (never a second, independently-wired accuracy computation)."""
    accs = {}
    for mode in ("both_intact", "s0_zeroed", "s1_zeroed"):
        n_hit, n_tot = 0.0, 0
        for b in eval_batches:
            _, final_states = model(b["token_ids"], return_states=True)
            assert len(final_states) == 2, (
                f"{arch}: S0/S1-zeroing assumes exactly 2 layers, got {len(final_states)}")
            states = list(final_states)
            if mode == "s0_zeroed":
                states[0] = torch.zeros_like(states[0])
            elif mode == "s1_zeroed":
                states[-1] = torch.zeros_like(states[-1])
            answer_logits = _recurrent_continuation_answer_logits(arch, model, states,
                                                                   b["query_tokens"], pools.buffer_id)
            pred_slot = _rung1_k_restricted_pred_slot(answer_logits, b["entity_ids"])
            n_hit += (pred_slot == b["tgt_slot"]).float().sum().item()
            n_tot += b["tgt_slot"].numel()
        accs[mode] = {"accuracy": n_hit / max(1, n_tot), "n": n_tot}
    return accs


def s0_hard_stop_check(acc_intact: float, acc_s0_zeroed: float, acc_s1_zeroed: float,
                       n: int, K: int, leg_a_passed: bool,
                       rung_chance_mult: float = ph.RUNG_CHANCE_MULT) -> dict:
    """sec 1.31.2 (Rev 5.1 F4): the S0-HARD-STOP, pinned NUMERICALLY.
      collapse condition:  acc_A(S0-zeroed) <= rung_chance_mult x chance (the SAME 3x-chance
                            demonstration bar as Leg A itself, sec 1.31.1 -- not a separate ad
                            hoc threshold).
      unchanged condition: |acc(both-intact) - acc(S1-zeroed)| <= 2*sigma,
                            sigma = sqrt(p_hat*(1-p_hat)/n), p_hat = acc(both-intact), computed
                            PER ARM from that arm's own observed both-states-intact accuracy,
                            at n = the pinned eval set's realized query count.
    If a cell's contender PASSES Leg A but S0-zeroing does NOT collapse its accuracy, the
    fast-weight-resident mechanism claim is BLOCKED for that cell (a hard-stop, not a
    footnote) -- `hard_stop_fires` reports exactly that condition."""
    chance = 1.0 / K
    collapse_bar = rung_chance_mult * chance
    collapsed = acc_s0_zeroed <= collapse_bar
    p_hat = acc_intact
    sigma = (p_hat * (1.0 - p_hat) / max(1, n)) ** 0.5
    two_sigma = 2.0 * sigma
    delta_s1 = abs(acc_intact - acc_s1_zeroed)
    unchanged = delta_s1 <= two_sigma
    return {"acc_intact": acc_intact, "acc_s0_zeroed": acc_s0_zeroed, "acc_s1_zeroed": acc_s1_zeroed,
            "n": n, "K": K, "collapse_bar": collapse_bar, "collapsed": collapsed,
            "sigma": sigma, "two_sigma": two_sigma, "delta_s1": delta_s1, "unchanged": unchanged,
            "passed": collapsed and unchanged,
            "hard_stop_fires": bool(leg_a_passed and not collapsed)}


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------

def run_cell_round4(cell: dict, provenance_manifest: dict, device: str, out_dir: str | None,
                    ckpt_dir: str | None = None, fresh_steps_override: int | None = None) -> dict:
    """sec 1.31.4 item 6: the actual re-metric pass for ONE cell -- Leg A (acc_A, sec 1.31.1),
    the relabeled+retargeted rung 2 (item 1), instrument health (item 2, both directions), Leg
    B offline ridge INCLUDING the transformer arm (sec 1.31.2, not yet exercised on that arm by
    sec 1.29/1.30) plus its own standing sanity control (item 2's rung-3/Leg-B half), and
    S0-necessity with the pinned numeric HARD-STOP (recurrent arms only). Fresh cells train via
    the already-fixed `train_grammar_cell` (item 4's ordering/versioning apply automatically);
    reused cells load via the provenance-checked loader (item 4's loader-side pinning).
    `fresh_steps_override`: None (the default) trains the FULL pinned schedule (`FULL_STEPS *
    cell['budget_frac']`, box-only production use, real GPU time) -- ONLY the CPU selftest ever
    passes a small integer here (a tiny fixture cell), so a missing/forgotten override can never
    silently under-train a real round-4 fresh cell."""
    t0 = time.time()
    device_obj = device
    pools = get_pools(device_obj)
    arch, task, K = cell["arch"], cell["task"], cell["K"]

    if cell["fresh"]:
        assert ckpt_dir is not None, "fresh cells need ckpt_dir (train_grammar_cell's own contract)"
        train_cell = cell["cell_config"]
        train_result = train_grammar_cell(train_cell, device_obj, ckpt_dir,
                                          steps_override=fresh_steps_override)
        ckpt_path = train_result["ckpt_path"]
        provenance = {"path": ckpt_path, "md5": _md5_of_file(ckpt_path), "fresh": True,
                      "round": current_dial_round()}
    else:
        rec = provenance_manifest[cell["cell_id"]]
        ckpt_path = rec["path"]
        provenance = {**rec, "fresh": False}
    model, rig, doc = (load_h2h_checkpoint(ckpt_path, device_obj) if cell["fresh"] else
                      load_h2h_checkpoint_with_provenance(ckpt_path, provenance["md5"], device_obj))

    cfg_eval = task_cfg(task, K, n_query=None)
    hop_set = tuple(cfg_eval.H_test) if task.startswith("task2") else tuple(cfg_eval.H_train)
    eval_batches = make_eval_episodes(cfg_eval, pools, device_obj, hop_set)

    # --- Leg A: episode-restricted discrete recall, the sec 1.31.1 WIN metric ---
    leg_a = eval_diagnostic_rung1_and_tell(arch, model, rig, eval_batches, pools)

    # --- rung 2 (item 1's relabel + retarget) ---
    rung2 = fit_rung2_identity_classifier(arch, model, cfg_eval, hop_set, pools, device_obj, K,
                                          seed=cell.get("seed", 0) + RUNG2_SEED_OFFSET)

    # --- instrument health (item 2's both-directions teeth, launch-blocking for ALL arms) ---
    instrument_health = check_instrument_health(arch, model, K,
                                                seed=cell.get("seed", 0) + INSTRUMENT_HEALTH_SEED_OFFSET)

    # --- Leg B: offline ridge at the §1.30-localized tap, incl. the transformer arm ---
    gen = torch.Generator(device=device_obj)
    gen.manual_seed(DIAG_FIT_SEED)
    fit_batches = [sample_batch_rd(cfg_eval, GRAMMAR_BATCH, gen, hop_set, pools, device=device_obj)
                  for _ in range(N_FIT_BATCHES)]
    Xf, Yf = _leg_b_extract(arch, model, rig, fit_batches, pools)
    Xe, Ye = _leg_b_extract(arch, model, rig, eval_batches, pools)
    leg_b = _ridge_table(Xf.to(device_obj), Yf.to(device_obj), Xe.to(device_obj), Ye.to(device_obj),
                         device_obj)
    leg_b_sanity = _ridge_harness_sanity_control(Xf.shape[-1], Yf.shape[-1], device_obj,
                                                 seed=cell.get("seed", 0) + LEG_B_SANITY_SEED_OFFSET)

    # --- S0-necessity (recurrent arms only) ---
    s0_check = None
    if arch != "transformer":
        zeroing = _s0_s1_zeroing_rung1(arch, model, eval_batches, pools)
        s0_check = s0_hard_stop_check(
            zeroing["both_intact"]["accuracy"], zeroing["s0_zeroed"]["accuracy"],
            zeroing["s1_zeroed"]["accuracy"], zeroing["both_intact"]["n"], K, leg_a["passed"])

    result = {
        "cell_id": cell["cell_id"], "arch": arch, "task": task, "K": K, "role": cell.get("role"),
        "provenance": provenance,
        "leg_a": {"acc_A": leg_a["accuracy"], "chance": leg_a["chance"],
                 "pass_bar": leg_a["pass_bar"], "passed": leg_a["passed"],
                 "cos_pred_episode_mean": leg_a["cos_pred_episode_mean"]},
        "rung2_identity_classifier": rung2,
        "instrument_health": instrument_health,
        "leg_b_ridge": leg_b,
        "leg_b_ridge_harness_sanity_control": leg_b_sanity,
        "s0_necessity_check": s0_check,
        "rung1_positive_control_note": ("selftest 17 in h2h_cell_train_rd.py (planted-answer, "
                                        "the K-restricted gather+argmax construction) -- a "
                                        "standing code-level unit test, not re-run per-cell"),
        "wall_s": time.time() - t0,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{cell['cell_id']}_round4.json")
        _atomic_dump(out_path, result)
        result["out_path"] = out_path
    return result


def run_all_round4(ckpt_dir: str, out_dir: str, device: str,
                   cells=ROUND4_CELL_SPEC, fresh_cell_configs: dict | None = None) -> dict:
    """Drives all 9 round-4 cells: builds the pre-flight provenance manifest for the 7 reused
    cells (sec 1.31.7), then runs each cell's `run_cell_round4`. `fresh_cell_configs` supplies
    the 2 fresh cells' `train_grammar_cell`-shaped config dicts (caller-injected -- box-only
    real training is never invoked from this module's own defaults)."""
    manifest = build_provenance_manifest(ckpt_dir, cells)
    manifest_path = os.path.join(out_dir, "ROUND4_PROVENANCE_MANIFEST.json")
    os.makedirs(out_dir, exist_ok=True)
    _atomic_dump(manifest_path, manifest)

    results = {}
    for cell in cells:
        run_cell = dict(cell)
        if cell["fresh"]:
            assert fresh_cell_configs is not None and cell["cell_id"] in fresh_cell_configs, (
                f"fresh cell {cell['cell_id']!r} needs a cell_config in fresh_cell_configs")
            run_cell["cell_config"] = fresh_cell_configs[cell["cell_id"]]
        results[cell["cell_id"]] = run_cell_round4(run_cell, manifest, device, out_dir, ckpt_dir)
    return {"manifest_path": manifest_path, "cells": results}


# ---------------------------------------------------------------------------
# CPU selftest -- full driver loop on FAKE cells (tiny freshly-trained models, temp checkpoints)
# ---------------------------------------------------------------------------

def mode_selftest() -> int:
    import tempfile

    failures = []

    def rep(item, ok, detail=""):
        print(f"[{item}] {'PASS' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}", flush=True)
        if not ok:
            failures.append(item)

    device = "cpu"

    # 1. _md5_of_file determinism + change-sensitivity.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "f.bin")
        with open(p, "wb") as f:
            f.write(b"hello world")
        m1 = _md5_of_file(p)
        m2 = _md5_of_file(p)
        with open(p, "wb") as f:
            f.write(b"hello world!")   # one byte different
        m3 = _md5_of_file(p)
    ok1 = m1 == m2 and m1 != m3
    rep("selftest 1: _md5_of_file is deterministic and change-sensitive", ok1,
        f"m1={m1[:8]} m2={m2[:8]} m3={m3[:8]}")

    # 2. Provenance loader: matching md5 loads; mismatched md5 REFUSES (ProvenanceError) -- run
    #    the negative test to completion (CLAUDE.md's own hard rule).
    with tempfile.TemporaryDirectory() as td:
        tiny_cell = {"arch": "ablation", "task": "task1_calib", "role": "selftest",
                    "budget_frac": 1.0, "seed": 3, "lr": 1e-3, "K": 8,
                    "name": "provenance_fixture", "seed_idx": 0}
        import h2h_cell_train_rd as hct
        saved = (hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH,
                hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES)
        hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH = 2, 1, 4
        hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES = 5, 1   # this fixture only needs A
                                                                # checkpoint to exist, not a
                                                                # meaningful rung-2 fit -- the
                                                                # item-1 retarget's continuation-
                                                                # padded tap makes the full
                                                                # 300-step fit real CPU cost
        try:
            tr = train_grammar_cell(tiny_cell, device, td, steps_override=2)
        finally:
            (hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH,
             hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES) = saved
        real_md5 = _md5_of_file(tr["ckpt_path"])
        loaded_ok = True
        try:
            load_h2h_checkpoint_with_provenance(tr["ckpt_path"], real_md5, device)
        except ProvenanceError:
            loaded_ok = False
        refused = False
        try:
            load_h2h_checkpoint_with_provenance(tr["ckpt_path"], "0" * 32, device)
            raise RuntimeError("NEGATIVE FAILED TO FAIL: wrong md5 did not raise ProvenanceError")
        except ProvenanceError:
            refused = True
    ok2 = loaded_ok and refused
    rep("selftest 2: load_h2h_checkpoint_with_provenance loads on a matching md5 and REFUSES "
        "(ProvenanceError) on a mismatched one (negative test run to completion)", ok2,
        f"loaded_ok={loaded_ok} refused={refused}")

    # 3. build_provenance_manifest: missing file is a hard error (FileNotFoundError).
    with tempfile.TemporaryDirectory() as td:
        missing_raised = False
        try:
            build_provenance_manifest(td, cells=ROUND4_CELL_SPEC)
        except FileNotFoundError:
            missing_raised = True
    rep("selftest 3: build_provenance_manifest hard-errors when an expected reused checkpoint "
        "is missing (never silently skips a cell)", missing_raised)

    # 4. check_manifest_against_recorded: identical manifests -> ok; ANY drifted md5 -> caught.
    rec = {"a": {"md5": "aaa"}, "b": {"md5": "bbb"}}
    same = check_manifest_against_recorded({"a": {"md5": "aaa"}, "b": {"md5": "bbb"}}, rec)
    drift = check_manifest_against_recorded({"a": {"md5": "aaa"}, "b": {"md5": "DRIFTED"}}, rec)
    ok4 = same["ok"] and not drift["ok"] and drift["mismatches"] == ["b"]
    rep("selftest 4: check_manifest_against_recorded passes identical manifests, catches a "
        "single drifted cell by name", ok4, f"same={same} drift={drift}")

    # 5. s0_hard_stop_check: the pinned numeric bar, both branches (matches sec 1.31.2's own
    #    worked contender numbers: p_hat=0.9990, n=4096 -> 2sigma~=0.00099).
    K = 32
    r_collapse_ok = s0_hard_stop_check(acc_intact=0.9990, acc_s0_zeroed=0.0286, acc_s1_zeroed=0.9990,
                                       n=4096, K=K, leg_a_passed=True)
    ok5a = (r_collapse_ok["collapsed"] and r_collapse_ok["unchanged"] and r_collapse_ok["passed"]
           and not r_collapse_ok["hard_stop_fires"]
           and abs(r_collapse_ok["two_sigma"] - 0.00099) < 2e-4)
    # negative: S0-zeroing FAILS to collapse (bottleneck-leak pattern) while Leg A passed ->
    # hard_stop_fires MUST be True -- run to completion.
    r_leak = s0_hard_stop_check(acc_intact=0.9990, acc_s0_zeroed=0.50, acc_s1_zeroed=0.9990,
                                n=4096, K=K, leg_a_passed=True)
    ok5b = (not r_leak["collapsed"]) and r_leak["hard_stop_fires"]
    # a cell that never passed Leg A in the first place must NEVER hard-stop on this check alone.
    r_no_leg_a = s0_hard_stop_check(acc_intact=0.03, acc_s0_zeroed=0.50, acc_s1_zeroed=0.03,
                                    n=4096, K=K, leg_a_passed=False)
    ok5c = not r_no_leg_a["hard_stop_fires"]
    ok5 = ok5a and ok5b and ok5c
    rep("sec 1.31.2 F4 selftest 5: S0-HARD-STOP reproduces the pinned worked numbers "
        "(2sigma~=0.00099 at p_hat=0.999,n=4096); a bottleneck-leak pattern (S0-zeroing fails "
        "to collapse while Leg A passed) fires hard_stop_fires; a cell that never passed Leg A "
        "never hard-stops on this check alone", ok5,
        f"collapse_case={r_collapse_ok} leak_case_fires={r_leak['hard_stop_fires']} "
        f"no_leg_a_fires={r_no_leg_a['hard_stop_fires']}")

    # 6. FULL per-cell driver loop, end-to-end, on a tiny FRESH fake cell (real pools/tokenizer,
    #    tiny model/steps -- the "fake cells" this item's own scope calls for).
    with tempfile.TemporaryDirectory() as td_ckpt, tempfile.TemporaryDirectory() as td_out:
        import h2h_cell_train_rd as hct
        saved = (hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH,
                 hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES)
        global N_FIT_BATCHES
        saved_n_fit = N_FIT_BATCHES
        hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH = 2, 1, 4
        hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES = 5, 1
        N_FIT_BATCHES = 1
        fake_cell = {
            "cell_id": "fake_fresh_ablation", "arch": "ablation", "task": "task1_calib", "K": 8,
            "role": "primary", "fresh": True, "seed": 5,
            "cell_config": {"arch": "ablation", "task": "task1_calib", "role": "selftest",
                            "budget_frac": 1.0, "seed": 5, "lr": 1e-3, "K": 8,
                            "name": "fake_fresh_cell", "seed_idx": 0},
        }
        try:
            r = run_cell_round4(fake_cell, provenance_manifest={}, device=device, out_dir=td_out,
                               ckpt_dir=td_ckpt, fresh_steps_override=2)
        finally:
            (hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH,
             hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES) = saved
            N_FIT_BATCHES = saved_n_fit
        ok6 = (r["provenance"]["fresh"] is True and os.path.isfile(r["provenance"]["path"])
              and 0.0 <= r["leg_a"]["acc_A"] <= 1.0
              and r["rung2_identity_classifier"]["labels"] == "entity_identity"
              and r["rung2_identity_classifier"]["tap"] == "leg_b_pre_lm_head"
              and "passed" in r["instrument_health"]
              and all(k in r["leg_b_ridge"] for k in ("cos_mean", "rf@0.9", "shuffled_control_cos_mean"))
              and r["leg_b_ridge_harness_sanity_control"]["passed"] is True
              and r["s0_necessity_check"] is not None and "hard_stop_fires" in r["s0_necessity_check"]
              and os.path.isfile(r["out_path"]))
    rep("selftest 6: FULL per-cell round-4 driver loop (fresh-cell branch) runs end-to-end -- "
        "Leg A + relabeled/retargeted rung 2 + instrument health + Leg B ridge (with its own "
        "standing sanity control) + S0-necessity, JSON written to disk", ok6,
        f"acc_A={r['leg_a']['acc_A']:.4f} rung2_labels={r['rung2_identity_classifier']['labels']} "
        f"leg_b_sanity_passed={r['leg_b_ridge_harness_sanity_control']['passed']}")

    # 7. FULL per-cell driver loop on a tiny REUSED fake cell (provenance-checked load path,
    #    the SAME checkpoint selftest 6 already wrote to disk, exercising the OTHER branch).
    with tempfile.TemporaryDirectory() as td_ckpt, tempfile.TemporaryDirectory() as td_out:
        import h2h_cell_train_rd as hct
        saved = (hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH,
                 hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES)
        saved_n_fit = N_FIT_BATCHES
        hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH = 2, 1, 4
        hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES = 5, 1
        N_FIT_BATCHES = 1
        tiny_cell = {"arch": "ablation", "task": "task1_calib", "role": "selftest",
                    "budget_frac": 1.0, "seed": 6, "lr": 1e-3, "K": 8,
                    "name": "fake_reused_cell", "seed_idx": 0}
        try:
            tr = train_grammar_cell(tiny_cell, device, td_ckpt, steps_override=2)
            fake_manifest = {"fake_reused": {"path": tr["ckpt_path"],
                                            "md5": _md5_of_file(tr["ckpt_path"]),
                                            "mtime": os.path.getmtime(tr["ckpt_path"]),
                                            "arch": "ablation", "task": "task1_calib"}}
            reused_cell = {"cell_id": "fake_reused", "arch": "ablation", "task": "task1_calib",
                          "K": 8, "role": "primary", "fresh": False}
            r2 = run_cell_round4(reused_cell, fake_manifest, device=device, out_dir=td_out)
        finally:
            (hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH,
             hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES) = saved
            N_FIT_BATCHES = saved_n_fit
        ok7 = (r2["provenance"]["fresh"] is False and 0.0 <= r2["leg_a"]["acc_A"] <= 1.0
              and r2["s0_necessity_check"] is not None)
    rep("selftest 7: FULL per-cell round-4 driver loop (reused-cell branch, provenance-checked "
        "load) runs end-to-end", ok7, f"acc_A={r2['leg_a']['acc_A']:.4f}")

    # 8. transformer arm: s0_necessity_check MUST be None (no fast-weight state to zero) --
    #    proves the arch-gating has teeth, not merely "usually populated."
    with tempfile.TemporaryDirectory() as td_ckpt, tempfile.TemporaryDirectory() as td_out:
        import h2h_cell_train_rd as hct
        saved = (hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH,
                 hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES)
        saved_n_fit = N_FIT_BATCHES
        hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH = 2, 1, 4
        hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES = 5, 1
        N_FIT_BATCHES = 1
        fake_cell_tr = {
            "cell_id": "fake_fresh_transformer", "arch": "transformer", "task": "task1_calib",
            "K": 8, "role": "primary", "fresh": True, "seed": 7,
            "cell_config": {"arch": "transformer", "task": "task1_calib", "role": "selftest",
                            "budget_frac": 1.0, "seed": 7, "lr": 1e-3, "K": 8,
                            "name": "fake_fresh_transformer_cell", "seed_idx": 0},
        }
        try:
            r3 = run_cell_round4(fake_cell_tr, provenance_manifest={}, device=device, out_dir=td_out,
                                ckpt_dir=td_ckpt, fresh_steps_override=2)
        finally:
            (hct.GRAMMAR_EVAL_EVERY, hct.EVAL_EPISODE_BATCHES, hct.GRAMMAR_BATCH,
             hct.RUNG2_TRAIN_STEPS, hct.RUNG2_EVAL_BATCHES) = saved
            N_FIT_BATCHES = saved_n_fit
    ok8 = r3["s0_necessity_check"] is None and all(
        k in r3["leg_b_ridge"] for k in ("cos_mean", "rf@0.9"))
    rep("selftest 8: transformer arm's s0_necessity_check is None (no fast-weight state to "
        "zero -- arch-gating has teeth) while Leg B ridge STILL runs on the transformer's own "
        "native tap (sec 1.31.2's own requirement: Leg B now exercises this arm too)", ok8,
        f"s0_necessity_check={r3['s0_necessity_check']}")

    print("=" * 70)
    if failures:
        print(f"SELFTEST: {len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        return 1
    print("SELFTEST: ALL ITEMS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--run-all", action="store_true",
                    help="BOX-ONLY: drives all 9 round-4 cells end-to-end (real GPU time).")
    ap.add_argument("--fresh-cell-configs", type=str,
                    help="JSON {cell_id -> train_grammar_cell-shaped config} for the 2 fresh "
                         "cells (transformer_task2_calib, transformer_task1_stress_K48) -- "
                         "required with --run-all.")
    ap.add_argument("--ckpt-dir", type=str, default="/data/h2h_rung1_ckpts")
    ap.add_argument("--out-dir", type=str, default="results/h2h_rung1/round4")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.selftest:
        return mode_selftest()
    if args.build_manifest:
        manifest = build_provenance_manifest(args.ckpt_dir)
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "ROUND4_PROVENANCE_MANIFEST.json")
        _atomic_dump(out_path, manifest)
        print(f"provenance manifest written: {out_path} ({len(manifest)} reused cells)")
        return 0
    if args.run_all:
        if not args.fresh_cell_configs:
            print("REFUSE: --run-all requires --fresh-cell-configs (the 2 fresh cells' "
                  "train_grammar_cell-shaped configs -- box-only, not invented here).",
                  file=sys.stderr)
            return 2
        with open(args.fresh_cell_configs) as f:
            fresh_cfgs = json.load(f)
        report = run_all_round4(args.ckpt_dir, args.out_dir, args.device,
                                fresh_cell_configs=fresh_cfgs)
        summary_path = os.path.join(args.out_dir, "ROUND4_SUMMARY.json")
        _atomic_dump(summary_path, {cid: {"leg_a": r["leg_a"], "s0_necessity_check": r["s0_necessity_check"]}
                                    for cid, r in report["cells"].items()})
        print(f"round 4 complete: {len(report['cells'])} cells -> {args.out_dir}, "
              f"summary at {summary_path}")
        return 0
    print("BOX-ONLY: --build-manifest requires the real round-3 checkpoints on disk; --run-all "
          "requires real GPU time and the 2 fresh cells' training configs, disclosed as a "
          "box-only follow-up per this build-fix's scope. Use --selftest for the CPU-provable "
          "driver-loop proof.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
