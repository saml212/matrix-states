#!/usr/bin/env python3
"""analyze_exactness_w0.py -- DELTANET_RD_EXACTNESS_DESIGN.md (Rev 3) Wave 0:
the closed-form crosstalk reconstruction (sec 3), the value-geometry
ablation (sec 3.4), the NCE-crowding ranking-vs-threshold gap (sec 5.1),
the per-K key/value Gram-deviation band extraction (sec 3.1 -- ALSO feeds
arm (iv)'s calibration target, sec 4.5), the mechanism-(d) flat-tail
pre-check (sec 5.2), and the mechanism-(e) T_bind/chunk-boundary audit
(sec 5.3). Wave 0 is FREE -- pure numpy + stdlib on already-archived
result JSONs, no GPU/CUDA/fla, "runs on a laptop" (mirrors
analyze_eval_truncation_rd.py's own framing and reuses its load/discover
pattern, with attribution, per this repo's pod-safety convention).

BETA STATUS (sec 3.2, Rev 2 findings 1/3): every archived dump under
experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA,waveBprobe}/
predates the beta-dump build item, so every number this script produces
against THOSE dumps is under the **PROVISIONAL beta=1 idealization** --
labeled as such in every output, carrying no standalone mechanism-(g)
attribution (sec 3.2 item 1, sec 9's retraction of Rev 1's "single most
decisive free result" framing). Once arm (iii-beta)'s fresh dumps exist
(sec 4.1 -- Wave 1, not built by this script), this SAME script auto-
detects the "beta_items" field in Z_dump and switches to the measured-beta
variant for that run, per sec 3.3's "Test A splits further" / Wave 5's
"supersedes the beta=1 pass" plan -- forward-compatible by construction,
not a separate script.

TEST A vs TEST B (sec 3.3 -- read this before trusting either number):
Test A (Z_dump-internal state fit) is a DIAGNOSTIC on the beta=1
idealization only -- geometry is IDENTICAL on both sides by construction,
so a close fit is near-tautological and earns NO explanatory claim about
the frontier. Test B (predicted-vs-INDEPENDENTLY-measured frontier) is
the section's real evidentiary weight -- it compares a prediction built
from write-time geometry alone against the live training-eval pipeline's
own logged recovered_frac@0.9 numbers (a different evaluation path/batch/
bookkeeping). h=1-only on archived dumps (sec 3.3's "principled boundary"
-- succ/tgt_slot are not dumped; multi-hop needs the NEW dump fields,
auto-detected the same way beta is).

SMALL-n CAVEAT (sec 3.3, Rev 2 finding 10): each dump carries only 4 eval
examples x K items (32-128 item-scores per seed). Mitigation: aggregate
across every archived seed and report per-seed spread alongside every
Test A/B number.

Usage:
  python analyze_exactness_w0.py --smoke
      # hand-built K=2 exact-arithmetic check of the reconstruction
      # recursion (orthonormal-beta=1 special case AND a genuinely
      # non-orthonormal 2-step case verified by hand) -- run before
      # trusting anything below.
  python analyze_exactness_w0.py --dirs DIR [DIR ...] [--out-json PATH]
      # discovers every unconstrained-arm (force_rank_k=None) *.json with
      # a Z_dump under the given dirs, groups by K, runs Test A/B, the
      # b-blind/b-full ablation, the ranking-vs-threshold gap, the per-K
      # Gram-deviation band extraction, the flat-tail pre-check, and the
      # chunk-boundary audit; prints tables and (optionally) writes one
      # JSON report.

CANONICAL INPUT DIRECTORIES (FIX-3, 2026-07-03 audit -- data hygiene):
  on the H100 box (relative to this script's own directory):
      results/deltanet_rd_w0b/wave0   results/deltanet_rd_w0b/waveA
  on the local Mac (repo-relative):
      experiment-runs/2026-07-03_deltanet_rd_waves/wave0_rerun
      experiment-runs/2026-07-03_deltanet_rd_waves/waveA
  ** NEVER point --dirs at results/deltanet_rd/wave0 (box) or the local
  archive's wave0/ (as opposed to wave0_rerun/) -- both hold the STALE
  PRE-RERUN Wave-0 outputs under the SAME FILENAMES as the canonical
  rerun data (measured discrepancy: K16 s0 final key_gram_deviation
  2.726 stale vs 1.257 canonical; silently mixing them would corrupt
  every per-K band this script feeds into arm (iv)'s calibration). **
  A runtime guard enforces this error class mechanically: if two input
  files share a basename but differ in content hash, discovery ABORTS
  (see discover_unconstrained_runs).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# C16 thresholds -- COPIED (with attribution), not imported, from
# run_deltanet_rd.py's own module-level constants (pod-safety convention;
# this script must run with nothing but numpy+stdlib installed). Any change
# to the upstream constants must be re-copied here by hand -- flagged, not
# silent, exactly like rank_utils.py's own "DO NOT MODIFY... fix upstream
# first" discipline.
# ---------------------------------------------------------------------------
C16_SALVAGE_RATIO = 0.1
C16_ALIGNMENT_COS = 0.9
C16_VALUE_SALVAGE_RATIO = 0.1

# sec 5.3's structural constants (F15-LM-verified defaults this design
# inherits unchanged).
_CONV_SIZE = 4
_CHUNK_SIZE = 64
_MIN_KERNEL_T = 128


# ---------------------------------------------------------------------------
# sec 3.2: the closed-form crosstalk reconstruction.
# ---------------------------------------------------------------------------

def delta_rule_reconstruct(k_eff: np.ndarray, v_eff: np.ndarray,
                            beta: np.ndarray | None = None) -> np.ndarray:
    """sec 3.2's K-step reduction (EXACT, not an approximation -- every
    non-write position has beta=0 and leaves S unchanged, so restricting
    to the K real write events is lossless):
        S_0 = 0
        S_{j+1} = S_j @ (I - beta_j k_j k_j^T) + beta_j v_j k_j^T
                = S_j + beta_j (v_j - S_j @ k_j) k_j^T
        S_pred := S_K
    write/slot order = clause order = row order in k_eff/v_eff (VALUE
    tokens appear in ascending slot order in the token stream, sec 3.2).
    Deliberately a plain O(K) sequential loop, not a closed-form matrix
    expression -- "no dependence on any particular closed-form matrix
    expression is required for correctness" (sec 3.2); K<=32 makes this
    cheap. Identical recurrence to deltanet_core.py's delta_rule_state
    (already self-tested there), reimplemented in numpy here for a
    torch-free, laptop-runnable Wave 0 (pod-safety convention).

    k_eff, v_eff: (E,K,d) float64. beta: (E,K) float64 or None (PROVISIONAL
    beta=1 idealization, sec 3.2 item 1 -- callers must label output
    accordingly, this function does not editorialize).
    Returns S_pred: (E,d,d) float64.
    """
    E, K, d = k_eff.shape
    assert v_eff.shape == (E, K, d), (v_eff.shape, k_eff.shape)
    if beta is None:
        beta = np.ones((E, K), dtype=np.float64)
    else:
        assert beta.shape == (E, K), beta.shape
    S = np.zeros((E, d, d), dtype=np.float64)
    for j in range(K):
        kj = k_eff[:, j, :]                          # (E,d)
        vj = v_eff[:, j, :]                           # (E,d)
        bj = beta[:, j]                                # (E,)
        Skj = np.einsum("eij,ej->ei", S, kj)           # S_j @ k_j
        delta = bj[:, None] * (vj - Skj)               # (E,d)
        S = S + np.einsum("ei,ej->eij", delta, kj)
    return S


def cosine(a: np.ndarray, b: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    num = (a * b).sum(axis=axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    return num / np.clip(den, eps, None)


def cosine_matrix(pred: np.ndarray, v: np.ndarray) -> np.ndarray:
    """cos_mat[e,i,j] = cosine(pred[e,i], v[e,j]). pred,v: (E,K,d)."""
    predn = pred / np.clip(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-12, None)
    vn = v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)
    return np.einsum("eid,ejd->eij", predn, vn)


def h1_recovery(S_pred: np.ndarray, k_eff: np.ndarray, v_eff: np.ndarray) -> np.ndarray:
    """h=1 self-consistency readout (sec 3.3's principled h=1-only
    boundary on archived dumps): pred_i = S_pred @ k_eff_i, scored against
    v_eff_i, for every item i -- exact and succ-independent at h=1 (the
    same reduction analyze_eval_truncation_rd.py's own docstring derives).
    Returns cos: (E,K)."""
    pred = np.einsum("eij,ekj->eki", S_pred, k_eff)     # (E,K,d)
    return cosine(pred, v_eff, axis=-1)


def recovered_frac(cos: np.ndarray, tau: float = 0.9) -> float:
    return float((cos > tau).mean())


# ---------------------------------------------------------------------------
# Loading / discovery (pattern reused from analyze_eval_truncation_rd.py,
# with attribution; extended here to also surface `beta_items`/`succ`/
# `tgt_slot` when present, and the run's own logged checkpoints for the
# Gram-band/flat-tail instruments).
# ---------------------------------------------------------------------------

def load_run(path: str) -> dict | None:
    with open(path) as f:
        d = json.load(f)
    z = d.get("Z_dump")
    if z is None:
        return None
    out = {
        "path": path, "name": os.path.basename(path),
        "K": d["K"], "d_state": d.get("d_state"), "seed": d.get("seed"),
        "run_force_rank_k": d.get("force_rank_k"), "complete": d.get("complete"),
        "H_train": d.get("H_train"), "H_test": d.get("H_test"), "H_extra": d.get("H_extra"),
        "S_T_raw": np.array(z["S_T_raw"], dtype=np.float64),
        "k_eff": np.array(z["k_eff_items"], dtype=np.float64),
        "v_eff": np.array(z["v_eff_items"], dtype=np.float64),
        "checkpoints": d.get("checkpoints") or [],
    }
    # Forward-compatible with the build item 1/2 dump extensions (beta,
    # succ, tgt_slot) -- present on future arm (iii-beta) dumps, absent on
    # every dump archived before this build. auto-detected, never assumed.
    out["beta_items"] = np.array(z["beta_items"], dtype=np.float64) if "beta_items" in z else None
    out["succ"] = np.array(z["succ"], dtype=np.int64) if "succ" in z else None
    out["tgt_slot"] = np.array(z["tgt_slot"], dtype=np.int64) if "tgt_slot" in z else None
    out["beta_mode"] = "measured" if out["beta_items"] is not None else "idealized_beta1_PROVISIONAL"
    return out


def discover_unconstrained_runs(dirs: list[str]) -> list[dict]:
    """Every *.json with a Z_dump AND the run's own force_rank_k is None
    (see analyze_eval_truncation_rd.py::discover_unconstrained_runs for
    why forced runs are excluded from this population).

    DATA-HYGIENE GUARD (FIX-3, 2026-07-03 audit): the box carries TWO
    differently-rooted wave0 result trees with IDENTICAL FILENAMES but
    different data (results/deltanet_rd/wave0 = stale pre-rerun;
    results/deltanet_rd_w0b/wave0 = canonical -- see the module
    docstring's measured 2.726-vs-1.257 discrepancy). If two input files
    share a basename but differ in content hash, discovery ABORTS with an
    explicit error rather than silently pooling two populations of the
    same nominal run. Bit-identical duplicates (the same file reachable
    via two --dirs entries) are deduped silently -- that is a path
    artifact, not a data conflict."""
    runs = []
    seen: dict[str, tuple[str, str]] = {}     # basename -> (first path, sha256)
    for dd in dirs:
        for path in sorted(glob.glob(os.path.join(dd, "*.json"))):
            base = os.path.basename(path)
            if base == "AGGREGATE.json":
                continue
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            if base in seen:
                prev_path, prev_digest = seen[base]
                if prev_digest != digest:
                    raise SystemExit(
                        f"DATA-HYGIENE ABORT (FIX-3): two input files share the basename "
                        f"{base!r} but differ in content:\n"
                        f"  {prev_path}\n    sha256 {prev_digest[:16]}...\n"
                        f"  {path}\n    sha256 {digest[:16]}...\n"
                        f"This is exactly the stale-vs-canonical wave0 mixing hazard the "
                        f"module docstring warns about (results/deltanet_rd/wave0 is the "
                        f"STALE pre-rerun tree; results/deltanet_rd_w0b/wave0 is canonical). "
                        f"Point --dirs at ONE consistent tree and re-run.")
                continue                       # bit-identical duplicate: dedupe silently
            seen[base] = (path, digest)
            try:
                r = load_run(path)
            except Exception as e:
                print(f"  (skip {path}: {e!r})")
                continue
            if r is None or r["run_force_rank_k"] is not None:
                continue
            runs.append(r)
    return runs


def group_by_K(runs: list[dict]) -> dict[int, list[dict]]:
    by_k: dict[int, list[dict]] = {}
    for r in runs:
        by_k.setdefault(r["K"], []).append(r)
    return by_k


# ---------------------------------------------------------------------------
# sec 3.3 Test A (beta-idealization adequacy, Z_dump-internal, DIAGNOSTIC
# weight only) and Test B (a+g sufficiency vs. the independently measured
# frontier, the section's real evidentiary weight).
# ---------------------------------------------------------------------------

def test_a(run: dict) -> dict:
    """||S_pred - S_T_raw||_F / ||S_T_raw||_F under this run's beta_mode.
    Per sec 3.3: once beta_items exists this becomes a pure numerics/
    write-order smoke of the instrument itself (expect ~0); under the
    idealized beta=1 pass it ALSO carries the "how far is SGD's beta from
    1" signal, entangled with kernel-numerics -- both readings recorded,
    disentangling is Wave 5's job once measured beta exists everywhere."""
    S_pred = delta_rule_reconstruct(run["k_eff"], run["v_eff"], run["beta_items"])
    num = np.linalg.norm((S_pred - run["S_T_raw"]).reshape(S_pred.shape[0], -1), axis=-1)
    den = np.linalg.norm(run["S_T_raw"].reshape(run["S_T_raw"].shape[0], -1), axis=-1)
    resid = num / np.clip(den, 1e-12, None)
    return {"beta_mode": run["beta_mode"], "residual_per_example": resid.tolist(),
            "residual_mean": float(resid.mean()), "residual_std": float(resid.std())}


def _measured_recovered_frac_h1(run: dict) -> float | None:
    """The run's own LAST checkpoint's officially-logged h=1
    recovered_frac@0.9 (the live training-eval pipeline's number, on a
    DIFFERENT batch/bookkeeping than the 4-example dump -- Test B's
    independent-measurement side)."""
    ck = run["checkpoints"]
    if not ck:
        return None
    h1_key = str(min(run["H_train"])) if run.get("H_train") else "1"
    m2 = ck[-1].get("M2_in_distribution", {})
    entry = m2.get(h1_key) or (list(m2.values())[0] if m2 else {})
    return entry.get("recovered_frac@0.9")


def test_b(run: dict) -> dict:
    """Predicted (from write-time geometry + the delta rule's own
    uncorrected sequential-write algebra, via S_pred) vs. measured
    (the run's own logged, independently-computed) recovered_frac@0.9 at
    h=1. Residual = predicted - measured; sec 3.3's sufficiency claim
    requires this residual to NOT grow with K across the aggregate."""
    S_pred = delta_rule_reconstruct(run["k_eff"], run["v_eff"], run["beta_items"])
    cos = h1_recovery(S_pred, run["k_eff"], run["v_eff"])
    predicted = recovered_frac(cos, 0.9)
    measured = _measured_recovered_frac_h1(run)
    residual = (predicted - measured) if measured is not None else None
    return {"beta_mode": run["beta_mode"], "predicted_recovered_frac_h1": predicted,
            "measured_recovered_frac_h1": measured, "residual": residual,
            "cos_per_item_dump": cos.reshape(-1).tolist()}


# ---------------------------------------------------------------------------
# sec 3.4: the value-geometry ablation (b-blind vs b-full).
# ---------------------------------------------------------------------------

def _orthonormal_like(v_eff: np.ndarray, seed: int) -> np.ndarray:
    """Hand-built orthonormal value set of the SAME K, per-item magnitude
    matched to the measured v_eff (so only CORRELATION structure differs,
    not overall scale), no cross-item correlation. v_eff: (E,K,d)."""
    E, K, d = v_eff.shape
    assert K <= d, f"cannot build {K} orthonormal rows in R^{d}"
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(v_eff, axis=-1)          # (E,K)
    out = np.zeros_like(v_eff)
    for e in range(E):
        G = rng.standard_normal((d, K))
        Q, _ = np.linalg.qr(G)                        # (d,K) orthonormal columns
        out[e] = Q.T * norms[e][:, None]
    return out


def value_geometry_ablation(run: dict, seed: int = 0) -> dict:
    """(b-blind): measured k_eff + a HAND-BUILT orthonormal v (isolates
    "how bad would key-side leakage look if value geometry couldn't hide
    or expose it"). (b-full): measured k_eff + measured v_eff (= Test B's
    own prediction). If b-full tracks the measured frontier meaningfully
    better than b-blind, value geometry is a real, separable contributor
    (sec 3.4). Non-additivity caveat (finding 8) is Wave 5's job, not
    computed here -- this function reports the two curves side by side."""
    v_blind = _orthonormal_like(run["v_eff"], seed=seed + run["seed"] * 1000 + run["K"])
    S_blind = delta_rule_reconstruct(run["k_eff"], v_blind, run["beta_items"])
    S_full = delta_rule_reconstruct(run["k_eff"], run["v_eff"], run["beta_items"])
    cos_blind = h1_recovery(S_blind, run["k_eff"], v_blind)
    cos_full = h1_recovery(S_full, run["k_eff"], run["v_eff"])
    return {"beta_mode": run["beta_mode"],
            "b_blind_recovered_frac_h1": recovered_frac(cos_blind, 0.9),
            "b_full_recovered_frac_h1": recovered_frac(cos_full, 0.9),
            "measured_recovered_frac_h1": _measured_recovered_frac_h1(run)}


# ---------------------------------------------------------------------------
# sec 5.1: NCE-crowding ranking-vs-absolute-threshold gap. Uses S_T_raw
# (the ACTUAL trained state, not the beta-idealized reconstruction --
# ranking is a property of the trained operator, not of the idealization).
# ---------------------------------------------------------------------------

def ranking_vs_threshold_gap(run: dict, tau: float = 0.9) -> dict:
    """top-1 ranking accuracy (argmax_j cos(pred_i,v_j)==i, what L_nce's
    gradient actually optimizes) vs. the WITHIN-DUMP absolute-cosine
    recovered_frac@tau (same pred/v population, so the two are directly
    comparable per-item) -- 'gap' = ranking-correct-but-sub-tau fraction.
    The already-logged OFFICIAL recovered_frac@0.9 (larger, independent
    eval batch) is reported alongside for context, never substituted in
    (sec 14.3's no-eval-leak discipline: ranking is diagnostic-only,
    never blended into a headline recovery number)."""
    pred = np.einsum("eij,ekj->eki", run["S_T_raw"], run["k_eff"])   # (E,K,d), h=1 proxy
    cos_mat = cosine_matrix(pred, run["v_eff"])                        # (E,K,K)
    K = cos_mat.shape[-1]
    argmax_idx = cos_mat.argmax(axis=-1)                                # (E,K)
    correct_rank = (argmax_idx == np.arange(K)[None, :])
    diag = np.diagonal(cos_mat, axis1=-2, axis2=-1)                     # (E,K) cos(pred_i,v_i)
    above_tau = diag > tau
    gap = float((correct_rank & ~above_tau).mean())
    return {"K": run["K"], "top1_ranking_accuracy": float(correct_rank.mean()),
            "within_dump_recovered_frac_h1": float(above_tau.mean()),
            "ranking_correct_but_sub_tau_gap": gap,
            "official_logged_recovered_frac_h1": _measured_recovered_frac_h1(run)}


# ---------------------------------------------------------------------------
# sec 3.1: per-K key/value Gram-deviation band extraction (feeds arm
# (iv)'s calibration target, sec 4.5). Selection follows run_deltanet_rd_
# sweep.py's aggregate()/`_premise_valid_entry` convention (COPIED with
# attribution, minimal reimplementation -- pod-safety convention): the
# LAST premise-valid checkpoint carries a run's number; a run with none is
# excluded from the band and counted separately.
# ---------------------------------------------------------------------------

def _premise_valid(entry: dict) -> bool:
    align_ok = entry.get("alignment_cos_min", -1.0) >= C16_ALIGNMENT_COS
    key_ok = entry.get("salvage_ratio_mean", -1.0) >= C16_SALVAGE_RATIO
    val_ok = entry.get("value_salvage_ratio_mean", -1.0) >= C16_VALUE_SALVAGE_RATIO
    return bool(key_ok and val_ok and align_ok)


def gram_band_for_run(run: dict) -> dict | None:
    ck = run["checkpoints"]
    if not ck:
        return None
    h1_key = str(min(run["H_train"])) if run.get("H_train") else "1"
    chosen = None
    for c in reversed(ck):
        entry = (c.get("M2_in_distribution") or {}).get(h1_key) or {}
        if _premise_valid(entry):
            chosen = entry
            break
    if chosen is None:
        return None
    return {"key_gram_deviation_mean": chosen.get("key_gram_deviation_mean"),
            "value_gram_deviation_mean": chosen.get("value_gram_deviation_mean"),
            "step": chosen.get("h")}


def per_k_gram_bands(by_k: dict[int, list[dict]]) -> dict:
    report = {}
    for K, runs in sorted(by_k.items()):
        key_devs, val_devs, n_valid, n_total = [], [], 0, len(runs)
        for r in runs:
            g = gram_band_for_run(r)
            if g is None or g["key_gram_deviation_mean"] is None:
                continue
            n_valid += 1
            key_devs.append(g["key_gram_deviation_mean"])
            val_devs.append(g["value_gram_deviation_mean"])
        entry = {"n_premise_valid_seeds": n_valid, "n_total_seeds": n_total}
        if key_devs:
            entry["key_gram_deviation"] = {"min": min(key_devs), "max": max(key_devs),
                                            "mean": float(np.mean(key_devs)),
                                            "midpoint": (min(key_devs) + max(key_devs)) / 2}
            entry["value_gram_deviation"] = {"min": min(val_devs), "max": max(val_devs),
                                              "mean": float(np.mean(val_devs)),
                                              "midpoint": (min(val_devs) + max(val_devs)) / 2}
        report[K] = entry
    return report


# ---------------------------------------------------------------------------
# sec 5.2: mechanism (d) free pre-check -- flat-tail criterion on ALREADY-
# ARCHIVED checkpoint trajectories (no new runs). Pre-registered criterion
# (Rev 2, finding 9): net rise >= +0.02 in recovered_frac@0.9 at the
# flagged hop over the final 5 logged checkpoints, WITH a positive trend
# (not oscillation) -- calibrated against sec 14.2/15.2's own convention
# that "flat" tails oscillate within a +/-0.02 band with no trend.
# ---------------------------------------------------------------------------

def flat_tail_check(run: dict, pool: str, hop_key: str) -> dict | None:
    ck = run["checkpoints"]
    if len(ck) < 2:
        return None
    tail = ck[-5:] if len(ck) >= 5 else ck
    xs, ys = [], []
    for c in tail:
        entry = (c.get(pool) or {}).get(hop_key)
        if not entry or "recovered_frac@0.9" not in entry:
            continue
        xs.append(c["step"])
        ys.append(entry["recovered_frac@0.9"])
    if len(ys) < 2:
        return None
    net_rise = ys[-1] - ys[0]
    # positive trend: a simple sign-consistency check on successive deltas
    # (majority of steps move upward), not just first-vs-last -- guards
    # against a single large jump reading as "trend" when the window
    # actually oscillates.
    deltas = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    n_up = sum(1 for d in deltas if d > 0)
    positive_trend = n_up >= (len(deltas) + 1) // 2
    still_climbing = (net_rise >= 0.02) and positive_trend
    return {"K": run["K"], "seed": run["seed"], "pool": pool, "hop": hop_key,
            "steps": xs, "recovered_frac_h_tail": ys, "net_rise": net_rise,
            "positive_trend": positive_trend, "still_climbing_TRIGGERS_wave2": still_climbing}


def mechanism_d_precheck(by_k: dict[int, list[dict]]) -> dict:
    """K=24 (never checked for a flat tail) and K=16's held-out-hop
    (h=4..7) trajectories specifically (sec 5.2). K=32's h=1 flatness is
    already on record (DELTANET_REALDATA_DESIGN.md sec 15.1 item 4) and
    is not re-derived here."""
    report = {}
    for K in (16, 24):
        runs = by_k.get(K, [])
        cells = []
        if K == 24:
            for r in runs:
                res = flat_tail_check(r, "M2_in_distribution", "1")
                if res:
                    cells.append(res)
        else:  # K == 16, held-out hops
            for r in runs:
                for h in ("4", "5", "6", "7"):
                    res = flat_tail_check(r, "M3_held_out", h)
                    if res:
                        cells.append(res)
        report[K] = cells
    return report


# ---------------------------------------------------------------------------
# sec 5.3: T_bind / chunk-boundary structural audit (pure arithmetic, no
# data needed). conv_size=4 -> buf_len=3, clause_len=7, T_bind=7K.
# ---------------------------------------------------------------------------

def chunk_boundary_audit(k_values=(8, 16, 24, 32)) -> dict:
    buf_len = max(1, _CONV_SIZE - 1)
    clause_len = buf_len + 4
    report = {}
    for K in k_values:
        T_bind = K * clause_len
        T_eff = max(T_bind, _MIN_KERNEL_T)
        padded = T_eff > T_bind
        n_chunks = -(-T_eff // _CHUNK_SIZE)   # ceil div
        report[K] = {"T_bind": T_bind, "T_effective_padded": T_eff, "was_padded": padded,
                     "n_chunk_boundaries": n_chunks}
    return report


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

def _print_header(title):
    print("\n" + "=" * 100 + f"\n  {title}\n" + "=" * 100)


def run_all(dirs: list[str]) -> dict:
    runs = discover_unconstrained_runs(dirs)
    by_k = group_by_K(runs)
    print(f"loaded {len(runs)} unconstrained-arm runs with Z_dump: "
          f"{ {K: len(v) for K, v in sorted(by_k.items())} }")

    report: dict = {"dirs": dirs, "n_runs_by_K": {K: len(v) for K, v in by_k.items()},
                     "beta_modes_seen": sorted({r["beta_mode"] for r in runs})}

    _print_header("TEST A -- beta-idealization adequacy (DIAGNOSTIC weight only, sec 3.3)")
    test_a_report: dict = {}
    for K, rs in sorted(by_k.items()):
        vals = [test_a(r) for r in rs]
        means = [v["residual_mean"] for v in vals]
        test_a_report[K] = {"beta_mode": vals[0]["beta_mode"] if vals else None,
                             "residual_mean_across_seeds": float(np.mean(means)) if means else None,
                             "residual_std_across_seeds": float(np.std(means)) if means else None,
                             "n_seeds": len(vals), "per_seed": vals}
        print(f"  K={K:3d}  beta_mode={test_a_report[K]['beta_mode']:28s}  "
              f"residual mean={test_a_report[K]['residual_mean_across_seeds']:.4f}  "
              f"std={test_a_report[K]['residual_std_across_seeds']:.4f}  n_seeds={len(vals)}")
    report["test_a"] = test_a_report

    _print_header("TEST B -- a+g sufficiency vs. the independently measured frontier (sec 3.3)")
    test_b_report: dict = {}
    for K, rs in sorted(by_k.items()):
        vals = [test_b(r) for r in rs]
        resids = [v["residual"] for v in vals if v["residual"] is not None]
        test_b_report[K] = {"beta_mode": vals[0]["beta_mode"] if vals else None,
                             "predicted_mean": float(np.mean([v["predicted_recovered_frac_h1"] for v in vals])),
                             "measured_mean": (float(np.mean([v["measured_recovered_frac_h1"] for v in vals
                                                               if v["measured_recovered_frac_h1"] is not None]))
                                                if any(v["measured_recovered_frac_h1"] is not None for v in vals)
                                                else None),
                             "residual_mean": float(np.mean(resids)) if resids else None,
                             "residual_std": float(np.std(resids)) if resids else None,
                             "n_seeds": len(vals), "per_seed": vals}
        e = test_b_report[K]
        print(f"  K={K:3d}  predicted={e['predicted_mean']:.4f}  measured={e['measured_mean']}  "
              f"residual mean={e['residual_mean']}  std={e['residual_std']}  n_seeds={len(vals)}")
    residual_by_K = {K: v["residual_mean"] for K, v in test_b_report.items() if v["residual_mean"] is not None}
    if len(residual_by_K) >= 2:
        Ks_sorted = sorted(residual_by_K)
        grows = all(residual_by_K[Ks_sorted[i + 1]] is not None and
                    residual_by_K[Ks_sorted[i]] is not None and
                    abs(residual_by_K[Ks_sorted[i + 1]]) >= abs(residual_by_K[Ks_sorted[i]]) - 1e-6
                    for i in range(len(Ks_sorted) - 1))
        print(f"  residual |predicted-measured| monotone non-decreasing in K "
              f"(diagnostic read of sec 3.3's sufficiency test): {grows}")
        report["test_b_residual_grows_with_K"] = grows
    report["test_b"] = test_b_report

    _print_header("sec 3.4 -- value-geometry ablation (b-blind vs b-full)")
    ablation_report: dict = {}
    for K, rs in sorted(by_k.items()):
        vals = [value_geometry_ablation(r) for r in rs]
        ablation_report[K] = {
            "beta_mode": vals[0]["beta_mode"] if vals else None,
            "b_blind_mean": float(np.mean([v["b_blind_recovered_frac_h1"] for v in vals])),
            "b_full_mean": float(np.mean([v["b_full_recovered_frac_h1"] for v in vals])),
            "n_seeds": len(vals),
        }
        e = ablation_report[K]
        print(f"  K={K:3d}  b-blind={e['b_blind_mean']:.4f}  b-full={e['b_full_mean']:.4f}  "
              f"(b-full tracking measured better than b-blind => value geometry is a real "
              f"separable contributor, non-additivity caveat vs (c) pending sec 5.1's dependency test)")
    report["value_geometry_ablation"] = ablation_report

    _print_header("sec 5.1 -- NCE-crowding ranking-vs-absolute-threshold gap")
    gap_report: dict = {}
    for K, rs in sorted(by_k.items()):
        vals = [ranking_vs_threshold_gap(r) for r in rs]
        gap_report[K] = {"ranking_accuracy_mean": float(np.mean([v["top1_ranking_accuracy"] for v in vals])),
                          "gap_mean": float(np.mean([v["ranking_correct_but_sub_tau_gap"] for v in vals])),
                          "n_seeds": len(vals)}
        e = gap_report[K]
        print(f"  K={K:3d}  top1_ranking_acc={e['ranking_accuracy_mean']:.4f}  "
              f"ranking-correct-but-sub-0.9 gap={e['gap_mean']:.4f}  n_seeds={len(vals)}")
    report["ranking_vs_threshold_gap"] = gap_report

    _print_header("sec 3.1 -- per-K key/value Gram-deviation bands (feeds arm (iv) calibration target)")
    band_report = per_k_gram_bands(by_k)
    for K, e in sorted(band_report.items()):
        if "key_gram_deviation" in e:
            print(f"  K={K:3d}  key_gram_dev [{e['key_gram_deviation']['min']:.3f}, "
                  f"{e['key_gram_deviation']['max']:.3f}] mid={e['key_gram_deviation']['midpoint']:.3f}  "
                  f"value_gram_dev [{e['value_gram_deviation']['min']:.3f}, "
                  f"{e['value_gram_deviation']['max']:.3f}] mid={e['value_gram_deviation']['midpoint']:.3f}  "
                  f"({e['n_premise_valid_seeds']}/{e['n_total_seeds']} premise-valid seeds)")
        else:
            print(f"  K={K:3d}  NO premise-valid checkpoint found across {e['n_total_seeds']} seeds")
    report["gram_bands_per_K"] = band_report

    _print_header("sec 5.2 -- mechanism (d) flat-tail free pre-check (K=24, K=16 held-out-hop)")
    d_report = mechanism_d_precheck(by_k)
    for K, cells in d_report.items():
        n_trig = sum(1 for c in cells if c["still_climbing_TRIGGERS_wave2"])
        print(f"  K={K:3d}  {len(cells)} (run,hop) cells checked, {n_trig} trigger Wave 2's "
              f"climbing-tail criterion (net rise >= +0.02, positive trend)")
        for c in cells:
            if c["still_climbing_TRIGGERS_wave2"]:
                print(f"    TRIGGER: seed={c['seed']} pool={c['pool']} hop={c['hop']} "
                      f"net_rise={c['net_rise']:.4f}")
    report["mechanism_d_precheck"] = d_report

    _print_header("sec 5.3 -- T_bind / chunk-boundary structural audit")
    chunk_report = chunk_boundary_audit()
    for K, e in chunk_report.items():
        print(f"  K={K:3d}  T_bind={e['T_bind']:4d}  T_effective(padded)={e['T_effective_padded']:4d}  "
              f"was_padded={e['was_padded']}  n_chunk_boundaries={e['n_chunk_boundaries']}")
    report["chunk_boundary_audit"] = chunk_report

    return report


# ---------------------------------------------------------------------------
# --smoke: hand-built exact-arithmetic checks of the reconstruction
# recursion, BEFORE trusting it against real dumps (sec 3.2's own
# discipline: "directly smoke-testable against a hand-built K=2/K=3 case
# by exact arithmetic").
# ---------------------------------------------------------------------------

def _smoke() -> None:
    print("=" * 70 + "\n  analyze_exactness_w0 SMOKE\n" + "=" * 70)

    print("\n[w0 1] orthonormal keys, beta=1 -> S_pred == sum v_j k_j^T exactly "
          "(architecture-native ideal, mirrors deltanet_core.py's own [core 1])")
    rng = np.random.default_rng(0)
    E, K, d = 3, 4, 6
    G = rng.standard_normal((E, d, K))
    Q = np.stack([np.linalg.qr(G[e])[0] for e in range(E)])   # (E,d,K) orthonormal columns
    keys = np.transpose(Q, (0, 2, 1))                            # (E,K,d) orthonormal rows
    vals = rng.standard_normal((E, K, d))
    S_pred = delta_rule_reconstruct(keys, vals, beta=None)
    S_ideal = np.einsum("eki,ekj->eij", vals, keys)
    diff = np.abs(S_pred - S_ideal).max()
    assert diff < 1e-8, f"orthonormal-beta=1 exactness FAILED: max abs diff {diff:.2e}"
    print(f"  max abs diff vs analytic ideal = {diff:.2e}")

    print("\n[w0 2] HAND-COMPUTED K=2, NON-orthonormal keys, beta != 1 -- verifies the "
          "recursion formula itself (not just the orthonormal special case)")
    k0 = np.array([1.0, 0.0])
    k1 = np.array([0.6, 0.8])          # NOT orthogonal to k0 (dot = 0.6)
    v0 = np.array([2.0, -1.0])
    v1 = np.array([0.0, 3.0])
    b0, b1 = 0.5, 0.7
    # by hand: S_0 = 0
    # S_1 = S_0 + b0*(v0 - S_0@k0) k0^T = b0 * v0 (outer) k0
    S1 = b0 * np.outer(v0, k0)
    # S_2 = S_1 + b1*(v1 - S_1@k1) k1^T
    S1k1 = S1 @ k1
    S2 = S1 + b1 * np.outer((v1 - S1k1), k1)
    k_eff = np.stack([k0, k1])[None, :, :]     # (1,2,2)
    v_eff = np.stack([v0, v1])[None, :, :]
    beta = np.array([[b0, b1]])
    S_pred = delta_rule_reconstruct(k_eff, v_eff, beta)[0]
    diff2 = np.abs(S_pred - S2).max()
    assert diff2 < 1e-10, f"hand-computed K=2 non-orthonormal case FAILED: max abs diff {diff2:.2e}"
    print(f"  max abs diff vs hand-computed S_2 = {diff2:.2e} (non-orthonormal keys, beta!=1)")

    print("\n[w0 3] h=1 self-consistency readout: orthonormal-beta=1 case recovers cos~1.0 "
          "for every item (idealized recall, negative control: garbage S must NOT)")
    cos = h1_recovery(S_pred_full := delta_rule_reconstruct(keys, vals, beta=None), keys, vals)
    assert cos.min() > 0.999, f"idealized recall FAILED: min cos {cos.min():.4f} (expected ~1.0)"
    bad_cos = h1_recovery(np.zeros_like(S_pred_full), keys, vals)
    assert bad_cos.max() < 0.5, f"negative control VACUOUS: zero-state recall gives max cos {bad_cos.max():.4f}"
    print(f"  idealized recall: min cos = {cos.min():.6f} (~1.0 required); "
          f"zero-state negative control: max cos = {bad_cos.max():.4f} (must be low)")

    print("\n[w0 4] ranking_vs_threshold_gap / value_geometry_ablation / gram_band / "
          "flat_tail_check / chunk_boundary_audit run without crashing on a synthetic run dict")
    fake_run = {"K": K, "d_state": d, "seed": 0, "H_train": [1, 2, 3],
                "S_T_raw": S_pred_full, "k_eff": keys, "v_eff": vals,
                "beta_items": None, "beta_mode": "idealized_beta1_PROVISIONAL",
                "checkpoints": [{"step": 100 * i,
                                  "M2_in_distribution": {"1": {"recovered_frac@0.9": 0.5 + 0.01 * i,
                                                                "key_gram_deviation_mean": 1.5,
                                                                "value_gram_deviation_mean": 1.7,
                                                                "salvage_ratio_mean": 0.3,
                                                                "value_salvage_ratio_mean": 0.3,
                                                                "alignment_cos_min": 0.95}},
                                  "M3_held_out": {"4": {"recovered_frac@0.9": 0.4 + 0.005 * i}}}
                                 for i in range(6)]}
    g = ranking_vs_threshold_gap(fake_run)
    ab = value_geometry_ablation(fake_run)
    band = gram_band_for_run(fake_run)
    ft = flat_tail_check(fake_run, "M3_held_out", "4")
    cb = chunk_boundary_audit()
    assert 0.0 <= g["top1_ranking_accuracy"] <= 1.0
    assert band is not None and band["key_gram_deviation_mean"] == 1.5
    assert ft is not None and ft["still_climbing_TRIGGERS_wave2"] is True   # net rise 0.025 over 5 ckpts, monotone
    assert cb[8]["T_bind"] == 56 and cb[8]["T_effective_padded"] == 128 and cb[8]["was_padded"]
    assert cb[32]["T_bind"] == 224 and not cb[32]["was_padded"] and cb[32]["n_chunk_boundaries"] == 4
    print(f"  ranking gap={g['ranking_correct_but_sub_tau_gap']:.4f}  "
          f"b-blind={ab['b_blind_recovered_frac_h1']:.4f} b-full={ab['b_full_recovered_frac_h1']:.4f}  "
          f"gram_band key_dev={band['key_gram_deviation_mean']}  "
          f"flat_tail_climbing={ft['still_climbing_TRIGGERS_wave2']} (synthetic +0.025 net rise, monotone up)  "
          f"chunk_audit K=8 T_bind={cb[8]['T_bind']} padded={cb[8]['was_padded']}, "
          f"K=32 T_bind={cb[32]['T_bind']} n_chunks={cb[32]['n_chunk_boundaries']}")

    print("\nanalyze_exactness_w0 SMOKE PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dirs", nargs="+", default=None,
                     help="directories to scan for *.json result files with a Z_dump "
                          "(e.g. experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA})")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return

    if not args.dirs:
        print("ERROR: --dirs is required (or pass --smoke for the offline arithmetic check).",
              file=sys.stderr)
        sys.exit(1)

    report = run_all(args.dirs)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
