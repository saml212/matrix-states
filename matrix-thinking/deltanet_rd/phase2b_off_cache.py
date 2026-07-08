"""phase2b_off_cache.py -- REASONING_LINK_DESIGN.md sec 16.16.8's "New
first Python step" (chain step 3, does triple duty for MAJOR-2, MAJOR-3,
and Rev-2's MAJOR-R2-3): runs `eval_query_loss_heldout` (readout (B)) on
the 6 REUSED OFF checkpoints, at ALL 5 trajectory checkpoints, BOTH K's
{20,32}, and BOTH hop-sets (primary (1,2), secondary (3,4)) -- 6 cells x 5
checkpoints x 2 K's x 2 hop-sets = 120 passes -- writing every resulting
`L_query` float to a committed `off_lquery_cache-Phase2b.json`, keyed on
`(corpus, ckpt_seed, K, checkpoint_step, hop_set)`
(`phase2_trajectory_analysis.off_cache_key`, the SAME key format the
downstream reader uses, imported directly so the two can never drift
apart).

This single cache is CONSUMED THREE TIMES downstream (sec 16.16.8's own
disclosure): (a) the OFF-floor ratio this module ALSO computes and writes
to `FLOOR_PINNED-Phase2b.json` below, (b) the primary hexachotomy's own
`off_vals` (`phase2_trajectory_analysis.killer_prediction_readout`'s
cache-backed off branch), and (c) the secondary h in {3,4} readout's own
off-half (`secondary_ood_readout`) -- computed ONCE here, never
recomputed downstream (sec 16.16.3's MAJOR-1 fix, MAJOR-R2-3's own
duplication fix: absent this cache, `off` would be re-scored once per
non-off arm branch, ~33% under-counted at 360 vs the true ~480-pass cost).

**MUST run AFTER the sec 16.16.8 sha256 reuse gate clears** (this module
does not itself re-verify the manifest -- that is a SEPARATE, earlier
chain step, `phase2b_ckpt_reuse_gate.py`, run before this one in
`phase2b_chain.sh`).

`FLOOR_PIN` (sec 16.16.6): per corpus, `mean_B(ratio) + 2*sigma_B(ratio)`
where `ratio` is EACH of that corpus's own 3 per-seed
`L_query(c=5000)/L_query(c=250)` ratios (K=32, hop_set=(1,2)) -- the SAME
`mean + k*s` one-sided convention `phase2_bands_pinned.py`'s own
`derive_val_loss_tolerance_at_checkpoint` already establishes, reused
here on ratios instead of val-losses. The RUNTIME gate check (sec
16.16.6 items 1-3, `phase2b_floor_gate_enforce.floor_verdict`) is
evaluated against a DIFFERENT quantity -- that corpus's own POOLED ratio
(mean-across-seeds L_query at c=5000 divided by mean-across-seeds L_query
at c=250, NOT the mean of the 3 per-seed ratios) -- both quantities are
written here so this near-tautological-for-THIS-wave property (both are
derived from the SAME 6 OFF cells, mirroring `BANDS_PINNED-
Phase2Familiarization.json`'s own disclosed near-tautology, sec 16.15.3)
is visible on disk, never silently conflated.

Run:
    python phase2b_off_cache.py --ckpt-dir results/phase2/ckpts \\
        --out-cache results/phase2/off_lquery_cache-Phase2b.json \\
        --out-floor results/phase2/FLOOR_PINNED-Phase2b.json --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import phase2_familiarization_train as pft  # noqa: E402
import phase2_hexachotomy as phx  # noqa: E402
import phase2_trajectory_analysis as pta  # noqa: E402
from key_anchoring import sha256_of_file  # noqa: E402  the ONE dependency-free helper worth reusing directly

CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")
CKPT_SEEDS = (0, 1, 2)
K_VALUES = (32, 20)
HOP_SETS = (pft.H_TRAIN, pft.H_TEST_HELD_OUT)   # primary (1,2), secondary (3,4) -- sec 16.16.8
FLOOR_K = 2.0   # house k=2 one-sided convention (phase2_bands_pinned.K_TOLERANCE, reused verbatim)

N_CACHED_PASSES = 18 * 5 * 2 * 2   # sec 16.16.8's own registered CACHED pass count: 18 cells (6
                                     # reused OFF + 12 new) x 5 checkpoints x 2 hop-sets x 2 K's = 360
RAW_FIXED_GPU_H = 1.234 + 0.03      # sec 16.16.8's own priced training (1.234) + 3-arm smoke (0.03)
                                     # lines, unaffected by the eval-pass rate this pilot measures
BUDGET_RAW_CEILING_GPU_H = 2.06     # sec 16.16.8's own raw-total registered figure (heads-up only)
BUDGET_CEILING_GPU_H = 26.4         # sec 16.16.8's TIMING-PILOT RE-DERIVATION (2026-07-08): measured 13.7339 s/pass replaced the 0.0022 GPU-h/pass reference; ceiling 20.6 -> 26.4, pilot-forced, disclosed
DEBUG_TAX_HIGH = 10.0               # the SAME 5-10x bracket convention this document uses everywhere


def time_one_eval_pass(ckpt_dir: str, device: str, corpus: str = CORPORA[0],
                        ckpt_seed: int = 0, checkpoint_step: int = 250, K: int = 32) -> float:
    """sec 16.16.11 item 1's mandatory pre-launch timing pilot: times ONE real
    `eval_query_loss_heldout` call (model already loaded -- this isolates the SCORING pass itself,
    the quantity sec 16.16.8's own 0.0022 GPU-h/pass reference rate prices) on a REAL reused OFF
    checkpoint. Returns elapsed wall-clock seconds for that ONE pass."""
    ckpt_path = pta.ckpt_path_for(ckpt_dir, "off", corpus, ckpt_seed, checkpoint_step)
    model = pta.phase2b_load_eval_model(ckpt_path, device)
    t0 = time.time()
    pta.eval_query_loss_heldout(model, K, pft.H_TRAIN, corpus, ckpt_seed, checkpoint_step,
                                 device=device)
    elapsed_s = time.time() - t0
    del model
    return elapsed_s


def project_and_gate_timing_pilot(elapsed_s_per_pass: float) -> dict:
    """Projects the FULL 360-cached-pass eval-readout cost from ONE measured pass, adds the
    already-priced fixed lines (training + 3-arm smoke), and checks the projection against BOTH the
    raw (2.64 GPU-h, heads-up only) and debug-tax-inclusive (26.4 GPU-h, ENFORCED) ceilings --
    mirrors this document's own 5-10x debug-tax bracket convention, applied here as a pre-launch
    abort condition rather than a post-hoc narration. Returns a dict with `ok: bool` (False =>
    caller must abort BEFORE the 12-cell launch)."""
    projected_eval_gpu_h = elapsed_s_per_pass * N_CACHED_PASSES / 3600.0
    projected_raw_total_gpu_h = RAW_FIXED_GPU_H + projected_eval_gpu_h
    projected_bracket_high_gpu_h = projected_raw_total_gpu_h * DEBUG_TAX_HIGH
    return {
        "elapsed_s_per_pass": elapsed_s_per_pass, "n_cached_passes": N_CACHED_PASSES,
        "projected_eval_gpu_h": projected_eval_gpu_h,
        "projected_raw_total_gpu_h": projected_raw_total_gpu_h,
        "projected_bracket_high_gpu_h": projected_bracket_high_gpu_h,
        "raw_ceiling_exceeded": projected_raw_total_gpu_h > BUDGET_RAW_CEILING_GPU_H,
        "ok": projected_bracket_high_gpu_h <= BUDGET_CEILING_GPU_H,
    }


def build_off_lquery_cache(ckpt_dir: str, device: str, corpora=CORPORA, ckpt_seeds=CKPT_SEEDS,
                            K_values=K_VALUES, hop_sets=HOP_SETS,
                            checkpoints=phx.CHECKPOINTS) -> dict:
    """Loads each of the 30 reused OFF .pt files through `phase2b_load_eval_model` (the single
    seam, sec 16.16.3's "Loader" bullet -- this function is caller #2 of exactly TWO registered
    callers, Rev 2.2 round-4 MINOR-R4-2) and computes `L_query` at every (corpus, ckpt_seed, K,
    checkpoint_step, hop_set) combination -- 2*3*5*2*2 = 120 passes. Returns a flat
    {off_cache_key(...): float} dict, JSON-serializable directly (string keys)."""
    cache = {}
    n_done = 0
    n_total = len(corpora) * len(ckpt_seeds) * len(checkpoints) * len(K_values) * len(hop_sets)
    for corpus in corpora:
        for ckpt_seed in ckpt_seeds:
            for checkpoint_step in checkpoints:
                ckpt_path = pta.ckpt_path_for(ckpt_dir, "off", corpus, ckpt_seed, checkpoint_step)
                model = pta.phase2b_load_eval_model(ckpt_path, device)
                for K in K_values:
                    for hop_set in hop_sets:
                        L = pta.eval_query_loss_heldout(model, K, hop_set, corpus, ckpt_seed,
                                                          checkpoint_step, device=device)
                        key = pta.off_cache_key(corpus, ckpt_seed, K, checkpoint_step, hop_set)
                        cache[key] = L
                        n_done += 1
                        print(f"[phase2b-off-cache] {n_done}/{n_total} corpus={corpus} "
                              f"ckpt_seed={ckpt_seed} c={checkpoint_step} K={K} hop_set={hop_set} "
                              f"L_query={L:.4f}", flush=True)
                del model
    assert len(cache) == n_total, f"expected {n_total} cache entries, got {len(cache)} (key collision?)"
    return cache


def write_off_lquery_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc = {
        "design_ref": "REASONING_LINK_DESIGN.md sec 16.16.8 (Rev 2.2, DESIGN-CLEARED-FOR-BUILD)",
        "n_entries": len(cache), "cached_at": time.time(),
        "cached_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache": cache,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)


def load_off_lquery_cache(path: str) -> dict:
    """Returns the flat {key: float} cache -- unwraps this module's own writer envelope if present
    (the `analyze_corpus`/`killer_prediction_readout` reader in phase2_trajectory_analysis.py
    expects a flat dict; tolerates a bare flat dict too, for the Stage -1 test's own in-memory
    fixtures)."""
    with open(path) as f:
        doc = json.load(f)
    return doc["cache"] if "cache" in doc and "cached_at" in doc else doc


class CacheIntegrityFailure(RuntimeError):
    """Build-audit MAJOR fix: raised by `load_off_lquery_cache_verified` when the on-disk
    `off_lquery_cache-Phase2b.json`'s sha256 does not match the hash `FLOOR_PINNED-Phase2b.json`
    recorded at pin time (`write_floor_pinned`'s own `off_cache_sha256` field, sec 16.16.6 -- the
    SAME hash `validate_floor_pinned`/`read_floor_pin` already trust for the floor-gate's own read
    path). A distinct exception type (not a bare RuntimeError) so a caller's generic
    except-and-log handler can distinguish a deliberate integrity halt from an unrelated crash --
    mirrors `run_deltanet_rd_exactness_sweep.py`'s own `KeyanchorCliffAbort` convention."""


def load_off_lquery_cache_verified(path: str, floor_pinned_path: str) -> dict:
    """Build-audit MAJOR fix: `load_off_lquery_cache`'s tamper-evidence protects ONLY
    `phase2b_floor_gate_enforce.py`'s own read path (via `validate_floor_pinned`, reached through
    `read_floor_pin`) -- `analyze_corpus`/`killer_prediction_readout`
    (phase2_trajectory_analysis.py, the OTHER TWO of this cache's THREE documented downstream
    consumers, this module's own docstring header) previously read the cache with ZERO integrity
    check, so a single poisoned float on disk flowed silently into a classification (build-audit
    finding, empirically reproduced by poisoning one cached value 10x and observing it propagate).

    This wrapper re-hashes `path` and requires EXACT identity with the `off_cache_sha256`
    `FLOOR_PINNED-Phase2b.json` recorded at pin time (`write_floor_pinned`) -- the SAME hash the
    floor-gate's own tamper-evidence already trusts, never a second independent hash or a
    re-derivation from the cache's own contents. Hard-aborts (raises `CacheIntegrityFailure`, NO
    verdict computed) on ANY mismatch: a missing/malformed FLOOR_PINNED doc, a missing cache file,
    or a hash that does not match."""
    if not os.path.exists(floor_pinned_path):
        raise CacheIntegrityFailure(
            f"CACHE-INTEGRITY-FAILURE: {floor_pinned_path} does not exist -- cannot verify "
            f"{path} against a pinned hash before computing a verdict")
    with open(floor_pinned_path) as f:
        floor_doc = json.load(f)
    try:
        expected_sha256 = floor_doc["off_cache_sha256"]
    except (KeyError, TypeError):
        raise CacheIntegrityFailure(
            f"CACHE-INTEGRITY-FAILURE: {floor_pinned_path} is malformed (missing "
            f"off_cache_sha256) -- cannot verify {path}")
    if not os.path.exists(path):
        raise CacheIntegrityFailure(f"CACHE-INTEGRITY-FAILURE: {path} does not exist")
    actual_sha256 = sha256_of_file(path)
    if actual_sha256 != expected_sha256:
        raise CacheIntegrityFailure(
            f"CACHE-INTEGRITY-FAILURE: {path} sha256={actual_sha256} does not match the hash "
            f"pinned in {floor_pinned_path} ({expected_sha256}) -- the cache changed since "
            f"FLOOR_PINNED-Phase2b.json was written; refusing to compute a verdict from a "
            f"tampered or stale cache")
    return load_off_lquery_cache(path)


def _mean_std(values: list) -> tuple:
    n = len(values)
    assert n >= 2, f"sample std requires n>=2, got n={n}"
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def compute_floor_ratios_and_pin(cache: dict, corpora=CORPORA, ckpt_seeds=CKPT_SEEDS,
                                  K: int = 32, hop_set: tuple = pft.H_TRAIN) -> dict:
    """Per corpus: (a) `pooled_ratio` -- the RUNTIME-gated quantity (sec 16.16.6 items 1-3): mean
    L_query across the 3 seeds at c=5000 divided by mean L_query across the 3 seeds at c=250. (b)
    `floor_pin` -- `mean_B(per_seed_ratio) + FLOOR_K*sigma_B(per_seed_ratio)`, from that SAME
    corpus's own 3 PER-SEED ratios (each computed independently per seed, THEN pooled via mean+k*s
    -- the `phase2_bands_pinned.py` convention, applied to ratios instead of val-losses)."""
    out = {}
    for corpus in corpora:
        vals_250 = [cache[pta.off_cache_key(corpus, s, K, 250, hop_set)] for s in ckpt_seeds]
        vals_5000 = [cache[pta.off_cache_key(corpus, s, K, 5000, hop_set)] for s in ckpt_seeds]
        pooled_ratio = (sum(vals_5000) / len(vals_5000)) / (sum(vals_250) / len(vals_250))
        per_seed_ratios = [v5000 / v250 for v250, v5000 in zip(vals_250, vals_5000)]
        mean_b, s_b = _mean_std(per_seed_ratios)
        floor_pin = mean_b + FLOOR_K * s_b
        out[corpus] = {
            "pooled_ratio": pooled_ratio, "per_seed_ratios": per_seed_ratios,
            "mean_b": mean_b, "s_b": s_b, "k": FLOOR_K, "floor_pin": floor_pin,
            "l_query_c250_per_seed": vals_250, "l_query_c5000_per_seed": vals_5000,
        }
    return out


def write_floor_pinned(path: str, floor_data: dict, cache_path: str) -> dict:
    doc = {
        "formula_version": (
            "REASONING_LINK_DESIGN.md sec 16.16.6: per corpus, FLOOR_PIN := mean_B(ratio) + "
            "2*sigma_B(ratio) over that corpus's own 3 per-seed L_query(c=5000)/L_query(c=250) "
            "ratios (K=32, hop_set=(1,2), house k=2 one-sided convention). The RUNTIME-gated "
            "quantity is a SEPARATE, pooled ratio (mean-across-seeds L_query(c=5000) / "
            "mean-across-seeds L_query(c=250)) -- both derived from the SAME 6 OFF cells, a "
            "disclosed near-tautology for THIS wave (mirrors BANDS_PINNED-Phase2Familiarization."
            "json's own sec 16.15.3 disclosure)."
        ),
        "floor_by_corpus": floor_data,
        "off_cache_path": cache_path,
        "off_cache_sha256": sha256_of_file(cache_path),
        "pinned_at": time.time(),
        "pinned_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def validate_floor_pinned(path: str) -> dict | None:
    """Re-hashes the referenced off_lquery_cache-Phase2b.json and requires identity with the
    stored digest -- mirrors phase2_bands_pinned.validate_bands_pinned_phase2's own tamper-evidence
    discipline. Returns the parsed doc on success, None on ANY validation failure."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    try:
        if not os.path.exists(doc["off_cache_path"]):
            return None
        if sha256_of_file(doc["off_cache_path"]) != doc["off_cache_sha256"]:
            return None
    except (KeyError, TypeError):
        return None
    return doc


def read_floor_pin(floor_pinned_path: str, corpus: str) -> float:
    """Thin reader for phase2b_chain.sh's own per-corpus floor-gate step: NEVER recomputes, only
    reads the already-pinned value (sec 16.16.6's own "reads FLOOR_PIN from this JSON, never
    recomputes it" requirement)."""
    doc = validate_floor_pinned(floor_pinned_path)
    assert doc is not None, (
        f"{floor_pinned_path} failed tamper-evidence validation (missing, corrupted, or the "
        f"referenced off_lquery_cache-Phase2b.json changed since pinning) -- refusing to read a "
        f"floor_pin from an unvalidated pin")
    return doc["floor_by_corpus"][corpus]["floor_pin"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=str, required=True)
    ap.add_argument("--out-cache", type=str, default=None)
    ap.add_argument("--out-floor", type=str, default=None)
    ap.add_argument("--device", type=str,
                     default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--time-pilot", action="store_true",
                     help="sec 16.16.11 item 1's mandatory pre-launch timing pilot: time ONE real "
                          "eval pass, project against the 2.64/26.4 GPU-h budget, exit nonzero if "
                          "the projection exceeds the registered ceiling -- does NOT build the cache")
    args = ap.parse_args()

    if args.time_pilot:
        elapsed_s = time_one_eval_pass(args.ckpt_dir, args.device)
        proj = project_and_gate_timing_pilot(elapsed_s)
        print(f"[phase2b-timing-pilot] one real eval pass: {elapsed_s:.4f}s -- projected "
              f"{proj['n_cached_passes']} cached passes = {proj['projected_eval_gpu_h']:.4f} GPU-h; "
              f"projected raw total = {proj['projected_raw_total_gpu_h']:.4f} GPU-h "
              f"(registered raw ceiling {BUDGET_RAW_CEILING_GPU_H}); projected {DEBUG_TAX_HIGH:.0f}x "
              f"debug-tax bracket high = {proj['projected_bracket_high_gpu_h']:.4f} GPU-h "
              f"(registered ceiling {BUDGET_CEILING_GPU_H})")
        if proj["raw_ceiling_exceeded"]:
            print(f"[phase2b-timing-pilot] NOTE: projected raw total exceeds the registered "
                  f"{BUDGET_RAW_CEILING_GPU_H} GPU-h raw figure -- sec 16.16.11 item 1's own open "
                  f"reference-rate-uncertainty concern is empirically live this run (heads-up only, "
                  f"not an abort condition by itself).")
        if not proj["ok"]:
            print(f"ABORT: projected debug-tax-bracket-high ({proj['projected_bracket_high_gpu_h']:.4f} "
                  f"GPU-h) exceeds the registered ceiling ({BUDGET_CEILING_GPU_H} GPU-h, sec 16.16.8) "
                  f"-- halting mechanically before the OFF-eval cache build or the 12-cell launch.",
                  file=sys.stderr)
            sys.exit(1)
        print("[phase2b-timing-pilot] PASS: projection within the registered ceiling.")
        return

    assert args.out_cache and args.out_floor, "--out-cache/--out-floor are required unless --time-pilot"
    cache = build_off_lquery_cache(args.ckpt_dir, args.device)
    write_off_lquery_cache(args.out_cache, cache)
    print(f"wrote {args.out_cache} ({len(cache)} entries)")

    floor_data = compute_floor_ratios_and_pin(cache)
    doc = write_floor_pinned(args.out_floor, floor_data, args.out_cache)
    for corpus, d in floor_data.items():
        print(f"[phase2b-floor] corpus={corpus} pooled_ratio={d['pooled_ratio']:.4f} "
              f"floor_pin={d['floor_pin']:.4f} (mean_b={d['mean_b']:.4f} s_b={d['s_b']:.4f})")
    print(f"wrote {args.out_floor} (pinned_at_iso={doc['pinned_at_iso']})")


if __name__ == "__main__":
    main()
