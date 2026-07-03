#!/usr/bin/env python3
"""analyze_eval_truncation_rd.py -- the real-data (RD) eval-time
rank-truncation staircase, the pre-registered fallback per
DELTANET_REALDATA_DESIGN.md section 14.7-16 now that train-time force-rank
has collapsed identically at k=K-1/K/K+1 on real language (see the Bprobe
record printed by --bprobe-dir below), mirroring
DELTANET_CAUSAL_RANK_DESIGN.md section 12.8's synthetic-DeltaNet method and
its stated verdict ("put the causal weight on (i) an exact-rank no-inflation
result plus (ii) eval-time optimal truncation").

WHY THIS SCRIPT LOADS DUMPS INSTEAD OF RETRAINING (unlike the synthetic
matrix-thinking/deltanet/analyze_eval_truncation.py, which retrained fresh
because NO archived run there carried a dump): run_deltanet_rd.py turns
--save-z ON BY DEFAULT (its own module docstring: "the project's own
'eval-truncation lesson' ... applies here from the start"), and every
completed RD result JSON in the local archive
(experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA,waveBprobe}/)
carries a `Z_dump` key -- verified programmatically by this script's
--verify-schema mode before any staircase is trusted. So no GPU/CUDA/fla is
needed here at all; this is pure numpy + stdlib, runs on a laptop.

DUMP SCHEMA (verified against run_deltanet_rd.py's own save-z block and
model_rd.py's bind()/forward(), not assumed):
  Z_dump = {
    "S_T_raw":            (4, d, d)  UNFORCED bind (force_rank_k=None
                            regardless of the run's own --force-rank-k) --
                            the object every eval-truncation below acts on.
    "s_ideal_effective":  (4, d, d)  sum_j v_eff_j @ k_eff_j^T, the
                            architecture-native ideal built from THIS run's
                            own (possibly non-orthonormal) k_eff/v_eff.
    "k_eff_items":        (4, K, d)  L2-normalized effective keys (the
                            kernel's literal write-time input rows).
    "v_eff_items":         (4, K, d)  raw effective values.
    "S_T_forced":         (4, d, d)  ONLY present if the run itself was
                            launched with --force-rank-k (e.g. Bprobe's
                            fr15/16/17): the TRAIN-TIME forced state, for
                            the interference cross-reference only, never
                            fed into this script's staircase.
  }
  4 EVAL EXAMPLES, hop_set=(1,) FIXED -- the batch that produced this dump
  was drawn with `sample_batch_rd(cfg, 4, gz, hop_set=(1,), ...)`
  (run_deltanet_rd.py's save-z block). Critically, `succ` (the per-row
  K-cycle permutation), `query_tokens`, `hops`, and `tgt_slot` are NOT
  dumped -- only S_T/k_eff/v_eff survive.

WHY THE STAIRCASE BELOW IS h=1 ONLY (a finding, not an oversight -- verify
before claiming): target_clause_index (model_rd.py, the mini-audit's FATAL
fix) maps hop-h target slot `tgt_slot` to clause `inv_succ[tgt_slot]`. At
h=1, `tgt_slot = succ[a_slot]`, so `inv_succ[tgt_slot] = a_slot` -- the
target clause IS the query's own slot, independent of succ. So querying
with entity i's own bind-time key (`k_eff_items[i]`, used here as the query
proxy -- see next paragraph) against `v_eff_items[i]` is the EXACT,
succ-independent h=1 readout, for every i = 0..K-1, requiring nothing beyond
what the dump has. h>=2 needs succ to know which entity is h hops from a
given start; it is not in the dump, and reconstructing it via nearest-
neighbor matching between k_eff/v_eff rows would decode via argmax over a
candidate set -- exactly what CLAUDE.md's standing rule prohibits ("Readout
must force EXACT CONTINUOUS recovery, never argmax/nearest-neighbor over a
codebook, when a rank>=K bound depends on it"). So h=1-only is the
principled boundary of what this dump can support, not a shortcut.

QUERY PROXY CAVEAT (stated once, applies to every number below): the
harness's REAL query path computes q_eff through a SEPARATE window
[buf...,KEY,REL,<Q>] via the same embed->conv->W_k weights (model_rd.py's
effective_key_window), which is close to but not identical to
k_eff_items[i] (extracted at the VALUE position of the bind window,
[buf...,KEY,REL,VALUE]) -- premise (iii)'s "alignment" cosine measures
exactly this gap, and DELTANET_REALDATA_DESIGN.md section 16.5 documents it
decaying to 0.6-0.7 in 5/11 Wave A cells. Using k_eff_items[i] as the query
IS the only query-shaped object the dump contains (no model weights are
dumped either, so the real query path cannot be recomputed even if the
token sequences were saved) -- this script's readout is therefore a
"write-key self-consistency" unbind test, structurally IDENTICAL to
model_rd.py's own [model 10] idealized-recall self-test
(`apply_state_power(S_design, keys_r, h1)`), not a bit-exact reproduction of
evaluate_pool()'s logged recovered_frac@0.9. --verify-schema cross-checks
this proxy against each run's own logged final-checkpoint number so the gap
(driven by alignment, not by rank) is measured, not glossed.

Usage:
  python analyze_eval_truncation_rd.py --smoke
      # synthetic hand-built near-degenerate operator: validates SVD
      # truncation + readout + entity-aligned theory formula agree, before
      # trusting any of it against real dumps (seconds, no data needed).
  python analyze_eval_truncation_rd.py --dir DIR [DIR ...] [--verify-schema]
      # discovers every *.json with a Z_dump and force_rank_k=None (the
      # unconstrained arm -- forced runs' S_T_raw is still an unconstrained
      # bind per the dump note, but from a model TRAINED under a different,
      # often-collapsed regime, so it is excluded from the primary
      # staircase and reported separately via --bprobe-dir instead),
      # groups by K, runs the truncation staircase, prints tables, writes
      # --out-json if given.
  python analyze_eval_truncation_rd.py --bprobe-dir DIR
      # prints the train-time force-rank interference record (fr=K-1/K/K+1
      # trajectories) for the write-up's Bprobe section -- no truncation,
      # just a formatted read of the already-logged checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

TAUS = (0.9, 0.95, 0.99)
# The task brief's fixed absolute grid (kept for cross-K comparability).
FIXED_K_GRID = (8, 12, 14, 15, 16, 17, 18, 20)


# ---------------------------------------------------------------------------
# Core numerics: SVD truncation + readout (pure numpy, batched over the 4
# dumped examples via numpy.linalg.svd's native stacked-matrix support).
# ---------------------------------------------------------------------------

def svd_truncate(S: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation of each (d,d) matrix in a (E,d,d) stack,
    via direct SVD (deterministic, no stochastic sketch -- the eigh(ZZ^T)
    path the harness uses internally is numerically equivalent by
    Eckart-Young; see rank_utils.truncate_to_rank's own docstring). A no-op
    (returns S unchanged) when k >= d."""
    d = S.shape[-1]
    if k >= d:
        return S
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)   # (E,d,d),(E,d),(E,d,d)
    Uk, svk, Vtk = U[..., :, :k], sv[..., :k], Vt[..., :k, :]
    return np.einsum("eik,ek,ekj->eij", Uk, svk, Vtk)


def cosine(a: np.ndarray, b: np.ndarray, axis=-1, eps=1e-12) -> np.ndarray:
    num = (a * b).sum(axis=axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)
    return num / np.clip(den, eps, None)


def unbind_readout(S: np.ndarray, k_eff: np.ndarray) -> np.ndarray:
    """pred[e,i,:] = S[e] @ k_eff[e,i,:] -- apply_state_power's h=1 update
    (deltanet_core.py: `torch.einsum('bij,bqj->bqi', S, cur)`), reproduced
    exactly in numpy. S: (E,d,d); k_eff: (E,K,d) -> (E,K,d)."""
    return np.einsum("eij,ekj->eki", S, k_eff)


def recovered_frac(cos: np.ndarray, tau: float = 0.9) -> float:
    return float((cos > tau).mean())


def cos_stats(cos: np.ndarray) -> dict:
    flat = cos.reshape(-1)
    d = {"n": int(flat.size), "mean_cos": float(flat.mean()),
         "cos_p10": float(np.quantile(flat, 0.10)),
         "cos_p50": float(np.quantile(flat, 0.50)),
         "cos_p90": float(np.quantile(flat, 0.90)),
         "frac_cos_lt_0.1": float((np.abs(flat) < 0.1).mean())}
    for tau in TAUS:
        d[f"recovered_frac@{tau}"] = recovered_frac(flat, tau)
    return d


def theory_entity_aligned(K: int, k: int) -> float:
    """Closed-form h=1 prediction under the synthetic design's 'one whole
    entity mode dropped, orthonormal keys' model (analyze_eval_truncation.py):
    recovered_frac@0.9 = max(K - m, 0) / K, m = max(K - k, 0) modes dropped.
    NOT expected to hold exactly on real data (measured key Gram deviation
    1.26-2.77 vs the synthetic design's tau=0.03 -- see DELTANET_REALDATA_
    DESIGN.md section 14.7 item 2); reported as a labeled clean-regime
    reference, not a fitted curve."""
    m = max(K - k, 0)
    return max(K - m, 0) / K


# ---------------------------------------------------------------------------
# Dump loading + discovery
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
        "run_force_rank_k": d.get("force_rank_k"), "trunc_impl": d.get("trunc_impl"),
        "complete": d.get("complete"),
        "S_T_raw": np.array(z["S_T_raw"], dtype=np.float64),
        "s_ideal": np.array(z["s_ideal_effective"], dtype=np.float64),
        "k_eff": np.array(z["k_eff_items"], dtype=np.float64),
        "v_eff": np.array(z["v_eff_items"], dtype=np.float64),
    }
    # cross-reference: the run's OWN logged final-checkpoint h=1 number
    # (real query path), for the proxy-readout validation check.
    ck = d.get("checkpoints") or []
    if ck:
        h1 = ck[-1].get("M2_in_distribution", {}).get("1", {})
        out["logged_final_step"] = ck[-1].get("step")
        out["logged_recovered_frac_h1"] = h1.get("recovered_frac@0.9")
        out["logged_entity_subspace_rank_h1"] = h1.get("entity_subspace_effective_rank_mean")
        out["logged_alignment_cos_min_h1"] = h1.get("alignment_cos_min")
    return out


def discover_unconstrained_runs(dirs: list[str]) -> list[dict]:
    """Every *.json with a Z_dump AND the RUN's own force_rank_k is None --
    S_T_raw is always an unconstrained bind per the dump note, but a forced
    run's model was TRAINED under a different (often collapsed) regime, so
    its S_T_raw is excluded from the primary staircase population (kept
    separate, see --bprobe-dir) to avoid silently mixing two populations."""
    runs = []
    seen_paths = set()
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.json"))):
            if os.path.basename(path) in ("AGGREGATE.json",) or path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                r = load_run(path)
            except Exception as e:
                print(f"  (skip {path}: {e!r})")
                continue
            if r is None:
                continue
            if r["run_force_rank_k"] is not None:
                continue
            runs.append(r)
    return runs


# ---------------------------------------------------------------------------
# Staircase
# ---------------------------------------------------------------------------

def per_k_grid(K: int, d: int) -> list[int]:
    rel = {K - 2, K - 1, K, K + 1, K + 2}
    grid = sorted({k for k in (set(FIXED_K_GRID) | rel) if 1 <= k <= d})
    return grid


def run_staircase(run: dict, k_grid: list[int]) -> dict:
    S, k_eff, v_eff, s_ideal = run["S_T_raw"], run["k_eff"], run["v_eff"], run["s_ideal"]
    K, d = run["K"], run["d_state"]
    out = {"k_grid": k_grid, "measured": {}, "ideal_truncated": {}, "theory_entity_aligned": {}}
    for k in k_grid:
        S_k = svd_truncate(S, k)
        cos_m = cosine(unbind_readout(S_k, k_eff), v_eff)
        out["measured"][k] = cos_stats(cos_m)

        Si_k = svd_truncate(s_ideal, k)
        cos_i = cosine(unbind_readout(Si_k, k_eff), v_eff)
        out["ideal_truncated"][k] = cos_stats(cos_i)

        out["theory_entity_aligned"][k] = theory_entity_aligned(K, k)
    # sanity: k>=d is a no-op on the measured state (Eckart-Young identity)
    return out


def aggregate_over_seeds(staircases: list[dict], k_grid: list[int], key: str,
                          stat: str = "recovered_frac@0.9") -> dict:
    agg = {}
    for k in k_grid:
        vals = [s[key][k][stat] for s in staircases if k in s[key]]
        if vals:
            agg[k] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    return agg


# ---------------------------------------------------------------------------
# Schema verification (--verify-schema): confirms the dump has exactly the
# fields this script relies on, reports shapes, and cross-checks the k=d
# (no-truncation) proxy readout against each run's own logged number.
# ---------------------------------------------------------------------------

def verify_schema(runs: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("DUMP SCHEMA VERIFICATION")
    print("=" * 100)
    if not runs:
        print("  (no runs)")
        return
    r0 = runs[0]
    print(f"  fields present (from {r0['name']}): S_T_raw {r0['S_T_raw'].shape}, "
          f"s_ideal_effective {r0['s_ideal'].shape}, k_eff_items {r0['k_eff'].shape}, "
          f"v_eff_items {r0['v_eff'].shape}")
    print(f"  n_eval_examples={r0['S_T_raw'].shape[0]} (fixed at hop_set=(1,) by "
          f"run_deltanet_rd.py's save-z block); NOT present: succ, query_tokens, hops, tgt_slot")
    print(f"\n  {'run':<38}{'K':>4}{'d':>5}{'n_items(=4*K)':>15}"
          f"{'proxy@k=d rec@0.9':>20}{'logged rec@0.9':>16}{'align_min':>11}")
    for r in runs:
        S, k_eff, v_eff = r["S_T_raw"], r["k_eff"], r["v_eff"]
        cos_full = cosine(unbind_readout(S, k_eff), v_eff)
        proxy = recovered_frac(cos_full)
        logged = r.get("logged_recovered_frac_h1")
        align = r.get("logged_alignment_cos_min_h1")
        logged_s = f"{logged:.4f}" if logged is not None else "n/a"
        align_s = f"{align:.4f}" if align is not None else "n/a"
        print(f"  {r['name']:<38}{r['K']:>4}{r['d_state']:>5}{S.shape[0]*r['K']:>15}"
              f"{proxy:>20.4f}{logged_s:>16}{align_s:>11}")
    print("  (proxy uses k_eff_items as the query -- see module docstring's QUERY PROXY CAVEAT;")
    print("   expect proxy ~= logged when align_min ~= 1, proxy > logged when align_min is low --")
    print("   that gap is the alignment-decay phenomenon (DELTANET_REALDATA_DESIGN.md section 16.5),")
    print("   not a rank effect.)")


# ---------------------------------------------------------------------------
# Bprobe interference record (train-time force-rank arm; no truncation math
# here, just a formatted read of the already-logged trajectories).
# ---------------------------------------------------------------------------

def bprobe_record(bprobe_dir: str) -> None:
    print("\n" + "=" * 100)
    print(f"BPROBE TRAIN-TIME FORCE-RANK INTERFERENCE RECORD ({bprobe_dir})")
    print("=" * 100)
    rows = []
    for path in sorted(glob.glob(os.path.join(bprobe_dir, "*.json"))):
        name = os.path.basename(path)
        if name == "AGGREGATE.json":
            continue
        with open(path) as f:
            d = json.load(f)
        fr = d.get("force_rank_k")
        if fr is None:
            continue
        ck = d.get("checkpoints") or []
        if not ck:
            continue
        last = ck[-1]
        h1 = last.get("M2_in_distribution", {}).get("1", {})
        rows.append({
            "name": name, "fr": fr, "seed": d.get("seed"),
            "complete": d.get("complete"), "step": last.get("step"),
            "steps_total": d.get("steps"),
            "rec@0.9": h1.get("recovered_frac@0.9"),
            "esr": h1.get("entity_subspace_effective_rank_mean"),
            "align_min": h1.get("alignment_cos_min"),
            "step6000_rec@0.9": next((c["M2_in_distribution"]["1"]["recovered_frac@0.9"]
                                       for c in ck if c["step"] == 6000), None),
        })
    rows.sort(key=lambda r: (r["fr"], r["seed"]))
    print(f"  {'run':<38}{'fr':>4}{'complete':>10}{'step':>8}{'rec@0.9':>10}"
          f"{'esr':>8}{'align_min':>11}{'rec@0.9(step6000)':>20}")
    for r in rows:
        rec_s = f"{r['rec@0.9']:.4f}" if r["rec@0.9"] is not None else "n/a"
        esr_s = f"{r['esr']:.3f}" if r["esr"] is not None else "n/a"
        al_s = f"{r['align_min']:.4f}" if r["align_min"] is not None else "n/a"
        s6_s = f"{r['step6000_rec@0.9']:.4f}" if r["step6000_rec@0.9"] is not None else "n/a"
        print(f"  {r['name']:<38}{r['fr']:>4}{str(r['complete']):>10}{r['step']:>8}"
              f"{rec_s:>10}{esr_s:>8}{al_s:>11}{s6_s:>20}")
    incomplete = [r for r in rows if not r["complete"]]
    if incomplete:
        print(f"\n  NOTE: {len(incomplete)}/{len(rows)} runs are NON-FINAL (partial, "
              f"'complete': false) -- reported as in-flight reads, not headline numbers.")


# ---------------------------------------------------------------------------
# Smoke test (hand-built near-degenerate operator, no data/GPU needed)
# ---------------------------------------------------------------------------

def _smoke() -> None:
    rng = np.random.default_rng(0)
    d, K = 32, 16
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    keys = Q[:, :K].T                                   # (K,d) orthonormal rows
    succ = np.roll(np.arange(K), shift=-1)               # single K-cycle
    values = keys[succ]                                  # v_i = k_succ(i)
    scale = 1.0 - 0.01 * np.arange(K)
    m = 3
    scale[m] = 1.0 - 0.01 * K                             # force mode m smallest
    S_scaled = np.einsum("k,ki,kj->ij", scale, values, keys)
    S_batch = S_scaled[None, :, :]
    k_eff_batch = keys[None, :, :]
    v_eff_batch = values[None, :, :]

    # k=K-1 should drop exactly mode m -> bimodal (K-1)/K at cos>0.9,
    # matching theory_entity_aligned(K, K-1) = (K-1)/K.
    S_trunc = svd_truncate(S_batch, K - 1)
    cos = cosine(unbind_readout(S_trunc, k_eff_batch), v_eff_batch).reshape(-1)
    measured = recovered_frac(cos, 0.9)
    expect = theory_entity_aligned(K, K - 1)
    assert abs(measured - expect) < 1e-6, f"measured {measured:.6f} != theory {expect:.6f}"
    print(f"[smoke 1] k=K-1 entity-aligned drop: measured recovered_frac@0.9={measured:.6f} "
          f"== theory (K-1)/K={expect:.6f}")

    # k=K (no real truncation of an exactly-rank-K ideal-shaped operator
    # restricted to its own K-dim span) must reproduce cos~1 everywhere for
    # the UNSCALED ideal (S_ideal, not S_scaled -- S_scaled is already
    # rank-K by construction, so k=K truncation of it is also a no-op).
    S_full_trunc = svd_truncate(S_batch, K)
    diff = np.abs(S_full_trunc - S_batch).max()
    assert diff < 1e-8, f"k=K truncation of an already-rank-K operator changed it: {diff:.2e}"
    print(f"[smoke 2] k=K truncation is a no-op on an already-rank-K operator (max diff {diff:.2e})")

    # k>d is a no-op by construction (svd_truncate short-circuits)
    S_noop = svd_truncate(S_batch, d + 10)
    assert np.array_equal(S_noop, S_batch)
    print("[smoke 3] k>d short-circuits to a literal no-op (identity check, not just close)")

    # ideal-truncated path: truncating s_ideal_effective-equivalent (here,
    # the UNSCALED S_ideal) at k=K-1 must ALSO show the bimodal drop, since
    # it is exactly the same construction as S_scaled at k=K-1 restricted to
    # this K-dim orthonormal frame (scale doesn't change singular DIRECTIONS
    # here because keys/values are already orthonormal -- the m'th mode is
    # still the smallest by the same forced ordering).
    S_ideal = np.einsum("ki,kj->ij", values, keys)[None, :, :]
    S_ideal_trunc = svd_truncate(S_ideal, K - 1)
    # the UNSCALED ideal is exactly degenerate (all singular values equal),
    # so which mode gets dropped at k=K-1 is an SVD tie-break, not
    # necessarily m -- this smoke check instead confirms k=K is a full-rank
    # no-op and k=K-1 drops exactly ONE mode (rank goes from K to K-1).
    rank_before = np.linalg.matrix_rank(S_ideal[0])
    rank_after = np.linalg.matrix_rank(S_ideal_trunc[0])
    assert rank_before == K and rank_after == K - 1, \
        f"expected rank K={K} -> K-1={K-1}, got {rank_before} -> {rank_after}"
    print(f"[smoke 4] ideal-truncated path: rank {rank_before} -> {rank_after} at k=K-1 (exactly one "
          f"mode dropped, as the entity-aligned theory assumes)")

    print("\nanalyze_eval_truncation_rd smoke test PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_staircase_tables(by_K: dict) -> None:
    for K in sorted(by_K):
        staircases = by_K[K]["staircases"]
        k_grid = by_K[K]["k_grid"]
        n_seeds = len(staircases)
        print("\n" + "=" * 100)
        print(f"K={K}  (n_seeds={n_seeds})  h=1 EVAL-TRUNCATION STAIRCASE -- recovered_frac@0.9, "
              f"mean over seeds (std in parens)")
        print("=" * 100)
        header = "source".ljust(18) + "".join(f"{k:>12d}" for k in k_grid)
        print(header)
        for label, key in (("measured (S_T)", "measured"), ("ideal-truncated", "ideal_truncated")):
            agg = aggregate_over_seeds(staircases, k_grid, key)
            row = [label.ljust(18)]
            for k in k_grid:
                if k in agg:
                    m, s, n = agg[k]
                    row.append(f"{m:>7.4f}±{s:<4.3f}")
                else:
                    row.append(f"{'--':>12}")
            print("".join(row))
        theory_row = ["theory (K-m)/K".ljust(18)]
        for k in k_grid:
            theory_row.append(f"{theory_entity_aligned(K, k):>12.4f}")
        print("".join(theory_row))
        print(f"  seeds: {', '.join(s['name'] for s in staircases)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dir", nargs="+", default=None,
                     help="one or more directories of RD result JSONs with Z_dump (unconstrained arm)")
    ap.add_argument("--bprobe-dir", default=None,
                     help="directory of train-time force-rank result JSONs (Bprobe) -- prints the "
                          "interference record, no truncation math")
    ap.add_argument("--k-grid", type=int, nargs="+", default=None,
                     help="override the fixed absolute grid (default: FIXED_K_GRID union'd per-K "
                          "with {K-2..K+2})")
    ap.add_argument("--verify-schema", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return

    if args.bprobe_dir:
        bprobe_record(args.bprobe_dir)

    if not args.dir:
        if not args.bprobe_dir:
            print("nothing to do: pass --smoke, --dir, or --bprobe-dir", file=sys.stderr)
            sys.exit(1)
        return

    runs = discover_unconstrained_runs(args.dir)
    print(f"discovered {len(runs)} unconstrained (force_rank_k=None) runs with Z_dump across {args.dir}")
    by_K_names = {}
    for r in runs:
        by_K_names.setdefault(r["K"], []).append(r["name"])
    for K, names in sorted(by_K_names.items()):
        print(f"  K={K}: {len(names)} run(s): {names}")

    if args.verify_schema:
        verify_schema(runs)

    by_K = {}
    for r in runs:
        K, d = r["K"], r["d_state"]
        grid = args.k_grid if args.k_grid else per_k_grid(K, d)
        stair = run_staircase(r, grid)
        by_K.setdefault(K, {"k_grid": grid, "staircases": [], "runs": []})
        by_K[K]["staircases"].append(stair | {"name": r["name"], "seed": r["seed"]})
        by_K[K]["runs"].append(r["name"])

    print_staircase_tables(by_K)

    if args.out_json:
        out = {}
        for K, info in by_K.items():
            out[str(K)] = {
                "k_grid": info["k_grid"], "runs": info["runs"],
                "measured_agg": {str(k): v for k, v in
                                  aggregate_over_seeds(info["staircases"], info["k_grid"], "measured").items()},
                "ideal_truncated_agg": {str(k): v for k, v in
                                         aggregate_over_seeds(info["staircases"], info["k_grid"],
                                                               "ideal_truncated").items()},
                "theory_entity_aligned": {str(k): theory_entity_aligned(K, k) for k in info["k_grid"]},
                "per_seed": [{"name": s["name"], "seed": s["seed"],
                              "measured": {str(k): v for k, v in s["measured"].items()},
                              "ideal_truncated": {str(k): v for k, v in s["ideal_truncated"].items()}}
                             for s in info["staircases"]],
            }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
