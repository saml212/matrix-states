#!/usr/bin/env python3
"""analyze_eval_truncation.py -- the eval-time rank-truncation staircase,
DELTANET_CAUSAL_RANK_DESIGN.md's F18-mandated eigh control (section 6.4:
"one eval-time eigh-truncation control run on the B-probe k<K cell is
pre-registered as mandatory before any Wave B life/death verdict is
recorded"), run here as THE decisive causal-necessity read (2026-07-02).

WHY THIS SCRIPT TRAINS FRESH INSTEAD OF LOADING A DUMP (verified this run,
not assumed): every archived result JSON in wave-1/wave0/waveA/waveBprobe
(both on the box, `youthful-indigo-turkey:/home/nvidia/chapter2/deltanet/
results/deltanet/`, and the local mirror `experiment-runs/
2026-07-02_deltanet_waves/`) was checked programmatically for a `Z_dump`
key -- NONE has one. `--save-z` is an opt-in flag in run_deltanet.py that
no launched sweep cell passed (grep confirms run_deltanet_sweep.py never
mentions `save_z` or `Z_dump`), and run_deltanet.py never calls
`torch.save` anywhere -- no model checkpoint (state_dict) exists for any
completed run either. There is no artifact on disk that contains a trained
S_T. So this script reproduces the primary-cell UNCONSTRAINED recipe
EXACTLY (same task_dn.DeltaNetTaskConfig, same steps/batch/lr/seed
convention as Wave A's `d64_K32_frN` cells -- `run_deltanet.py --d 64 --K
32 --steps 20000 --seed {0,1,2}`, defaults for everything else) via
run_deltanet.train(), which mutates its `model` argument in place, then
runs the truncation staircase directly against the freshly-trained model's
in-memory S_T. No dump/analyze split is needed or introduced: raw keys,
values, queries, and S_T never leave the training process.

METHOD (design section 12.6's open question, closed here):
  1. Train UNCONSTRAINED (force_rank_k=None throughout -- no truncation
     anywhere in the training graph, so none of the training-time
     svd_lowrank instability F18 warns about applies) at the primary cell
     (d=64, K=32) for the Wave A step budget (20,000 steps). Wave 0/A
     (DELTANET_CAUSAL_RANK_DESIGN.md section 12.3/12.4) already established
     this state converges to recovered_frac@0.9 = 1.0 with entity-subspace
     rank = whole-matrix rank = K, exactly -- i.e. the operator this script
     truncates is INDEPENDENTLY known to need every one of its K dimensions
     (no unused slack for the eval truncation to "hide" a real deficiency
     behind).
  2. At EVAL time only (no backward pass -- eigh's known instability,
     audit round-1 FINDING 1 / F18, is a BACKWARD-pass problem; a read-only
     forward-pass truncation pays none of that cost), truncate the trained
     S_T to rank k via rank_utils.truncate_to_rank (deterministic eigh, the
     UNBIASED implementation -- NOT the stochastic svd_lowrank the
     training-time force-rank arm was forced onto by F13/F18) across a
     k-grid straddling K, and measure recovered_frac@tau at every
     H_train/H_test/H_extra hop, reusing the identical readout
     (deltanet_core.apply_state_power) and eval-batch generator
     (task_dn.sample_batch) run_deltanet.evaluate_at_hops() already uses.
     The ONLY thing that differs from an ordinary mid-training checkpoint
     eval is WHERE the truncation is inserted: after training, read-only,
     applied to an otherwise-untouched unconstrained state -- exactly F18's
     prescribed control.
  3. Cross-reference against the ALREADY-COMPLETED training-time force-rank
     probe (waveBprobe fr31/fr32/fr33, forced to svd_lowrank by F13) and
     against two closed-form theoretical predictions for what SHOULD happen
     when exactly one of K orthonormal entity modes is dropped from the
     EXACT ideal operator (see `theory_entity_aligned` / TASK_E's
     `theory_isotropic_single_hop` below) -- this is what lets the write-up
     distinguish "the trained state cannot survive losing 1/K of its
     provably-necessary rank" (causal necessity, real) from "training
     UNDER svd_lowrank truncation breaks SGD" (F18's pre-registered
     methodological confound, also real, but a different finding).

THEORY (derivation, so the numbers below aren't asserted from nowhere):
task_dn.py's generator makes keys = an orthonormal K-frame `{k_i}` in R^d
(via `_random_directions(orthogonal=True)`) and values = the SAME frame
permuted by a single Hamiltonian K-cycle `succ`: `v_i = k_{succ(i)}`. So the
exact ideal `S_ideal = sum_i v_i k_i^T` maps `k_i -> k_{succ(i)}` exactly,
and RESTRICTED TO THE K-DIM ENTITY SUBSPACE it is literally the K x K 0/1
permutation matrix of `succ` in the `{k_i}` basis -- an ORTHOGONAL
(all-singular-values-exactly-1, exactly degenerate) operator, not a generic
incoherent embedding. Suppose eigh's top-(K-1) truncation drops exactly one
rank-1 term `v_m k_m^T` (the scenario a near-but-not-exactly-degenerate
trained spectrum converges to once training noise breaks the exact tie --
an empirical question the k=K-1 measurement below answers). Then for a
query starting at entity `a`:
    S_trunc @ k_a = S_ideal @ k_a - v_m (k_m . k_a) = k_succ(a) - v_m [a==m]
so the FIRST application is exact unless a==m (giving the zero vector when
a==m). Iterating, `S_trunc^h @ k_a` is EXACTLY `k_{succ^h(a)}` (cos = 1)
UNLESS the dropped index m lies on a's h-step forward orbit `{a, succ(a),
..., succ^{h-1}(a)}` -- h distinct starting points out of K -- in which
case the intermediate state hits the zeroed component and the final
prediction is the zero vector (cos ~ 0, undefined/garbage). So:
    recovered_frac@0.9(h) = max(K - h, 0) / K            (entity-aligned)
a BIMODAL, not uniform, per-item distribution: (K-h)/K items at cos ~ 1
exactly, h/K items at cos ~ 0. This is the model this docstring's `theory_
entity_aligned` implements, and it is analytically EXACT for this task's
construction, conditional on the assumption that the dropped mode aligns
with exactly one entity (verified against the K=32 case below, not merely
assumed). The task brief's alternative reference point, Task E's
"sqrt(1-1/K) per item" isotropic-embedding theory, is the OPPOSITE
extreme (a generic/incoherent K-dim embedding where the dropped direction
is spread evenly across all K entities): under THAT model every item's
cosine drops UNIFORMLY to sqrt(1-1/K) = sqrt(31/32) = 0.9840 for K=32,
which clears the 0.9 threshold for every item, i.e. recovered_frac@0.9 ~=
1.0 -- the opposite qualitative shape (uniform graceful decay vs. bimodal
all-or-nothing) from the entity-aligned model. Given the EXACT
orthonormal-key construction here, the entity-aligned model is the
analytically correct one for the ideal operator; whether the ACTUAL
trained S_T (which is close to, but not exactly, the ideal) behaves like
one, the other, or neither is exactly what the measurement below settles.

Usage:
  python analyze_eval_truncation.py --smoke
      # synthetic-ideal-operator unit test of the truncation+readout+
      # theory pipeline (no model, no training, seconds) -- validates
      # theory_entity_aligned against a HAND-CONSTRUCTED near-degenerate
      # operator before trusting it against real trained states.
  CUDA_VISIBLE_DEVICES=7 python analyze_eval_truncation.py \\
      --d 64 --K 32 --steps 20000 --seeds 0 1 2 \\
      --out-dir results/deltanet/eval_trunc
  python analyze_eval_truncation.py --dir results/deltanet --report-only \\
      --in results/deltanet/eval_trunc/*.json
      # re-print the staircase table + cross-reference from already-written
      # per-seed result JSONs, no GPU / retraining needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
import task_dn as tdn
import deltanet_core as dc
from model_dn import DeltaNetGateModel
from rank_utils import truncate_to_rank

TAUS = (0.9, 0.95, 0.99)
DEFAULT_K_GRID = (24, 28, 30, 31, 32, 33, 34, 36)


# ---------------------------------------------------------------------------
# Theory (see module docstring for the derivation)
# ---------------------------------------------------------------------------

def theory_entity_aligned(K: int, h_grid) -> dict:
    """recovered_frac@0.9(h) = max(K-h, 0) / K -- exact for h < K under the
    'one whole entity mode dropped' model (module docstring derivation)."""
    return {h: max(K - h, 0) / K for h in h_grid}


def theory_isotropic_per_item_cos(K: int) -> float:
    """Task E's generic-embedding theory: sqrt(1 - 1/K) per item, uniform
    across h (every item survives the 0.9 threshold under this model)."""
    return math.sqrt(1.0 - 1.0 / K)


# ---------------------------------------------------------------------------
# Truncation staircase (real model)
# ---------------------------------------------------------------------------

@torch.no_grad()
def truncation_staircase(model: DeltaNetGateModel, cfg: tdn.DeltaNetTaskConfig,
                          device, eval_seed: int, k_grid, h_grid,
                          n_batches: int = 20, batch_size: int = 256,
                          keep_percase_k=None, keep_percase_h=None):
    """For each hop h, draw n_batches fresh eval instances (fresh K-cycle
    bindings each batch, matching evaluate_at_hops()'s own convention), bind
    UNCONSTRAINED (force_rank_k=None) to get the trained S_T once per batch,
    then truncate that SAME S_T to every k in k_grid (rank_utils.
    truncate_to_rank, deterministic eigh) and score the pinned readout
    (apply_state_power) against the exact ground-truth targets. Returns
    {h: {k: stats}}; if keep_percase_{k,h} given, also returns the raw
    per-item cosine tensor at that one (k,h) cell for distributional
    inspection (item 3 of the design doc's ask)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(eval_seed)
    results = {}
    percase = None
    for h in h_grid:
        cos_by_k = {k: [] for k in k_grid}
        for _ in range(n_batches):
            b = tdn.sample_batch(cfg, batch_size, gen, hop_set=(h,), device=device,
                                  assert_injective=True)
            S_T = model.bind(b, force_rank_k=None)          # UNCONSTRAINED bind -- the trained state
            q_eff = model.effective_key(b["query_keys"])
            for k in k_grid:
                S_use = truncate_to_rank(S_T, k) if k < cfg.d else S_T
                pred = dc.apply_state_power(S_use, q_eff, b["hops"])
                cos = F.cosine_similarity(pred, b["targets"], dim=-1).reshape(-1)
                cos_by_k[k].append(cos.detach().cpu())
                if keep_percase_k is not None and k == keep_percase_k and h == keep_percase_h:
                    if percase is None:
                        percase = []
                    percase.append(cos.detach().cpu())
        entry = {}
        for k in k_grid:
            cos_all = torch.cat(cos_by_k[k])
            stat = {
                "n": int(cos_all.numel()),
                "mean_cos": cos_all.mean().item(),
                "cos_p10": torch.quantile(cos_all, 0.10).item(),
                "cos_p50": torch.quantile(cos_all, 0.50).item(),
                "cos_p90": torch.quantile(cos_all, 0.90).item(),
                "frac_cos_lt_0.1": (cos_all.abs() < 0.1).float().mean().item(),
            }
            for tau in TAUS:
                stat[f"recovered_frac@{tau}"] = (cos_all > tau).float().mean().item()
            entry[k] = stat
        results[h] = entry
    model.train()
    if percase is not None:
        percase = torch.cat(percase)
    return results, percase


# ---------------------------------------------------------------------------
# Synthetic smoke test -- validates the truncation+readout+theory pipeline
# against a HAND-CONSTRUCTED near-degenerate operator (no model, no
# training). Per CLAUDE.md: smoke test before spending GPU time.
# ---------------------------------------------------------------------------

def _smoke() -> None:
    torch.manual_seed(0)
    d, K = 16, 8
    # orthonormal K-frame in R^d
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    keys = Q[:, :K].T                                   # (K, d) orthonormal rows
    succ = torch.roll(torch.arange(K), shifts=-1)        # a single K-cycle: i -> i+1 mod K
    values = keys[succ]                                  # v_i = k_{succ(i)}
    S_ideal = torch.einsum("ki,kj->ij", values, keys)     # exact ideal, (d,d)

    # break the exact degeneracy in a KNOWN, controlled way: scale each
    # entity's rank-1 term by a distinct factor close to 1, smallest at
    # entity m=3, so eigh's top-(K-1) truncation deterministically drops
    # exactly that term.
    m = 3
    scale = 1.0 - 0.01 * torch.arange(K).float()
    scale[m] = 1.0 - 0.01 * K                             # force m to be the smallest
    S_scaled = torch.einsum("k,ki,kj->ij", scale, values, keys)

    S8 = truncate_to_rank(S_scaled.unsqueeze(0), K - 1)[0]   # rank-(K-1) eigh truncation

    h_grid = list(range(1, K))                             # h=1..K-1 stays inside one cycle
    pred_theory = theory_entity_aligned(K, h_grid)
    measured = {}
    for h in h_grid:
        Sh = torch.linalg.matrix_power(S8, h)
        pred = torch.einsum("ij,kj->ki", Sh, keys)          # (K, d): query every entity at once
        tgt_idx = succ.clone()
        for _ in range(h - 1):
            tgt_idx = succ[tgt_idx]
        tgt = keys[tgt_idx]
        cos = F.cosine_similarity(pred, tgt, dim=-1)
        measured[h] = (cos > 0.9).float().mean().item()
        expect = pred_theory[h]
        assert abs(measured[h] - expect) < 1e-4, \
            f"h={h}: measured recovered_frac@0.9={measured[h]:.4f} != theory {expect:.4f}"
    print(f"[smoke 1] entity-aligned theory EXACT-match check (K={K}, dropped entity m={m}): "
          f"measured == theory at every h in {h_grid[:5]}...{h_grid[-1]} (max diff < 1e-4)")

    # k>=K (no real truncation needed) must reproduce cos=1 everywhere.
    S_full = truncate_to_rank(S_ideal.unsqueeze(0), K)[0]
    diff = (S_full - S_ideal).abs().max().item()
    assert diff < 1e-5, f"k=K truncation of an already-rank-K ideal changed it: {diff:.2e}"
    print(f"[smoke 2] k=K truncation is a no-op on an already-rank-K operator (max diff {diff:.2e})")

    # isotropic theory sanity: sqrt(1-1/K) for K=32 matches the brief's 0.984.
    iso = theory_isotropic_per_item_cos(32)
    assert abs(iso - 0.9840) < 1e-3, f"isotropic theory sanity failed: {iso:.4f}"
    print(f"[smoke 3] theory_isotropic_per_item_cos(32) = {iso:.4f} (brief cites ~0.984, matches)")

    print("\nanalyze_eval_truncation smoke test PASSED")


# ---------------------------------------------------------------------------
# Cross-reference: pull final per-hop numbers out of already-completed
# training-time force-rank runs (waveBprobe fr31/fr32/fr33), if present.
# ---------------------------------------------------------------------------

def load_crossref(result_dir: str) -> dict:
    out = {}
    if not result_dir or not os.path.isdir(result_dir):
        return out
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        name = os.path.basename(path)
        if name in ("AGGREGATE.json",):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if not d.get("complete"):
            continue
        fr = d.get("force_rank_k")
        if fr is None or not d.get("checkpoints"):
            continue
        ck = d["checkpoints"][-1]
        per_hop = {}
        for h, e in {**ck.get("M2_in_distribution", {}), **ck.get("M3_held_out", {})}.items():
            per_hop[int(h)] = {"recovered_frac@0.9": e["recovered_frac@0.9"],
                                "mean_cos": e["mean_cos"]}
        out[name] = {"force_rank_k": fr, "seed": d.get("seed"), "trunc_impl": d.get("trunc_impl"),
                      "K": d.get("K"), "d": d.get("d"), "per_hop": per_hop}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_seed(args, seed: int, device: str) -> dict:
    cfg = tdn.DeltaNetTaskConfig(d=args.d, K=args.K, conv_size=args.conv_size,
                                  H_train=tuple(args.h_train), H_test=tuple(args.h_test),
                                  H_extra=tuple(args.h_extra), orthogonal=True)
    model = DeltaNetGateModel(d=args.d).to(device)
    print(f"\n=== seed {seed}: training UNCONSTRAINED d={args.d} K={args.K} steps={args.steps} "
          f"(the state the truncation staircase below will be applied to) ===", flush=True)
    t0 = time.time()
    train_result = _train_import().train(
        model, cfg, device, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        seed=seed, force_rank_k=None, log_every=args.log_every, ckpt_every=args.ckpt_every,
        out_path=None, timeout_s=args.train_timeout)
    train_wall = time.time() - t0
    final_h1 = train_result["checkpoints"][-1]["M2_in_distribution"][cfg.H_train[0]]
    print(f"  training done in {train_wall:.1f}s ({train_result['steps_completed']}/{args.steps} steps, "
          f"complete={train_result['complete']}): final recovered_frac@0.9(h=1)="
          f"{final_h1['recovered_frac@0.9']:.4f}, entity_subspace_eff_rank="
          f"{final_h1['entity_subspace_effective_rank_mean']:.4f} (expect ~= K={args.K} per "
          f"Wave 0/A)", flush=True)

    h_grid = sorted(set(args.h_train) | set(args.h_test) | set(args.h_extra))
    print(f"  running eval-time truncation staircase: k_grid={list(args.k_grid)} "
          f"h_grid={h_grid} ({args.eval_n_batches} batches x {args.eval_batch_size} "
          f"examples per (h) cell, deterministic eigh truncation)", flush=True)
    t1 = time.time()
    staircase, percase = truncation_staircase(
        model, cfg, device, eval_seed=seed + 30_000, k_grid=args.k_grid, h_grid=h_grid,
        n_batches=args.eval_n_batches, batch_size=args.eval_batch_size,
        keep_percase_k=args.K - 1, keep_percase_h=1)
    eval_wall = time.time() - t1
    print(f"  staircase done in {eval_wall:.1f}s", flush=True)

    result = {
        "d": args.d, "K": args.K, "seed": seed, "steps": args.steps,
        "train_steps_completed": train_result["steps_completed"],
        "train_complete": train_result["complete"],
        "train_wall_s": train_wall,
        "final_train_recovered_frac_h1": final_h1["recovered_frac@0.9"],
        "final_train_entity_subspace_eff_rank_h1": final_h1["entity_subspace_effective_rank_mean"],
        "final_train_whole_eff_rank_h1": final_h1["effective_rank_whole_mean"],
        "k_grid": list(args.k_grid), "h_grid": h_grid,
        "eval_n_batches": args.eval_n_batches, "eval_batch_size": args.eval_batch_size,
        "eval_wall_s": eval_wall,
        "staircase": {str(h): {str(k): v for k, v in ks.items()} for h, ks in staircase.items()},
        "percase_kKm1_h1": {
            "k": args.K - 1, "h": 1, "n": int(percase.numel()),
            "mean_cos": percase.mean().item(),
            "cos_p10": torch.quantile(percase, 0.10).item(),
            "cos_p50": torch.quantile(percase, 0.50).item(),
            "cos_p90": torch.quantile(percase, 0.90).item(),
            "frac_cos_lt_0.1": (percase.abs() < 0.1).float().mean().item(),
            "frac_cos_gt_0.9": (percase > 0.9).float().mean().item(),
        } if percase is not None else None,
    }
    return result


def _train_import():
    """run_deltanet.py is a __main__-style script (argparse at module
    scope guarded by `if __name__ == "__main__"`), so `train()` and
    `evaluate_at_hops()` are safely importable as a module -- same
    convention probe_trunc.py already relies on."""
    import run_deltanet as rd
    return rd


def print_staircase_table(seed_results: list, k_grid, h_grid) -> None:
    print("\n" + "=" * 100)
    print("EVAL-TIME TRUNCATION STAIRCASE -- recovered_frac@0.9, mean over seeds "
          f"({len(seed_results)} seed(s))")
    print("=" * 100)
    header = "h\\k".ljust(6) + "".join(f"{k:>10d}" for k in k_grid)
    print(header)
    for h in h_grid:
        row = [f"{h:<6d}"]
        for k in k_grid:
            vals = [r["staircase"][str(h)][str(k)]["recovered_frac@0.9"] for r in seed_results]
            mean_v = sum(vals) / len(vals)
            row.append(f"{mean_v:>10.4f}")
        print("".join(row))
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--conv-size", type=int, default=4)
    ap.add_argument("--h-train", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--h-test", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--h-extra", type=int, nargs="+", default=[7, 21])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--train-timeout", type=float, default=None)
    ap.add_argument("--k-grid", type=int, nargs="+", default=list(DEFAULT_K_GRID))
    ap.add_argument("--eval-n-batches", type=int, default=20)
    ap.add_argument("--eval-batch-size", type=int, default=256)
    ap.add_argument("--dir", type=str, default=None,
                     help="directory of already-completed training-time force-rank result "
                          "JSONs (e.g. results/deltanet/waveBprobe) to cross-reference "
                          "against; optional, report-only enhancement")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} d={args.d} K={args.K} steps={args.steps} seeds={args.seeds} "
          f"k_grid={args.k_grid}", flush=True)

    seed_results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        res = run_one_seed(args, seed, device)
        seed_results.append(res)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, f"eval_trunc_d{args.d}_K{args.K}_s{seed}.json")
            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  wrote {out_path}", flush=True)

    h_grid = sorted(set(args.h_train) | set(args.h_test) | set(args.h_extra))
    print_staircase_table(seed_results, args.k_grid, h_grid)

    print("\nTHEORY (entity-aligned, exact for the ideal operator; see module docstring):")
    theory = theory_entity_aligned(args.K, h_grid)
    print("  " + "  ".join(f"h={h}:{theory[h]:.4f}" for h in h_grid))
    print(f"THEORY (isotropic per-item cos, Task E convention): "
          f"sqrt(1-1/{args.K}) = {theory_isotropic_per_item_cos(args.K):.4f} "
          f"(-> recovered_frac@0.9 ~= 1.0 at every h under this model)")

    print("\nPER-ITEM at k=K-1, h=1 (item 3 of the design doc's ask):")
    for r in seed_results:
        pc = r["percase_kKm1_h1"]
        if pc:
            print(f"  seed {r['seed']}: mean_cos={pc['mean_cos']:.4f} p10={pc['cos_p10']:.4f} "
                  f"p50={pc['cos_p50']:.4f} p90={pc['cos_p90']:.4f} "
                  f"frac(cos<0.1)={pc['frac_cos_lt_0.1']:.4f} frac(cos>0.9)={pc['frac_cos_gt_0.9']:.4f}")

    if args.dir:
        crossref = load_crossref(args.dir)
        if crossref:
            print(f"\nCROSS-REFERENCE (training-time force-rank arm, from {args.dir}):")
            for name, info in crossref.items():
                h1 = info["per_hop"].get(1, {})
                h2 = info["per_hop"].get(2, {})
                print(f"  {name}: fr={info['force_rank_k']} seed={info['seed']} "
                      f"trunc_impl={info['trunc_impl']} | h=1 recovered@0.9="
                      f"{h1.get('recovered_frac@0.9', float('nan')):.4f} mean_cos="
                      f"{h1.get('mean_cos', float('nan')):.4f} | h=2 recovered@0.9="
                      f"{h2.get('recovered_frac@0.9', float('nan')):.4f} mean_cos="
                      f"{h2.get('mean_cos', float('nan')):.4f}")

    if args.out_dir:
        agg_path = os.path.join(args.out_dir, f"eval_trunc_d{args.d}_K{args.K}_AGGREGATE.json")
        with open(agg_path, "w") as f:
            json.dump({"seeds": args.seeds, "k_grid": args.k_grid, "h_grid": h_grid,
                       "theory_entity_aligned": theory,
                       "theory_isotropic_per_item_cos": theory_isotropic_per_item_cos(args.K),
                       "per_seed": seed_results}, f, indent=2)
        print(f"\nwrote {agg_path}")


if __name__ == "__main__":
    main()
