"""Stage-1 minimal-gate launcher + aggregator for Task D.

Runs the pre-registered Stage-1 grid (TASK_D_PREREGISTRATION.md §6/§7), MATRIX
MODEL ONLY, and aggregates into a preliminary PASS / FAIL / INCONCLUSIVE verdict.
The launcher's verdict is advisory — the human applies the §6 criteria to the
saved numbers. Runs sequentially on one GPU (~5-10 GPU-h). Stage-2 parallelizes.

  M1  effective/stable rank of Z vs K   (unconstrained, K in {1,4,8,16}, 2 seeds)
  M2  eval-time rank-k knee at K=8       (from the K=8 unconstrained rankk_curve)
  M3  train-time force-rank-k step at K  (K=8, k in {2,4,6,7,8}, 2 seeds)

Run smoke FIRST:  python run_task_d.py --smoke
Then:             python run_stage1.py --steps 2000 --out-dir results/stage1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_d import TaskDConfig
from run_task_d import make_model, train

D = 16
K_GATE = 8
UNCONSTRAINED_KS = (1, 4, 8, 16)
FORCE_RANKS = (2, 4, 6, 7, 8)
SEEDS = (0, 1)
TAU = 0.99  # knee test threshold


def one_run(model_kwargs, cfg, device, steps, seed, force_rank_k):
    torch.manual_seed(seed)                      # reproducible model init
    model = make_model("matrix", **model_kwargs).to(device)
    res = train(model, cfg, device, "matrix", steps=steps,
                force_rank_k=force_rank_k, seed=seed)
    res["seed"] = seed
    return res


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _spearman(xs, ys):
    """Spearman rank correlation (no scipy). Fine for the few distinct K values."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = (sum((rx[i] - mx) ** 2 for i in range(n))
           * sum((ry[i] - my) ** 2 for i in range(n))) ** 0.5
    return num / den if den > 0 else float("nan")


def aggregate(unconstrained, forced):
    """unconstrained: list of result dicts (force_rank_k None, various K).
    forced: list of result dicts (K=8, force_rank_k in FORCE_RANKS)."""
    frac = f"recovered_frac@{TAU}"
    report = {"tau": TAU, "K_gate": K_GATE, "d": D}

    # ---- M1: effective rank vs K (mean over seeds) ----
    m1 = {}
    for K in UNCONSTRAINED_KS:
        ers = [r["effective_rank_mean"] for r in unconstrained if r["K"] == K]
        srs = [r["stable_rank_mean"] for r in unconstrained if r["K"] == K]
        m1[K] = {"effective_rank": mean(ers), "stable_rank": mean(srs), "n": len(ers)}
    report["M1_effrank_vs_K"] = m1
    ks = sorted(m1)
    er_seq = [m1[k]["effective_rank"] for k in ks]
    # Pre-registered M1 CONFIRM (§6): Spearman(K, effrank) >= 0.8 AND
    # effrank in [0.7K, min(1.3K, d)] for each K <= d.
    m1_spearman = _spearman(ks, er_seq)
    m1_band_ok = all(0.7 * k <= m1[k]["effective_rank"] <= min(1.3 * k, D) + 1e-6
                     for k in ks if k <= D)
    m1_confirm = (m1_spearman == m1_spearman) and m1_spearman >= 0.8 and m1_band_ok
    report["M1_spearman"] = m1_spearman
    report["M1_band_ok"] = m1_band_ok

    # ---- M2: rank-k knee at K=8 (avg recovered_frac@tau over the K=8 unconstrained runs) ----
    curves = [r["rankk_curve"] for r in unconstrained if r["K"] == K_GATE and "rankk_curve" in r]
    m2_knee = None
    if curves:
        # rankk_curve keys are ints in-process (evaluate() builds {int(k): ...}).
        ks_curve = sorted(curves[0])
        avg = {k: mean([c[k][frac] for c in curves]) for k in ks_curve}
        report["M2_rankk_curve@K8"] = avg
        acc_d = avg[max(ks_curve)]
        for k in ks_curve:                        # knee = smallest k reaching 0.9*acc(d)
            if avg[k] >= 0.9 * acc_d:
                m2_knee = k
                break
        report["M2_knee"] = m2_knee
    m2_ok = (m2_knee is not None and K_GATE - 1 <= m2_knee <= K_GATE + 1)

    # ---- M3: force-rank-k step at K=8 (headline recovered_frac@tau, mean over seeds) ----
    m3 = {}
    for k in FORCE_RANKS:
        accs = [r[frac] for r in forced if r["force_rank_k"] == k]
        m3[k] = {"recovered_frac": mean(accs), "n": len(accs)}
    report["M3_forcerank_step"] = m3
    acc_atK = m3[K_GATE]["recovered_frac"]
    acc_belowK = max(m3[k]["recovered_frac"] for k in FORCE_RANKS if k < K_GATE)
    m3_step = (acc_atK >= 0.9 and acc_belowK <= acc_atK - 0.3)   # step at k=K
    m3_hard_falsify = acc_belowK >= 0.5                          # low-rank suffices -> shortcut

    # ---- preliminary verdict (advisory) ----
    if m3_hard_falsify:
        verdict = "FAIL (HARD FALSIFY: a rank<K model recovers K bindings -> shortcut/premise broken)"
    elif m1_confirm and m3_step:
        verdict = "PASS (M1 rank tracks K per §6; M3 shows a step at k=K)" + \
                  ("; M2 knee corroborates" if m2_ok else "; NOTE: M2 knee did not corroborate")
    elif not m1_confirm and not m3_step:
        verdict = "FAIL (M1 does not track K AND no M3 step -> gradient rank-averse even when required)"
    else:
        verdict = "INCONCLUSIVE (mixed signals -> diagnose before scaling)"
    report["m1_confirm"] = m1_confirm
    report["m2_ok"] = m2_ok
    report["m3_step"] = m3_step
    report["m3_hard_falsify"] = m3_hard_falsify
    report["PRELIMINARY_VERDICT"] = verdict
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-refine", type=int, default=1)
    ap.add_argument("--out-dir", type=str, default="results/stage1")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    mk = dict(d=D, h=args.h, n_layers=args.n_layers, n_heads=args.n_heads, n_refine=args.n_refine)
    print(f"Stage-1 gate: device={device}, steps={args.steps}, "
          f"{len(UNCONSTRAINED_KS)*len(SEEDS) + len(FORCE_RANKS)*len(SEEDS)} runs", flush=True)

    def _save(obj, name):
        with open(f"{args.out_dir}/{name}", "w") as f:
            json.dump(obj, f, indent=2)

    # try/except per run so one bad config doesn't kill the remaining sweep
    # (CLAUDE.md: "add try/except so one crash doesn't kill remaining configs").
    unconstrained, forced, failures = [], [], []
    for K in UNCONSTRAINED_KS:
        for seed in SEEDS:
            print(f"\n== M1/M2 unconstrained K={K} seed={seed} ==", flush=True)
            try:
                r = one_run(mk, TaskDConfig(d=D, K=K, orthogonal=True),
                            device, args.steps, seed, force_rank_k=None)
                unconstrained.append(r)
                _save(r, f"uncon_K{K}_s{seed}.json")
            except Exception as e:
                print(f"  FAILED uncon K={K} seed={seed}: {e!r}", flush=True)
                failures.append({"run": f"uncon_K{K}_s{seed}", "error": repr(e)})

    cfg8 = TaskDConfig(d=D, K=K_GATE, orthogonal=True)
    for k in FORCE_RANKS:
        for seed in SEEDS:
            print(f"\n== M3 force_rank_k={k} K={K_GATE} seed={seed} ==", flush=True)
            try:
                r = one_run(mk, cfg8, device, args.steps, seed, force_rank_k=k)
                forced.append(r)
                _save(r, f"force{k}_s{seed}.json")
            except Exception as e:
                print(f"  FAILED force{k} seed={seed}: {e!r}", flush=True)
                failures.append({"run": f"force{k}_s{seed}", "error": repr(e)})

    report = aggregate(unconstrained, forced)
    report["failures"] = failures
    _save(report, "AGGREGATE.json")
    with open(f"{args.out_dir}/SUMMARY.txt", "w") as f:
        f.write("Task D — Stage-1 minimal gate\n" + "=" * 40 + "\n")
        f.write(json.dumps(report, indent=2))
    if failures:
        print(f"\nWARNING: {len(failures)} run(s) failed — verdict is on the survivors.", flush=True)
    print("\n" + "=" * 60)
    print("PRELIMINARY VERDICT:", report["PRELIMINARY_VERDICT"])
    print("=" * 60)
    print(f"wrote {args.out_dir}/AGGREGATE.json + SUMMARY.txt")


if __name__ == "__main__":
    main()
