"""
compB drift/mechanism analysis -- EXPERIMENT_LOG.md "2026-08-18 #8" pre-registration.

Three legs, pre-registered bands (see EXPERIMENT_LOG.md #8, committed d9b230f):
  (a) geometry (TPC/o_pc vs retrieval24_acc@h=61) -- NOT BLIND, confirmatory only.
  (b) adapter conditioning (cond(entity_adapter) vs retrieval24_acc@h=61) -- BLIND.
        rho <= -0.5 and p<0.05 => SUPPORTS; |rho|<0.3 => NULL; else PARTIAL.
  (c) drift from seeded init (||W_final-W_init||_F/||W_init||_F vs retrieval24_acc@h=61) -- BLIND.
        same bands as (b).

Metric: retrieval24_acc from the archived P1b (teacher_force=true) records at
h=61, cells with ckpt_step == 20000 ONLY.

Init reconstruction (leg c) is NEW COMPUTATION, not archive retrieval: only
FINAL (step-20000) checkpoints exist on disk, so W_init is rebuilt by seeded
re-instantiation of the adapter module alone, via the EXACT construction
sequence the training runner used (torch.manual_seed(seed) -> build_backbone
-> build_ncr_head -> NCRIntegration(...)) so the entity_adapter's Linear
layer draws from the correct point in the RNG stream (its init consumes RNG
AFTER the ~98M-param backbone and the ncr_head have already drawn their own
weights under the same global torch RNG). No training; CPU only
(CUDA_VISIBLE_DEVICES="") -- no GPU beyond nothing (not even loading needs a
GPU here; checkpoints are also loaded map_location="cpu").
"""
import glob
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only -- no GPU needed for this analysis

sys.path.insert(0, "/home/nvidia/ncr_g3b31_contrastive")
os.chdir("/home/nvidia/ncr_g3b31_contrastive")

import numpy as np
import torch
from scipy import stats

import ncr_lm_wave1_runner as R  # noqa: E402 -- pinned runner, md5 9a93198b642242f512ff8489e32b0a53

RESULTS_DIR = "/home/nvidia/ncr_writecond/results"
VOCAB_SIZE_TOTAL = 50259  # from ckpt integ_config, matches build_grammar_pools_and_cfg's pool for K=24
CKPT_CANDIDATE_DIRS = [
    "/ephemeral/reseed_ckpts/mob_g3b31_compB_s{seed}_ckpts/mob_g3b31_compB_s{seed}.ckpt.pt",
    "/home/nvidia/ncr_g3b31_contrastive/results/mob_g3b31_compB_s{seed}_ckpts/mob_g3b31_compB_s{seed}.ckpt.pt",
]
OUT_DIR = "/home/nvidia/ncr_writecond/analysis"
HOPS = ["h=1", "h=13", "h=37", "h=61"]

N_PERM = 200_000  # Monte Carlo permutation resamples (full n! enumeration is combinatorially
                   # infeasible at n~18: 18! ~ 6.4e15 -- see report for full disclosure)


# ---------------------------------------------------------------------------
# 1. Load archived per-seed scores (leg a is zero-cost archive retrieval;
#    the target metric for legs b/c also comes from here).
# ---------------------------------------------------------------------------
def load_seed_rows():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "writecond_premise_REPL_compB_s*.json")))
    rows = []
    excluded = []
    for f in files:
        d = json.load(open(f))
        seed = int(os.path.basename(f).split("_s")[-1].split(".")[0])
        if d.get("ckpt_step") != 20000:
            excluded.append((seed, d.get("ckpt_step")))
            continue
        p1b = d["P1b"]["result"]
        assert d["P1b"]["teacher_force"] is True
        row = {"seed": seed, "file": f, "ckpt_recorded": d["ckpt"]}
        row["retrieval24_acc_h61"] = p1b["h=61"]["retrieval24_acc"]
        for h in HOPS:
            row[f"tpc_{h}"] = p1b[h]["target_pairwise_cos"]
            row[f"opc_{h}"] = p1b[h]["o_pairwise_cos"]
        rows.append(row)
    rows.sort(key=lambda r: r["seed"])
    return rows, excluded


def resolve_ckpt_path(seed):
    for tmpl in CKPT_CANDIDATE_DIRS:
        p = tmpl.format(seed=seed)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no checkpoint found for compB seed {seed} in any candidate dir")


# ---------------------------------------------------------------------------
# 2. Leg (b)+(c) per-seed computation: load final ckpt, extract W_final;
#    reconstruct W_init via seeded re-instantiation; compute cond + drift.
# ---------------------------------------------------------------------------
def matrix_condition_number(W: torch.Tensor) -> float:
    # W is (d_ncr=25, d_model=768), rectangular. The standard 2-norm condition
    # number generalizes via singular values: sigma_max / sigma_min.
    s = torch.linalg.svdvals(W.double())
    return (s[0] / s[-1]).item()


def compute_adapter_stats(seed: int):
    ckpt_path = resolve_ckpt_path(seed)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert ckpt["seed"] == seed, f"seed mismatch: file says {seed}, ckpt records {ckpt['seed']}"
    assert ckpt["step"] == 20000, f"expected step 20000, got {ckpt['step']}"
    assert ckpt["freeze_entity_adapter"] is False, "compB must be the TRAINABLE (unfrozen) adapter arm"
    W_final = ckpt["full_graft"]["integ_state"]["entity_adapter.weight"].detach().clone().float()
    integ_cfg = ckpt["full_graft"]["integ_config"]
    del ckpt

    # Reconstruct init: EXACT same construction sequence as build_two_arms/build_arm
    # (torch.manual_seed(seed) then backbone, then ncr_head, then NCRIntegration --
    # order matters because nn.Module weight init consumes the global RNG stream).
    arm = R.build_arm(integ_cfg["vocab_size"], seed, "cpu")
    W_init = arm["integ"].entity_adapter.weight.detach().clone().float()
    assert W_init.shape == W_final.shape == (integ_cfg["d_ncr"], integ_cfg["d_model"])
    del arm

    cond_final = matrix_condition_number(W_final)
    drift_num = (W_final - W_init).norm().item()
    drift_den = W_init.norm().item()
    drift_ratio = drift_num / drift_den

    return {
        "seed": seed,
        "ckpt_path": ckpt_path,
        "cond_entity_adapter": cond_final,
        "w_init_fro_norm": drift_den,
        "w_final_fro_norm": W_final.norm().item(),
        "drift_fro_norm": drift_num,
        "drift_ratio": drift_ratio,
    }


def verify_reconstruction_determinism(seed: int):
    """Sanity gate required by the task brief: re-instantiating with the same
    seed twice must give bit-identical weights, and the reconstructed init
    must NOT accidentally equal the final weights."""
    arm1 = R.build_arm(VOCAB_SIZE_TOTAL, seed, "cpu")
    w1 = arm1["integ"].entity_adapter.weight.detach().clone()
    arm2 = R.build_arm(VOCAB_SIZE_TOTAL, seed, "cpu")
    w2 = arm2["integ"].entity_adapter.weight.detach().clone()
    det_ok = torch.equal(w1, w2)

    ckpt_path = resolve_ckpt_path(seed)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    w_final = ckpt["full_graft"]["integ_state"]["entity_adapter.weight"]
    not_equal_to_final = not torch.equal(w1, w_final)
    return det_ok, not_equal_to_final


# ---------------------------------------------------------------------------
# 3. Stats: Spearman rho + scipy asymptotic p + Monte Carlo permutation p.
# ---------------------------------------------------------------------------
def spearman_with_perm_p(x, y, n_resamples=N_PERM, seed=12345):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho_asym, p_asym = stats.spearmanr(x, y)

    def stat(a, b):
        r, _ = stats.spearmanr(a, b)
        return r

    res = stats.permutation_test(
        (x, y), stat, permutation_type="pairings",
        n_resamples=n_resamples, alternative="two-sided",
        rng=np.random.default_rng(seed),
    )
    return {
        "n": len(x),
        "rho": float(rho_asym),
        "p_asymptotic_t_approx": float(p_asym),
        "p_permutation_mc": float(res.pvalue),
        "n_resamples": n_resamples,
    }


def verdict(rho, p):
    if rho <= -0.5 and p < 0.05:
        return "SUPPORTS"
    if abs(rho) < 0.3:
        return "NULL"
    return "PARTIAL"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    rows, excluded = load_seed_rows()
    print(f"Loaded {len(rows)} compB seeds at ckpt_step==20000: {[r['seed'] for r in rows]}")
    print(f"Excluded (ckpt_step != 20000): {excluded}")

    # Sanity gate: determinism + non-collapse-to-final, run on 3 spot-check seeds.
    print("\n--- reconstruction sanity checks ---")
    sanity = {}
    for s in [rows[0]["seed"], rows[len(rows) // 2]["seed"], rows[-1]["seed"]]:
        det_ok, not_final = verify_reconstruction_determinism(s)
        sanity[s] = {"deterministic": det_ok, "differs_from_final": not_final}
        print(f"seed {s}: deterministic={det_ok} differs_from_final_ckpt={not_final}")
        assert det_ok, f"reconstruction NOT deterministic for seed {s}"
        assert not_final, f"reconstructed init ACCIDENTALLY equals final weights for seed {s}"

    # Leg (b)/(c) per-seed adapter stats.
    print("\n--- computing per-seed adapter cond + drift (this loads 18x ~2.3GB checkpoints) ---")
    adapter_rows = {}
    for r in rows:
        s = r["seed"]
        stats_row = compute_adapter_stats(s)
        adapter_rows[s] = stats_row
        print(f"seed {s}: cond={stats_row['cond_entity_adapter']:.4f} "
              f"drift_ratio={stats_row['drift_ratio']:.6f} "
              f"retrieval24_acc@h61={r['retrieval24_acc_h61']:.4f}")

    # Merge into one table.
    table = []
    for r in rows:
        s = r["seed"]
        a = adapter_rows[s]
        merged = dict(r)
        merged.update({
            "cond_entity_adapter": a["cond_entity_adapter"],
            "drift_ratio": a["drift_ratio"],
            "w_init_fro_norm": a["w_init_fro_norm"],
            "w_final_fro_norm": a["w_final_fro_norm"],
            "ckpt_path_used": a["ckpt_path"],
        })
        table.append(merged)

    acc = [r["retrieval24_acc_h61"] for r in table]

    results = {"n": len(table), "seeds": [r["seed"] for r in table], "excluded_ckpt_step": excluded}

    # Leg (a) -- NOT BLIND, confirmatory only, TPC/o_pc at every hop.
    leg_a = {}
    for h in HOPS:
        tpc = [r[f"tpc_{h}"] for r in table]
        opc = [r[f"opc_{h}"] for r in table]
        leg_a[h] = {
            "target_pairwise_cos": spearman_with_perm_p(tpc, acc),
            "o_pairwise_cos": spearman_with_perm_p(opc, acc),
        }
    results["leg_a_geometry_NOT_BLIND"] = leg_a

    # Leg (b) -- BLIND, adapter conditioning.
    cond = [r["cond_entity_adapter"] for r in table]
    leg_b_stats = spearman_with_perm_p(cond, acc)
    leg_b_stats["verdict"] = verdict(leg_b_stats["rho"], leg_b_stats["p_permutation_mc"])
    results["leg_b_conditioning_BLIND"] = leg_b_stats

    # Leg (c) -- BLIND, drift from seeded init.
    drift = [r["drift_ratio"] for r in table]
    leg_c_stats = spearman_with_perm_p(drift, acc)
    leg_c_stats["verdict"] = verdict(leg_c_stats["rho"], leg_c_stats["p_permutation_mc"])
    results["leg_c_drift_BLIND"] = leg_c_stats

    results["sanity_checks"] = sanity
    results["table"] = table

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "compb_drift_analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print("\n=== LEG (b) conditioning: BLIND ===")
    print(leg_b_stats)
    print("\n=== LEG (c) drift: BLIND ===")
    print(leg_c_stats)
    print("\n=== LEG (a) geometry @ h=61 (NOT BLIND): ===")
    print(leg_a["h=61"])


if __name__ == "__main__":
    main()
