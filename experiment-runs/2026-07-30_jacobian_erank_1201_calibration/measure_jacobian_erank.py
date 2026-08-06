#!/usr/bin/env python3
"""
measure_jacobian_erank.py — Exp #0 (Jacobian erank), rank-recruitment-ws queue item.

Tests the "candidate refined mechanism" from the ICML 2026 MI Workshop paper
"The Gradient Does Not See Rank" (matrix-thinking/submissions/icml-mi-workshop-2026/,
PUBLISHED), section 5.3 ("Interpretation: a candidate refined mechanism").
Verbatim claim (PAPER_READER_VIEW.md, sections/05_positive_control.tex):

    "We advance a candidate refined hypothesis for future work: the trained
    readout's Jacobian at test inputs has an effectively rank-1 active
    subspace in Z, regardless of whether the readout is in-principle
    nonlinear. Concretely: for each trained positive-control checkpoint,
    compute J(Z) = d(phi)/d(vec(Z)) at Z from the test distribution and
    report the effective rank of the row-space of J averaged over test
    examples. The refined hypothesis predicts erank(J) ~= 1 for
    Bilinear+GELU, SVD-augmented, and Quadratic; a value >= 4 would
    falsify it. This measurement is a camera-ready experiment we have not
    yet run on these checkpoints."

Design note / discrepancy with the queue item (flagged for audit): the queue
item lists "4 nonlinear-in-Z readouts (flatten, bilinear+GELU, svd-aug,
quadratic)". Per the paper's own architecture section, flatten is phi(Z) =
W_down @ vec(Z), LINEAR in Z (constant Jacobian) -- not "nonlinear-in-Z" --
and the refined hypothesis's falsification bar (erank>=4 falsifies) is
stated by the paper ONLY for {bilinear_gelu, svd_aug, quadratic}. We still
measure and report flatten (it is one of the four readout variants the
paper trains, and it's a cheap, informative sanity control: since J is
provably Z-independent for flatten, the per-instance erank(J) should be
IDENTICAL across all captured Z instances -- this script asserts that as an
internal correctness check), but its number is reported as a LINEAR-BASELINE
CONTROL, not as a pass/fail leg of the falsifier.

phi is defined EXCLUDING the final LayerNorm (h_out = LayerNorm(phi(Z))) --
this matches the paper's architecture equations exactly, and is why flatten
is describable as "linear" at all (LayerNorm is not linear).

Z is captured as the model's own bottleneck.forward() produces it: post
w_up projection, post the optional multiplicative-thinking iteration,
pre-readout, pre-truncation (rank_project_k=None, force_rank_k=None) --
i.e. Z_out in the checkpoint's natural (unperturbed) inference regime, from
real held-out ProsQA test examples, matching "Z from the test distribution".

Usage:
  # Full measurement pass (requires all requested checkpoints to exist):
  python3 measure_jacobian_erank.py \
      --checkpoints-dir /home/nvidia/jacobian_erank_exp/results \
      --readouts flatten,bilinear_gelu,svd_aug,quadratic \
      --prosqa-val-path /home/nvidia/jacobian_erank_exp/data/prosqa_test.json \
      --n-examples 128 \
      --results-dir /home/nvidia/jacobian_erank_exp/results/jacobian_erank

  # Pure-math smoke test (no GPT-2, no data, no checkpoint needed):
  python3 measure_jacobian_erank.py --smoke-test
"""

import argparse
import copy
import functools
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Import the audited training/model code as a module (no side effects: main()
# only runs under `if __name__ == "__main__"`, which is not our __name__).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_matrix_codi as rmc  # noqa: E402


ALL_READOUTS = ["flatten", "bilinear", "bilinear_gelu", "svd_aug", "quadratic"]
# Readouts the paper's refined hypothesis actually predicts erank(J)~=1 for
# (nonlinear in Z; falsification bar erank>=4 applies to these per §5.3).
FALSIFIABLE_READOUTS = {"bilinear_gelu", "svd_aug", "quadratic"}
CONFIRM_BAR = 1.5   # queue item's stated confirm threshold (erank <= 1.5)
FALSIFY_BAR = 4.0   # paper's + queue item's stated falsify threshold (erank >= 4)


# =============================================================================
# phi: the readout function, EXCLUDING the bottleneck's final LayerNorm.
# Copied verbatim from MatrixBottleneck.forward's readout branch
# (matrix-thinking/scripts/run_matrix_codi.py lines ~341-361) so the
# Jacobian is taken of exactly the phi the paper's equations define
# (h_out = LayerNorm(phi(Z))), not of the LayerNorm-composed output.
# =============================================================================

def phi_readout(bottleneck, Z, readout):
    """phi(Z): (B, d, d) -> (B, D). Z must already be float32."""
    B = Z.shape[0]
    d = bottleneck.mat_dim
    if readout == "flatten":
        flat_out = Z.reshape(B, d * d)
        return bottleneck.w_down(flat_out)
    elif readout == "bilinear":
        MV = torch.einsum("bij,kj->bik", Z, bottleneck.V)
        probes = torch.einsum("ki,bik->bk", bottleneck.U, MV)
        return bottleneck.out(probes)
    elif readout == "bilinear_gelu":
        MV = torch.einsum("bij,kj->bik", Z, bottleneck.V)
        probes = torch.einsum("ki,bik->bk", bottleneck.U, MV)
        return bottleneck.out(F.gelu(probes))
    elif readout == "svd_aug":
        sigma = torch.linalg.svdvals(Z)  # (B, d); documented-stable backward
        flat_out = Z.reshape(B, d * d)
        return bottleneck.w_down(flat_out) + bottleneck.sigma_proj(sigma)
    elif readout == "quadratic":
        ZZt = torch.einsum("bij,bkj->bik", Z, Z)
        ZtZ = torch.einsum("bji,bjk->bik", Z, Z)
        quad = torch.cat([ZZt.reshape(B, d * d), ZtZ.reshape(B, d * d)], dim=-1)
        return bottleneck.w_down_quad(quad)
    else:
        raise ValueError(f"Unknown readout: {readout}")


def jacobian_single(bottleneck, Z_single, readout):
    """Autograd Jacobian d(phi)/d(vec(Z)) at one Z instance. Z_single: (d,d).

    Returns J of shape (D, d*d). Uses per-instance torch.autograd.functional
    .jacobian with vectorize=False (no vmap) so this is robust across torch
    versions regardless of whether every op used by every readout (in
    particular svdvals for svd_aug) has a registered vmap batching rule.
    Cheap regardless: D<=768, d*d=256, and the wrapped function only runs
    the small bottleneck readout, not GPT-2.
    """
    d = bottleneck.mat_dim

    def f(Zflat):
        Zm = Zflat.reshape(1, d, d)
        out = phi_readout(bottleneck, Zm, readout)
        return out.reshape(-1)

    Zflat = Z_single.reshape(-1).clone().detach().requires_grad_(True)
    J = torch.autograd.functional.jacobian(f, Zflat, vectorize=False)  # (D, d*d)
    return J.detach()


def erank_of_matrix(J):
    """Effective rank of a (possibly non-square) matrix via singular-value
    entropy -- reuses the repo's standard definition (run_matrix_codi.
    effective_rank), which is defined generically via torch.linalg.svdvals
    and works unchanged for the non-square (D, d*d) Jacobian.
    """
    return rmc.effective_rank(J.unsqueeze(0)).squeeze(0).item()


# =============================================================================
# Finite-difference cross-check (central differences over all d*d entries).
# =============================================================================

def jacobian_finite_diff(bottleneck, Z_single, readout, eps=1e-3):
    d = bottleneck.mat_dim
    D = bottleneck.hidden_dim
    Zflat = Z_single.reshape(-1).clone().detach()
    J_fd = torch.zeros(D, d * d, dtype=torch.float64)
    Zflat64 = Zflat.double()
    for i in range(d * d):
        e = torch.zeros_like(Zflat64)
        e[i] = eps
        Zp = (Zflat64 + e).float().reshape(1, d, d)
        Zm = (Zflat64 - e).float().reshape(1, d, d)
        with torch.no_grad():
            fp = phi_readout(bottleneck, Zp, readout).reshape(-1).double()
            fm = phi_readout(bottleneck, Zm, readout).reshape(-1).double()
        J_fd[:, i] = (fp - fm) / (2 * eps)
    return J_fd.float()


# =============================================================================
# Smoke test: pure math, no GPT-2 / data / checkpoint dependency.
# =============================================================================

def run_smoke_test():
    print("=" * 70)
    print("  measure_jacobian_erank.py SMOKE TEST (pure math, no GPT-2/data)")
    print("=" * 70)
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device = {device}")

    hidden_dim, mat_dim = 768, 16
    B = 4

    for readout in ALL_READOUTS:
        print(f"\n[{readout}]")
        bn = rmc.MatrixBottleneck(
            hidden_dim=hidden_dim, mat_dim=mat_dim,
            use_thinking_iter=True, readout=readout,
        ).to(device)
        bn.eval()

        Z_batch = torch.randn(B, mat_dim, mat_dim, device=device)

        # 1. Jacobian shape + finiteness.
        Js = []
        for i in range(B):
            J = jacobian_single(bn, Z_batch[i], readout)
            assert J.shape == (hidden_dim, mat_dim * mat_dim), f"bad J shape {J.shape}"
            assert torch.isfinite(J).all(), "non-finite entries in J"
            Js.append(J)
        ers = [erank_of_matrix(J) for J in Js]
        assert all(1.0 - 1e-3 <= e <= mat_dim * mat_dim + 1e-3 for e in ers), \
            f"erank out of range: {ers}"
        print(f"  per-instance erank(J): {[f'{e:.4f}' for e in ers]}")

        # 2. flatten/bilinear are LINEAR in Z -> J must be Z-independent.
        if readout in ("flatten", "bilinear"):
            j0, j1 = Js[0], Js[1]
            max_abs_diff = (j0 - j1).abs().max().item()
            assert max_abs_diff < 1e-3, (
                f"{readout} is supposed to be linear in Z (constant Jacobian) "
                f"but J differs across two random Z draws by {max_abs_diff}"
            )
            print(f"  linearity check: max|J(Z0)-J(Z1)| = {max_abs_diff:.2e} (OK, ~0)")

        # 3. Finite-difference cross-check on 2 instances.
        max_rel_err = 0.0
        for i in range(2):
            J_auto = Js[i].cpu()
            J_fd = jacobian_finite_diff(bn.cpu(), Z_batch[i].cpu(), readout).cpu()
            bn.to(device)
            frob_auto = J_auto.norm().item()
            frob_diff = (J_auto - J_fd).norm().item()
            rel = frob_diff / max(frob_auto, 1e-8)
            max_rel_err = max(max_rel_err, rel)
        assert max_rel_err < 5e-2, f"finite-diff cross-check failed: rel_err={max_rel_err}"
        print(f"  finite-diff cross-check: max relative Frobenius error = {max_rel_err:.2e} (OK)")

    print("\n" + "=" * 70)
    print("  ALL SMOKE CHECKS PASSED")
    print("=" * 70)


# =============================================================================
# Full measurement pass (real GPT-2 checkpoints, real ProsQA test examples).
# =============================================================================

def load_checkpoint(ckpt_path, device):
    assert Path(ckpt_path).exists(), f"Checkpoint not found: {ckpt_path}"
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved_cfg = sd.get("cfg", {})
    use_matrix = sd.get("use_matrix_bottleneck", True)
    assert use_matrix, f"{ckpt_path} is not a matrix-CODI checkpoint"

    tokenizer = rmc.GPT2TokenizerFast.from_pretrained(saved_cfg["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    rmc.assert_colon_tokenization(tokenizer)

    model = rmc.CodiModel(
        base_model_name=saved_cfg["base_model"],
        n_latents=saved_cfg["n_latents"],
        use_matrix_bottleneck=True,
        mat_dim=saved_cfg["mat_dim"],
        use_thinking_iter=saved_cfg.get("use_thinking_iter", True),
        readout=saved_cfg.get("readout", "flatten"),
    )
    model.add_special_tokens(tokenizer)
    model.load_state_dict(sd["model"])
    model.to(device)
    model.eval()
    return model, tokenizer, saved_cfg


def collect_Z(model, tokenizer, saved_cfg, prosqa_val_path, n_examples, device, batch_size=16):
    """Run the real student_forward pass on the first n_examples ProsQA test
    problems (deterministic order, no shuffling -- same discipline as Run C's
    rank-projection ablation, which is where the paper's "128 ProsQA test
    problems" / "standard CODI eval split" figure comes from: Run C caps
    eval to cfg['max_eval_batches'] * cfg['eval_batch_size'] = 8*16 = 128 by
    default). rank_project_k=None, force_rank_k=None -- Z is the model's own
    natural (untruncated) inference-time observable.

    Returns Z_all: (n_examples, n_latents, d, d) float32, and the number of
    examples actually collected (<= n_examples if the dataset is smaller).
    """
    cfg = copy.deepcopy(saved_cfg)
    cfg["prosqa_val_path"] = prosqa_val_path
    val_ds = rmc.ProsQADataset("val", tokenizer, cfg, model.special_token_ids)
    n_take = min(n_examples, len(val_ds))

    pad_id = tokenizer.pad_token_id
    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=functools.partial(rmc.collate_gsm8k, pad_id=pad_id, n_latents=cfg["n_latents"]),
        drop_last=False,
    )

    Z_chunks = []
    collected = 0
    with torch.no_grad():
        for batch in loader:
            if collected >= n_take:
                break
            q_ids = batch["q_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            tail_ids = batch["tail_ids"].to(device)
            tail_mask = batch["tail_mask"].to(device)
            out = model.student_forward(
                q_ids, q_mask, tail_ids, tail_mask,
                save_Z=True, rank_project_k=None, force_rank_k=None,
            )
            # Z_list: n_latents entries, each (B, d, d) float32, detached.
            Z_step = torch.stack(out["Z_list"], dim=1)  # (B, n_latents, d, d)
            take = min(Z_step.shape[0], n_take - collected)
            Z_chunks.append(Z_step[:take].cpu())
            collected += take

    Z_all = torch.cat(Z_chunks, dim=0)  # (n_take, n_latents, d, d)
    return Z_all, n_take


def measure_readout(model, saved_cfg, readout, Z_all, finite_diff_n, checkpoint_path):
    bottleneck = model.bottleneck
    device = next(bottleneck.parameters()).device
    n_examples, n_latents, d, _ = Z_all.shape

    per_instance_erank = torch.zeros(n_examples, n_latents)
    t0 = time.time()
    for ex in range(n_examples):
        for pos in range(n_latents):
            Zi = Z_all[ex, pos].to(device).float()
            J = jacobian_single(bottleneck, Zi, readout)
            per_instance_erank[ex, pos] = erank_of_matrix(J)
    elapsed = time.time() - t0

    flat = per_instance_erank.reshape(-1)
    mean_er = flat.mean().item()
    std_er = flat.std().item()
    by_position = per_instance_erank.mean(dim=0).tolist()  # n_latents

    # Internal consistency check for provably-linear readouts: J must be
    # Z-independent, so erank(J) must be (numerically) constant across all
    # captured instances.
    linearity_note = None
    if readout in ("flatten", "bilinear"):
        spread = (flat.max() - flat.min()).item()
        linearity_note = {
            "expected": "erank(J) constant across all instances (J is Z-independent)",
            "observed_spread_max_minus_min": spread,
            "consistent": bool(spread < 1e-2),
        }

    # Finite-difference cross-check on a small subsample (first finite_diff_n
    # (example, position) instances in row-major order).
    fd_checks = []
    flat_idx = 0
    for ex in range(n_examples):
        for pos in range(n_latents):
            if flat_idx >= finite_diff_n:
                break
            Zi = Z_all[ex, pos].to(device).float()
            J_auto = jacobian_single(bottleneck, Zi, readout)
            J_fd = jacobian_finite_diff(bottleneck, Zi, readout).to(device)
            frob_auto = J_auto.norm().item()
            frob_diff = (J_auto - J_fd).norm().item()
            rel_frob = frob_diff / max(frob_auto, 1e-8)
            er_auto = erank_of_matrix(J_auto)
            er_fd = erank_of_matrix(J_fd)
            fd_checks.append({
                "example_idx": ex, "latent_position": pos,
                "erank_autograd": er_auto, "erank_finite_diff": er_fd,
                "erank_abs_diff": abs(er_auto - er_fd),
                "jacobian_relative_frobenius_error": rel_frob,
            })
            flat_idx += 1
        if flat_idx >= finite_diff_n:
            break

    max_fd_erank_err = max((c["erank_abs_diff"] for c in fd_checks), default=None)
    max_fd_frob_err = max((c["jacobian_relative_frobenius_error"] for c in fd_checks), default=None)

    if readout in FALSIFIABLE_READOUTS:
        if mean_er <= CONFIRM_BAR:
            verdict = f"CONFIRMED (erank<={CONFIRM_BAR})"
        elif mean_er >= FALSIFY_BAR:
            verdict = f"FALSIFIED vs stated bar (erank>={FALSIFY_BAR}) -- interpret ONLY against the flatten linear-baseline erank in the same run (G3-audit R4: erank(J) does not discriminate rank-blind from rank-aware; see EXPERIMENT_LOG 2026-07-30)"
        else:
            verdict = f"INCONCLUSIVE ({CONFIRM_BAR}<erank<{FALSIFY_BAR})"
    else:
        verdict = "NOT GATED BY THE FALSIFIER (linear-in-Z control readout; paper's SS5.3 " \
                  "erank~=1 prediction and erank>=4 falsifier apply only to " \
                  "{bilinear_gelu, svd_aug, quadratic})"

    return {
        "readout": readout,
        "checkpoint": str(checkpoint_path),
        "n_examples": n_examples,
        "n_latent_positions": n_latents,
        "jacobian_erank_mean": mean_er,
        "jacobian_erank_std": std_er,
        "jacobian_erank_min": flat.min().item(),
        "jacobian_erank_max": flat.max().item(),
        "jacobian_erank_by_latent_position": by_position,
        "linearity_selfcheck": linearity_note,
        "finite_diff_crosscheck": {
            "n_checked": len(fd_checks),
            "max_erank_abs_error": max_fd_erank_err,
            "max_jacobian_relative_frobenius_error": max_fd_frob_err,
            "per_instance": fd_checks,
        },
        "gated_by_paper_falsifier": readout in FALSIFIABLE_READOUTS,
        "verdict": verdict,
        "measure_wall_time_sec": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description="Measure erank(J(Z)) for matrix-CODI positive-control readouts.")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--checkpoints-dir", type=str, default=None,
                     help="Directory containing <readout>/best_run_b_matrix.pt for each readout.")
    ap.add_argument("--readouts", type=str, default="flatten,bilinear_gelu,svd_aug,quadratic",
                     help="Comma-separated readout names to measure.")
    ap.add_argument("--prosqa-val-path", type=str, default=None,
                     help="Path to prosqa_test.json (the CODI eval split).")
    ap.add_argument("--n-examples", type=int, default=128)
    ap.add_argument("--finite-diff-n", type=int, default=5,
                     help="Number of (example,position) instances to finite-diff cross-check, PER READOUT.")
    ap.add_argument("--results-dir", type=str, default=None)
    ap.add_argument("--strict", action="store_true", default=True,
                     help="Fail hard if any requested readout's checkpoint is missing (default True).")
    ap.add_argument("--no-strict", dest="strict", action="store_false")
    args = ap.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    assert args.checkpoints_dir is not None, "--checkpoints-dir is required (unless --smoke-test)"
    assert args.prosqa_val_path is not None, "--prosqa-val-path is required (unless --smoke-test)"
    assert args.results_dir is not None, "--results-dir is required (unless --smoke-test)"

    readouts = [r.strip() for r in args.readouts.split(",") if r.strip()]
    for r in readouts:
        assert r in ALL_READOUTS, f"Unknown readout {r!r}, must be one of {ALL_READOUTS}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_paths = {r: ckpt_dir / r / "best_run_b_matrix.pt" for r in readouts}
    missing = {r: p for r, p in ckpt_paths.items() if not p.exists()}
    if missing:
        msg = "Missing checkpoints:\n" + "\n".join(f"  {r}: {p}" for r, p in missing.items())
        if args.strict:
            raise FileNotFoundError(msg + "\n(pass --no-strict to measure only the readouts that exist)")
        else:
            print(f"WARNING: {msg}\nContinuing with available readouts only.", file=sys.stderr)
            readouts = [r for r in readouts if r not in missing]

    all_results = {}
    for readout in readouts:
        print(f"\n{'=' * 70}\n  READOUT: {readout}\n{'=' * 70}")
        model, tokenizer, saved_cfg = load_checkpoint(ckpt_paths[readout], device)
        assert saved_cfg.get("readout", "flatten") == readout, (
            f"checkpoint's own saved readout {saved_cfg.get('readout')!r} != requested {readout!r}"
        )
        Z_all, n_take = collect_Z(
            model, tokenizer, saved_cfg, args.prosqa_val_path, args.n_examples, device,
        )
        print(f"  Collected Z for {n_take} examples x {Z_all.shape[1]} latent positions "
              f"= {n_take * Z_all.shape[1]} Jacobian instances")
        res = measure_readout(model, saved_cfg, readout, Z_all, args.finite_diff_n, ckpt_paths[readout])
        print(f"  jacobian_erank_mean = {res['jacobian_erank_mean']:.4f} "
              f"(std {res['jacobian_erank_std']:.4f})  -> {res['verdict']}")
        all_results[readout] = res
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    overall_status = "COMPLETED"
    out = {
        "status": overall_status,
        "n_examples_requested": args.n_examples,
        "readouts": readouts,
        "confirm_bar_erank_leq": CONFIRM_BAR,
        "falsify_bar_erank_geq": FALSIFY_BAR,
        "falsifiable_readouts": sorted(FALSIFIABLE_READOUTS),
        "results": all_results,
        "metric": "jacobian_erank_mean",
        "jacobian_erank_mean_by_readout": {r: v["jacobian_erank_mean"] for r, v in all_results.items()},
        "any_falsified": any(
            v["gated_by_paper_falsifier"] and v["jacobian_erank_mean"] >= FALSIFY_BAR
            for v in all_results.values()
        ),
    }
    out_path = results_dir / "jacobian_erank_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    summary_lines = [
        "=" * 70,
        "  JACOBIAN ERANK MEASUREMENT -- SS5.3 candidate refined mechanism",
        "=" * 70,
    ]
    for r, v in all_results.items():
        summary_lines.append(
            f"  {r:14s} erank(J)_mean = {v['jacobian_erank_mean']:.4f}  "
            f"(std {v['jacobian_erank_std']:.4f}, n={v['n_examples']*v['n_latent_positions']})  "
            f"-> {v['verdict']}"
        )
    summary_lines += ["=" * 70, f"  any_falsified = {out['any_falsified']}", "=" * 70, ""]
    summary = "\n".join(summary_lines)
    print(summary)
    with open(results_dir / "SUMMARY.txt", "a") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
