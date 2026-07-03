"""Task E runner -- train / eval / smoke for the Chapter 2 reasoning-transfer
gate (compositional multi-hop relational recall). See
NEXT_EXPERIMENT_DESIGN.md.

Measurements (design doc section 6):
  M1_E  learned effective rank vs K under multi-hop training (sanity re-confirmation of Task D's M1)
  M2_E  in-distribution multi-hop accuracy (h <= H_train)
  M3_E  compositional generalization at held-out hops (PRIMARY; vs chance, C_MLP floor, C7 ceiling)
  M4_E  causal rank-composition link (force-rank-k, measured at both M2_E and M3_E hops)

Success metric: recovered_frac@0.9 (Task D's re-registered trained-model
threshold; TASK_D_PREREGISTRATION.md section 6 calibration note) is the
PRIMARY decision metric here too. Reported alongside tau in {0.9, 0.95, 0.99}
plus the full cosine distribution, and alongside the C7 idealized-Z ceiling
computed on the SAME batches (so gap-to-ideal is a number, not a claim).

GATE DEFAULT is --orthogonal, matching Task D's audit finding and the C7
exactness requirement (ideal_Z is only exactly correct under orthonormal keys).

C2 (vector-state control) is explicitly OUT OF SCOPE for this build, per
NEXT_EXPERIMENT_DESIGN.md section 5: "a vector state has no natural 'apply the
same operator again' primitive without a bespoke learned hop-MLP" -- it is a
parallel, not-gating Stage-1b-style track with its own design, same precedent
Task D set for its own C2. --model mlp runs the C_MLP shortcut-baseline
control instead, which IS in scope (§5, §6).

Run the smoke gate FIRST on the cluster (seconds): python run_task_e.py --smoke
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
import task_d as td
import task_e as te
import eigen_utils as eu
from model_e import MatrixCompositionModel, MLPShortcutModel
from rank_utils import truncate_to_rank, effective_rank, stable_rank
from eigen_utils import eigenvalue_fidelity

TAUS = (0.9, 0.95, 0.99)


def make_model(model_type, d, h, n_layers, n_heads, n_refine, h_train_max, h_train=None):
    if model_type == "matrix":
        return MatrixCompositionModel(d, h, n_layers, n_heads, n_refine)
    if model_type == "mlp":
        # Pass the ACTUAL H_train set (not just its max) so MLPShortcutModel's
        # one-hot(h) uses exact set membership, not a range check (MINOR fix,
        # gauntlet/AUDIT_task_e_correctness.md item 2).
        return MLPShortcutModel(d, h, n_layers, n_heads, n_refine,
                                h_train_max=h_train_max, h_train=h_train)
    raise ValueError(f"unknown model_type {model_type}")


def cosine_loss(pred, target):
    """1 - mean cosine similarity; directly optimizes the (direction-only)
    success metric (same as Task D)."""
    return (1.0 - td.recovery_cosine(pred, target)).mean()


def _recovery_stats(cos, prefix=""):
    d = {
        f"{prefix}mean_cos": cos.mean().item(),
        f"{prefix}cos_p10": torch.quantile(cos, 0.10).item(),
        f"{prefix}cos_p50": torch.quantile(cos, 0.50).item(),
        f"{prefix}cos_p90": torch.quantile(cos, 0.90).item(),
    }
    for tau in TAUS:
        d[f"{prefix}recovered_frac@{tau}"] = (cos > tau).float().mean().item()
    return d


@torch.no_grad()
def evaluate_at_hops(model, cfg, gen, device, model_type, hops_to_eval,
                     force_rank_k=None, n_batches=8, batch_size=256,
                     compute_eig=False, eig_k=None):
    """Evaluate compositional recovery SEPARATELY at each hop depth in
    `hops_to_eval` (call with cfg.H_train for M2_E, with H_test+H_extra for
    M3_E). Reports the model's recovery stats AND (on the SAME batches) the C7
    idealized-Z ceiling and (optionally, C8) eigenstructure fidelity, so
    gap-to-ideal is a measured number, not a qualitative claim (design doc
    section 3, section 6)."""
    model.eval()
    per_hop = {}
    for h in hops_to_eval:
        cos_all, cos_ideal_all, eig_all, er_all, sr_all = [], [], [], [], []
        for _ in range(n_batches):
            b = te.sample_batch(cfg, batch_size, gen, hop_set=(h,), device=device)
            pred, Z = model(b, force_rank_k=force_rank_k)
            cos_all.append(td.recovery_cosine(pred, b["targets"]))

            # C7 ceiling: the SAME literal-iteration readout applied to the
            # analytic Z_ideal, on the SAME queries/targets.
            pred_ideal = MatrixCompositionModel.compose(b["z_ideal"], b["query_keys"], b["hops"])
            cos_ideal_all.append(td.recovery_cosine(pred_ideal, b["targets"]))

            if model_type == "matrix":
                er_all.append(effective_rank(Z))
                sr_all.append(stable_rank(Z))
                if compute_eig:
                    k = eig_k if eig_k is not None else cfg.K
                    eig_all.append(eigenvalue_fidelity(Z, b["z_ideal"], k))

        cos = torch.cat([c.reshape(-1) for c in cos_all])
        cos_ideal = torch.cat([c.reshape(-1) for c in cos_ideal_all])
        entry = {"h": h}
        # Stratify by EFFECTIVE composition distance, not just raw nominal h
        # (gauntlet/AUDIT_task_e_validity.md Finding B, fix item 3). The
        # permutation variant's pi is now a single Hamiltonian K-cycle
        # (task_e.py::_permutation_graph), so pi^h is periodic ONLY with
        # period K -- every entity shares the same cycle, so the effective
        # distance from any in-distribution hop is simply h % K, uniformly
        # across the whole hop group (no per-query heterogeneity to stratify
        # away, unlike the old general-permutation generator). The chain
        # variant is acyclic (no periodicity), so effective_hop == h there.
        entry["effective_hop"] = (h % cfg.K) if cfg.variant == "permutation" else h
        entry.update(_recovery_stats(cos))
        entry.update(_recovery_stats(cos_ideal, prefix="ideal_"))
        if er_all:
            entry["effective_rank_mean"] = torch.cat(er_all).mean().item()
            entry["stable_rank_mean"] = torch.cat(sr_all).mean().item()
        if eig_all:
            entry["eig_fidelity_mean"] = torch.cat(eig_all).mean().item()
        per_hop[h] = entry
    return per_hop


def train(model, cfg, device, model_type, steps=6000, batch_size=256, lr=3e-4,
          force_rank_k=None, seed=0, log_every=500, compute_eig=True):
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n_skipped = 0
    for step in range(1, steps + 1):
        # C6 injectivity is structural (identical for every batch of this cfg),
        # so assert it on step 1 only -- avoids a per-step GPU->CPU sync barrier
        # over ~20-30K steps. The smoke gate checks it exhaustively besides.
        b = te.sample_batch(cfg, batch_size, gen, hop_set=cfg.H_train, device=device,
                            assert_injective=(step == 1))
        pred, _ = model(b, force_rank_k=force_rank_k)
        loss = cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        # Gradient hygiene (reused verbatim from Task D): SVD/eigh backward
        # through rank-k truncation can occasionally produce a non-finite grad
        # on an unlucky batch. SKIP that step (don't kill the run), clip
        # otherwise, and report n_skipped.
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1
        if step % log_every == 0 or step == 1:
            extra = f"  [skipped {n_skipped} non-finite]" if n_skipped else ""
            print(f"  step {step:5d}  cosine_loss {loss.item():.4f}{extra}", flush=True)

    eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    m2 = evaluate_at_hops(model, cfg, eval_gen, device, model_type,
                          hops_to_eval=cfg.H_train, force_rank_k=force_rank_k,
                          compute_eig=(compute_eig and model_type == "matrix"))
    m3 = evaluate_at_hops(model, cfg, eval_gen, device, model_type,
                          hops_to_eval=(*cfg.H_test, *cfg.H_extra),
                          force_rank_k=force_rank_k,
                          compute_eig=(compute_eig and model_type == "matrix"))
    return {"n_skipped_steps": n_skipped, "M2_in_distribution": m2, "M3_held_out": m3}


# ---------------------------------------------------------------------------
# Smoke gate (run FIRST on cluster; no real training)
# ---------------------------------------------------------------------------

def smoke(device):
    print("=" * 60 + "\n  TASK E SMOKE GATE\n" + "=" * 60)

    print("\n[1] generator self-test (permutation + chain-into-sink variants, injectivity guard)")
    te._self_test()

    print("\n[2] eigen_utils self-test (Hungarian matcher vs brute force + roots-of-unity check)")
    eu._self_test()

    print("\n[3] rank_utils sanity + degenerate-spectrum NaN/Inf-free backward (reused from Task D)")
    Z = torch.randn(4, 16, 16, device=device)
    for k in (1, 2, 4, 8):
        er = effective_rank(truncate_to_rank(Z, k)).mean().item()
        assert er <= k + 1e-2, f"truncate_to_rank({k}) gave eff rank {er:.3f} > {k}"
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(4, 4, device=device))
    V, _ = torch.linalg.qr(torch.randn(4, 4, device=device))
    Zdeg = (U @ torch.diag(torch.tensor([3., 1., 1., 0.], device=device)) @ V.T
            ).unsqueeze(0).requires_grad_(True)
    truncate_to_rank(Zdeg, 2).sum().backward()
    assert Zdeg.grad is not None and not torch.isnan(Zdeg.grad).any() \
        and not torch.isinf(Zdeg.grad).any(), "NaN/Inf grad on degenerate spectrum"
    print("  truncation correct; degenerate-spectrum backward is NaN/Inf-free")

    print("\n[4] MatrixCompositionModel forward / backward / grad across MIXED hops (no NaN/Inf)")
    cfg = te.TaskEConfig(d=16, K=8, variant="permutation",
                         H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    gen = torch.Generator(device=device).manual_seed(1)
    model = MatrixCompositionModel(d=16, h=64).to(device)
    b = te.sample_batch(cfg, 32, gen, hop_set=cfg.H_train, device=device)
    pred, Zfwd = model(b)
    assert pred.shape == b["targets"].shape, (pred.shape, b["targets"].shape)
    assert not torch.isnan(pred).any()
    loss = cosine_loss(pred, b["targets"])
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), f"bad grad {name}"
    print(f"  forward {tuple(pred.shape)}, loss {loss.item():.4f}, all params finite grad")

    print("\n[5] numerical-stability probe: the REAL learned Z (post a few optimizer "
          "steps) stays finite through high h -- the repeated-matmul explosion/vanish/NaN "
          "risk the design doc flags, exercised on the model's OWN output, not just "
          "Z_ideal/random (gauntlet/AUDIT_task_e_correctness.md M2)")
    stab_model = MatrixCompositionModel(d=16, h=64).to(device)
    stab_opt = torch.optim.Adam(stab_model.parameters(), lr=3e-4)
    stab_gen = torch.Generator(device=device).manual_seed(2)
    for _ in range(20):     # a handful of real SGD steps -- not just random init
        bs = te.sample_batch(cfg, 32, stab_gen, hop_set=cfg.H_train, device=device)
        pred_s, _ = stab_model(bs)
        loss_s = cosine_loss(pred_s, bs["targets"])
        stab_opt.zero_grad()
        loss_s.backward()
        stab_opt.step()
    with torch.no_grad():
        for h_probe in cfg.H_extra:      # (7, 21) by default -- the actual far held-out probes
            bh = te.sample_batch(cfg, 64, stab_gen, hop_set=(h_probe,), device=device)
            pred_h, _ = stab_model(bh)
            assert torch.isfinite(pred_h).all(), \
                f"the model's OWN learned Z produced non-finite output at h={h_probe} -- " \
                f"repeated-matmul explosion/vanish/NaN (not just Z_ideal/random)"
    print(f"  real learned Z (after 20 SGD steps) stays finite through h in {cfg.H_extra}")

    print("\n[6] force_rank_k path (C1) -- composition uses the SAME rank-k Z at every hop")
    model.zero_grad()
    pred_k, Zk = model(b, force_rank_k=2)
    assert effective_rank(Zk).mean().item() <= 2 + 1e-2, "force_rank_k=2 didn't constrain rank"
    cosine_loss(pred_k, b["targets"]).backward()
    for p in model.parameters():
        assert not (p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()))
    print("  force_rank_k=2 constrains rank, composes across hops, backprops cleanly")

    print("\n[7] BLANK-OUT + composition-purity test (C_composition-purity)")
    sig = set(inspect.signature(MatrixCompositionModel.compose).parameters) - {"self"}
    assert sig <= {"Z", "query_keys", "hops"}, f"compose takes inputs beyond (Z, query_keys, hops): {sig}"

    n_model = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    assert n_model == n_encoder, \
        f"MatrixCompositionModel has {n_model - n_encoder} params beyond the encoder -- a " \
        f"hidden per-hop parameter would violate C_composition-purity"

    keys = b["keys"].clone().requires_grad_(True)
    values = b["values"].clone().requires_grad_(True)
    Zg = model.encode(keys, values)
    pred_g = model.compose(Zg, b["query_keys"], b["hops"])
    gk = torch.autograd.grad(pred_g.sum(), keys, retain_graph=True, allow_unused=True)[0]
    assert gk is not None and gk.abs().sum() > 0, "bindings don't affect pred at all?!"
    Z_leaf = Zg.detach().clone().requires_grad_(True)
    pred_leaf = model.compose(Z_leaf, b["query_keys"], b["hops"])
    g_leak = torch.autograd.grad(pred_leaf.sum(), keys, allow_unused=True)[0]
    assert g_leak is None, "LEAK: keys reach the decoder outside Z"

    with torch.no_grad():
        Zc = model.encode(b["keys"], b["values"])
        pred_shared = model.compose(Zc, b["query_keys"], b["hops"])
        for bb in range(4):     # a handful of rows/queries is enough -- structural, not statistical
            for qq in range(b["hops"].shape[1]):
                h_bq = int(b["hops"][bb, qq].item())
                cur = b["query_keys"][bb, qq]
                for _ in range(h_bq):
                    cur = Zc[bb] @ cur
                assert torch.allclose(cur, pred_shared[bb, qq], atol=1e-5), \
                    f"h-purity violated at (row {bb}, query {qq}, h={h_bq})"
    print("  compose() signature pinned to (Z, query_keys, hops); zero extra learned params; "
          "gradient blank-out clean; h-purity confirmed against an independent per-query loop")

    print("\n[8] C_MLP baseline: OOV one-hot(h) is structurally all-zero at held-out h")
    mlp_model = MLPShortcutModel(d=16, h=64, h_train_max=max(cfg.H_train),
                                 h_train=cfg.H_train).to(device)
    held_out_hops = torch.tensor([[4, 5, 6]], device=device)
    oh = mlp_model._one_hot_h(held_out_hops)
    assert (oh == 0).all(), "C_MLP baseline's one-hot(h) should be all-zero (OOV) at held-out h"
    pred_mlp, Zm = mlp_model(b)
    assert pred_mlp.shape == b["targets"].shape and not torch.isnan(pred_mlp).any()
    cosine_loss(pred_mlp, b["targets"]).backward()
    for p in mlp_model.parameters():
        assert not (p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()))
    print("  C_MLP baseline forward/backward clean; held-out one-hot(h) confirmed OOV (all-zero)")

    print("\n[9] RESOLUTION pre-flight: recovered_frac@0.9 distinguishes Z_ideal from a broken Z at held-out h")
    bb2 = te.sample_batch(cfg, 256, gen, hop_set=(5,), device=device)   # a held-out hop
    pred_ideal = MatrixCompositionModel.compose(bb2["z_ideal"], bb2["query_keys"], bb2["hops"])
    cos_ideal = td.recovery_cosine(pred_ideal, bb2["targets"])
    frac_ideal = (cos_ideal > 0.9).float().mean().item()
    assert frac_ideal > 0.95, f"Z_ideal should recover held-out hops near-perfectly (frac {frac_ideal:.3f})"
    Z_broken = torch.randn_like(bb2["z_ideal"])          # a deliberately WRONG operator
    pred_broken = MatrixCompositionModel.compose(Z_broken, bb2["query_keys"], bb2["hops"])
    cos_broken = td.recovery_cosine(pred_broken, bb2["targets"])
    frac_broken = (cos_broken > 0.9).float().mean().item()
    assert frac_broken < 0.2, \
        f"a random (non-composing) Z should mostly FAIL held-out recovery (frac {frac_broken:.3f}); " \
        f"M3_E's recovered_frac@0.9 metric would not be resolvable"
    print(f"  recovered_frac@0.9 at h=5: Z_ideal {frac_ideal:.3f} vs random Z {frac_broken:.3f} "
          f"-> M3_E metric is resolvable")

    print("\n" + "=" * 60 + "\n  ALL SMOKE CHECKS PASSED\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", choices=["matrix", "mlp"], default="matrix")
    ap.add_argument("--variant", choices=["permutation", "chain"], default="permutation")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--N", type=int, default=None, help="pool size (chain variant only; default K+4)")
    ap.add_argument("--h-train", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--h-test", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--h-extra", type=int, nargs="+", default=[7, 21],
                    help="further-out graceful-degradation probe; for the permutation "
                         "variant, TaskEConfig rejects any value h with h %% K == 0 or "
                         "h %% K equal to a training hop's residue (periodicity-confound "
                         "guard, gauntlet/AUDIT_task_e_validity.md Finding B)")
    ap.add_argument("--orthogonal", dest="orthogonal", action="store_true", default=True,
                    help="orthonormal entity pool (GATE DEFAULT)")
    ap.add_argument("--gaussian", dest="orthogonal", action="store_false",
                    help="near-orthogonal Gaussian pool (weaker knee, C7 only approximate; dev only)")
    ap.add_argument("--force-rank-k", type=int, default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--h", type=int, default=64, dest="h_dim")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-refine", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-z", action="store_true",
                    help="embed a small post-training Z dump in the output JSON "
                         "(4 eval examples at h=1: learned Z + analytic z_ideal, "
                         "matrix model only) for post-hoc entity-subspace-restricted "
                         "spectral analysis -- added 2026-07-02 after the 40K "
                         "calibration showed converged Z at stable_rank~d (not ~K), "
                         "which whole-matrix metrics cannot disambiguate")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        smoke(device)
        return

    if args.model == "mlp" and args.force_rank_k is not None:
        print("WARNING: --force-rank-k has no effect on --model mlp "
              "(C_MLP is rank-unconstrained by design).", flush=True)
    if args.model == "mlp" and sorted(args.h_train) != list(range(1, max(args.h_train) + 1)):
        # C_MLP's one-hot(h) encodes the RANGE [1, max(H_train)] as in-vocab,
        # not the literal H_train set. A non-contiguous H_train (e.g. {1,3})
        # would silently mark the skipped, untrained depth as in-vocabulary --
        # a misleading floor. Refuse rather than mislead.
        sys.exit(f"ERROR: --model mlp requires a contiguous H_train (1..max); got "
                 f"{sorted(args.h_train)}. The C_MLP one-hot(h) baseline assumes "
                 f"[1, max] in-vocabulary; a non-contiguous set would mislabel an "
                 f"untrained depth as in-vocab.")

    # Chain-variant feasibility pre-flight: a finite chain of length h needs
    # >= h edges (K >= H_max) and a pool of > h entities (N-1 >= H_max). The
    # permutation gate default K=8 with the far-out probe H_extra=(7,21) has
    # H_max=21 > K, so `--variant chain` with otherwise-default args is
    # INFEASIBLE. Fail with an actionable message here rather than a bare
    # AssertionError from deep inside TaskEConfig.__post_init__.
    if args.variant == "chain":
        h_max = max([*args.h_train, *args.h_test, *args.h_extra])
        n_default = args.N if args.N is not None else args.K + 4
        problems = []
        if args.K < h_max:
            problems.append(f"K={args.K} < H_max={h_max} (need K >= H_max: a chain of "
                            f"depth {h_max} requires at least {h_max} edges)")
        if n_default - 1 < h_max:
            problems.append(f"N={n_default} gives only N-1={n_default - 1} possible chain "
                            f"edges < H_max={h_max} (need N >= H_max+1={h_max + 1})")
        if n_default > args.d:
            problems.append(f"N={n_default} > d={args.d} (orthonormal pool needs N <= d)")
        if problems:
            sys.exit("ERROR: --variant chain is infeasible with these args:\n  - "
                     + "\n  - ".join(problems)
                     + f"\n  Fix e.g.: --variant chain --d 24 --K {max(h_max, args.K)} "
                       f"--N {max(h_max + 1, args.K + 2)}  (or lower --h-extra).")

    cfg = te.TaskEConfig(d=args.d, K=args.K, variant=args.variant, N=args.N,
                         H_train=tuple(args.h_train), H_test=tuple(args.h_test),
                         H_extra=tuple(args.h_extra), orthogonal=args.orthogonal)

    model = make_model(args.model, args.d, args.h_dim, args.n_layers, args.n_heads,
                       args.n_refine, h_train_max=max(cfg.H_train),
                       h_train=cfg.H_train).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} variant={args.variant} d={args.d} K={args.K} "
          f"H_train={cfg.H_train} H_test={cfg.H_test} H_extra={cfg.H_extra} "
          f"orthogonal={args.orthogonal} force_rank_k={args.force_rank_k} "
          f"params={n_params / 1e6:.3f}M device={device}", flush=True)

    result = train(model, cfg, device, args.model, steps=args.steps,
                   batch_size=args.batch_size, lr=args.lr,
                   force_rank_k=args.force_rank_k, seed=args.seed)
    # MAJOR fix (gauntlet/AUDIT_task_e_correctness.md M1): record the full run
    # configuration in the output JSON, mirroring run_task_d.py's evaluate()
    # dict, so a directory of Task E sweep result files is self-describing --
    # a downstream orchestrator (or a human in three weeks) doesn't have to
    # re-derive K/d/variant/hops from stdout logs or a filename convention.
    result.update({
        "model_type": args.model, "variant": args.variant, "d": args.d, "K": args.K,
        "N": cfg.pool_size, "H_train": list(cfg.H_train), "H_test": list(cfg.H_test),
        "H_extra": list(cfg.H_extra), "orthogonal": args.orthogonal,
        "force_rank_k": args.force_rank_k, "n_params": n_params,
        "seed": args.seed, "steps": args.steps,
    })
    # Optional Z dump (additive; default off -- no behavior change for existing
    # sweeps). One fixed, seeded eval batch; tiny (4 x 2 x d^2 floats in JSON).
    if args.save_z and args.model == "matrix":
        model.eval()
        gz = torch.Generator(device=device).manual_seed(args.seed + 20_000)
        bz = te.sample_batch(cfg, 4, gz, hop_set=(1,), device=device)
        with torch.no_grad():
            _, Z_dump = model(bz, force_rank_k=args.force_rank_k)
        result["Z_dump"] = {
            "Z": Z_dump.detach().cpu().tolist(),
            "z_ideal": bz["z_ideal"].detach().cpu().tolist(),
            "note": "4 eval examples, hop_set=(1,), seed+20000 generator; "
                    "project Z onto z_ideal's row/column spaces to test the "
                    "invariant-subspace hypothesis (restricted operator == "
                    "exact K-cycle, restricted rank == K).",
        }
    print("\nRESULT:", json.dumps(result, indent=2, default=str), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
