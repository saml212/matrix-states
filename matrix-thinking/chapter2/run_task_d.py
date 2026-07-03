"""Task D runner — train / eval / smoke for the Chapter 2 rank gate.

Measurements (see TASK_D_PREREGISTRATION.md §6):
  M1  learned effective/stable rank of Z vs K   (unconstrained model)
  M2  eval-time rank-k truncation curve          (knee expected at k ~ K)
  M3  train-time force-rank-k recovery            (--force-rank-k; PRIMARY causal test)

Success metric: a binding is recovered if cos(Z @ key_q, v_q) > tau. Reported at
tau in {0.9, 0.95, 0.99} plus the full cosine distribution. Continuous exact
recovery, never a K-way argmax (Nichani et al. 2412.06538).

GATE DEFAULT is --orthogonal (exactly orthonormal keys+values): the round-1 audit
showed that with Gaussian near-orthogonal vectors a rank-(K-2) matrix clears
tau=0.9 ~90% of the time, smearing the knee. Orthonormal + high tau + a dense
rank grid make rank K-1 genuinely fail one binding, sharpening the knee.

The vector baseline (M4) is NOT part of the Stage-1 gate: HRRVectorMemory is not
param/state-matched (needs redesign, Stage-1b). --model vector runs it for
development only; do not treat it as the C2 control yet.

Run the smoke gate FIRST on the cluster (seconds):  python run_task_d.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
import task_d as td
from task_d import TaskDConfig, sample_batch, recovery_cosine
from model_v4 import MatrixMemoryModel, HRRVectorMemory
from rank_utils import truncate_to_rank, effective_rank, stable_rank

TAUS = (0.9, 0.95, 0.99)


def make_model(model_type, d, h, n_layers, n_heads, n_refine):
    if model_type == "matrix":
        return MatrixMemoryModel(d, h, n_layers, n_heads, n_refine)
    if model_type == "vector":
        return HRRVectorMemory(d, h)
    raise ValueError(f"unknown model_type {model_type}")


def cosine_loss(pred, target):
    """1 - mean cosine similarity; directly optimizes the (direction-only)
    success metric."""
    return (1.0 - recovery_cosine(pred, target)).mean()


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
def evaluate(model, cfg, gen, device, model_type, force_rank_k=None,
             rank_ks=None, n_batches=8, batch_size=256):
    """Evaluate recovery. If force_rank_k is set, the HEADLINE metric is the
    rank-constrained model (M3) — this matches how the model was TRAINED. The M2
    truncation curve is only meaningful (and only computed) for the unconstrained
    model (force_rank_k is None)."""
    model.eval()
    if rank_ks is None:
        rank_ks = tuple(range(1, cfg.d + 1))          # dense grid straddling K
    cos_all, er_all, sr_all = [], [], []
    rk_cos = {k: [] for k in rank_ks}
    for _ in range(n_batches):
        b = sample_batch(cfg, batch_size, gen, device=device)
        if model_type == "matrix":
            # Headline Z uses the SAME rank constraint the model was trained under.
            Z = model.encode(b["keys"], b["values"], force_rank_k=force_rank_k)
            pred = model.unbind(Z, b["query_keys"])
            cos_all.append(recovery_cosine(pred, b["targets"]))
            er_all.append(effective_rank(Z))
            sr_all.append(stable_rank(Z))
            if force_rank_k is None:                    # M2 curve (unconstrained only)
                for k in rank_ks:
                    Zk = truncate_to_rank(Z, k)
                    pk = model.unbind(Zk, b["query_keys"])
                    rk_cos[k].append(recovery_cosine(pk, b["targets"]))
        else:
            pred, _ = model(b)
            cos_all.append(recovery_cosine(pred, b["targets"]))

    cos = torch.cat([c.reshape(-1) for c in cos_all])
    out = {"K": cfg.K, "d": cfg.d, "n_query": cfg.queries, "model": model_type,
           "orthogonal": cfg.orthogonal, "force_rank_k": force_rank_k}
    out.update(_recovery_stats(cos))
    if model_type == "matrix":
        er, sr = torch.cat(er_all), torch.cat(sr_all)
        out["effective_rank_mean"] = er.mean().item()      # M1 (primary metric)
        out["effective_rank_std"] = er.std().item()
        out["stable_rank_mean"] = sr.mean().item()         # M1 (secondary metric)
        if force_rank_k is None:
            out["rankk_curve"] = {                          # M2
                int(k): _recovery_stats(torch.cat([v.reshape(-1) for v in rk_cos[k]]))
                for k in rank_ks
            }
    return out


def train(model, cfg, device, model_type, steps=3000, batch_size=256, lr=3e-4,
          force_rank_k=None, seed=0, log_every=500):
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n_skipped = 0
    for step in range(1, steps + 1):
        b = sample_batch(cfg, batch_size, gen, device=device)
        pred, _ = model(b, force_rank_k=force_rank_k)
        loss = cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        # Gradient hygiene: SVD/eigh backward through rank-k truncation can
        # occasionally produce a non-finite grad on an unlucky batch. SKIP that
        # step (don't kill the run) and clip otherwise. A few skips out of `steps`
        # don't affect convergence; a mostly-skipped run just shows low recovery
        # (valid data), and n_skipped is reported so we can see it.
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
    # Final eval: fresh generator stream (held-out samples, C5), SAME rank
    # constraint the model trained under (fixes the round-1 FATAL).
    eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    res = evaluate(model, cfg, eval_gen, device, model_type, force_rank_k=force_rank_k)
    res["n_skipped_steps"] = n_skipped
    return res


# ---------------------------------------------------------------------------
# Smoke gate (run FIRST on cluster; no real training)
# ---------------------------------------------------------------------------

def smoke(device):
    print("=" * 60 + "\n  TASK D SMOKE GATE\n" + "=" * 60)

    print("\n[1] generator self-test (gaussian + orthonormal paths)")
    td._self_test()

    print("\n[2] rank_utils sanity + degenerate-spectrum NaN/Inf-free backward")
    Z = torch.randn(4, 16, 16, device=device)
    for k in (1, 2, 4, 8):
        er = effective_rank(truncate_to_rank(Z, k)).mean().item()
        assert er <= k + 1e-2, f"truncate_to_rank({k}) gave eff rank {er:.3f} > {k}"
    # Constructed degenerate spectrum [3,1,1,0]: full-SVD backward would NaN here.
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(4, 4, device=device))
    V, _ = torch.linalg.qr(torch.randn(4, 4, device=device))
    Zdeg = (U @ torch.diag(torch.tensor([3., 1., 1., 0.], device=device)) @ V.T
            ).unsqueeze(0).requires_grad_(True)
    truncate_to_rank(Zdeg, 2).sum().backward()
    assert Zdeg.grad is not None and not torch.isnan(Zdeg.grad).any() \
        and not torch.isinf(Zdeg.grad).any(), "NaN/Inf grad on degenerate spectrum"
    print("  truncation correct; degenerate-spectrum backward is NaN/Inf-free")

    print("\n[3] matrix model forward / backward / grad (no NaN/Inf)")
    cfg = TaskDConfig(d=16, K=8, orthogonal=True)
    gen = torch.Generator(device=device).manual_seed(1)
    model = MatrixMemoryModel(d=16, h=64).to(device)
    b = sample_batch(cfg, 32, gen, device=device)
    pred, Z = model(b)
    assert pred.shape == b["targets"].shape, (pred.shape, b["targets"].shape)
    assert not torch.isnan(pred).any()
    loss = cosine_loss(pred, b["targets"])
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), f"bad grad {name}"
    print(f"  forward {tuple(pred.shape)}, loss {loss.item():.4f}, all params finite grad")

    print("\n[4] force_rank_k train-time path (C1)")
    model.zero_grad()
    pred_k, Zk = model(b, force_rank_k=2)
    assert effective_rank(Zk).mean().item() <= 2 + 1e-2, "force_rank_k=2 didn't constrain rank"
    cosine_loss(pred_k, b["targets"]).backward()
    for p in model.parameters():
        assert not (p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()))
    print("  force_rank_k=2 constrains rank and backprops cleanly")

    print("\n[5] BLANK-OUT test (meaningful): decoder path touches bindings ONLY via Z")
    keys = b["keys"].clone().requires_grad_(True)
    values = b["values"].clone().requires_grad_(True)
    Zg = model.encode(keys, values)
    pred_g = model.unbind(Zg, b["query_keys"])
    gk = torch.autograd.grad(pred_g.sum(), keys, retain_graph=True, allow_unused=True)[0]
    assert gk is not None and gk.abs().sum() > 0, "bindings don't affect pred at all?!"
    # With Z as an independent leaf, keys must have NO remaining path to the decoder.
    Z_leaf = Zg.detach().clone().requires_grad_(True)
    pred_leaf = model.unbind(Z_leaf, b["query_keys"])
    g_leak = torch.autograd.grad(pred_leaf.sum(), keys, allow_unused=True)[0]
    assert g_leak is None, "LEAK: keys reach the decoder outside Z"
    import inspect
    sig = set(inspect.signature(MatrixMemoryModel.unbind).parameters) - {"self"}
    assert sig <= {"Z", "query_keys"}, f"unbind takes inputs beyond (Z, query_keys): {sig}"
    print("  decoder gradient flows only through Z; unbind signature is (Z, query_keys)")

    print("\n[6] RESOLUTION pre-flight: rank K-1 must FAIL the literal recovered_frac@0.99 criterion")
    # Checks the ACTUAL knee metric (recovered_frac@0.99), not a mean-cosine proxy.
    # The exact memory Sigma v_k k_k^T has a DEGENERATE spectrum (all sigma=1), so
    # rank-(K-1) truncation drops a spread direction: every binding degrades to
    # ~sqrt((K-1)/K) mean cosine, pushing ALL of them below tau=0.99 -> frac ~ 0.
    cfgo = TaskDConfig(d=16, K=8, orthogonal=True)
    bb = sample_batch(cfgo, 256, gen, device=device)
    Zexact = torch.einsum("bki,bkj->bij", bb["values"], bb["keys"])   # exact correlation memory
    pred_full = model.unbind(Zexact, bb["query_keys"])
    pred_tr = model.unbind(truncate_to_rank(Zexact, cfgo.K - 1), bb["query_keys"])
    cos_full = recovery_cosine(pred_full, bb["targets"])
    cos_tr = recovery_cosine(pred_tr, bb["targets"])
    frac_full = (cos_full > 0.99).float().mean().item()
    frac_tr = (cos_tr > 0.99).float().mean().item()
    assert frac_full > 0.95, f"exact recovery should pass tau=0.99 (frac {frac_full:.3f})"
    assert frac_tr < 0.5, \
        f"rank-{cfgo.K-1} should mostly FAIL tau=0.99 (frac {frac_tr:.3f}); knee unresolvable at real criterion"
    print(f"  recovered_frac@0.99: full {frac_full:.3f}, rank-{cfgo.K-1} {frac_tr:.3f} -> knee resolvable")

    print("\n[7] vector baseline forward/backward (dev only; NOT the Stage-1 C2 control)")
    vm = HRRVectorMemory(d=16, h=64).to(device)
    pv, _ = vm(b)
    assert pv.shape == b["targets"].shape
    cosine_loss(pv, b["targets"]).backward()
    print("  vector baseline runs (flagged: not param/state-matched — Stage-1b)")

    print("\n" + "=" * 60 + "\n  ALL SMOKE CHECKS PASSED\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", choices=["matrix", "vector"], default="matrix")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-query", type=int, default=None)   # None -> all K (theorem: rank>=K)
    ap.add_argument("--orthogonal", dest="orthogonal", action="store_true", default=True,
                    help="orthonormal keys+values (GATE DEFAULT)")
    ap.add_argument("--gaussian", dest="orthogonal", action="store_false",
                    help="near-orthogonal Gaussian keys+values (weaker knee; dev only)")
    ap.add_argument("--force-rank-k", type=int, default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-refine", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        smoke(device)
        return

    if args.n_query is not None and args.n_query < args.K:
        print(f"WARNING: n_query={args.n_query} < K={args.K}: the provable bound "
              f"weakens to rank>=n_query. The gate should query all K.", flush=True)
    cfg = TaskDConfig(d=args.d, K=args.K, n_query=args.n_query, orthogonal=args.orthogonal)
    if args.model == "matrix" and args.K > args.d:
        print(f"WARNING: K={args.K} > d={args.d}: no exact solution (lossy K>d regime).", flush=True)
    if args.model == "vector" and args.force_rank_k is not None:
        print("WARNING: --force-rank-k is ignored for --model vector (no matrix rank).", flush=True)

    model = make_model(args.model, args.d, args.h, args.n_layers, args.n_heads,
                       args.n_refine).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} d={args.d} K={args.K} orthogonal={args.orthogonal} "
          f"force_rank_k={args.force_rank_k} params={n_params/1e6:.3f}M device={device}", flush=True)

    result = train(model, cfg, device, args.model, steps=args.steps,
                   batch_size=args.batch_size, lr=args.lr,
                   force_rank_k=args.force_rank_k, seed=args.seed)
    result.update({"n_params": n_params, "seed": args.seed, "steps": args.steps})
    print("\nRESULT:", json.dumps(result, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
