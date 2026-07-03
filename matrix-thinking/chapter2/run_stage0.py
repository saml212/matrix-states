"""Stage 0 training entry point — see STAGE0_DESIGN.md (frozen spec).

Trains ONE (variant, d, K, seed, steps[, force_rank_k]) cell and writes a
single result JSON with Wave 0's diagnostic instrumentation (section 6.2):
  trajectory   dense (<=200-step default) loss / train-batch rank / grad-norm
               log — dates a sharp transition (H_dead-init vs H_late-
               transition vs H_undertrained, section 3)
  checkpoints  <=2000-step default full eval: recovered_frac@tau + mean cos +
               effective/stable rank + row-query AND post-reader collision
               stats — every-checkpoint life scoring (section 6.1) and the
               budget-matched 2x-mark read out of a 2.5x-step Wave 0 run
               (Arm 0, section 6) both depend on this being present, not
               inferred from the final number alone.
Both are written INCREMENTALLY to --out after every checkpoint (build
requirement, section 6.3), so a run killed by the orchestrator's per-run
timeout still leaves an inspectable partial result. Every incremental dump
carries "complete": false; ONLY the final post-training write in main()
carries "complete": true — run_stage0_sweep.is_done() requires it, so a
killed run is always retried, never silently accepted (audit FATAL-1).

variant="baseline" (default) reproduces model_v4.BindingEncoder's
architecture exactly (model_s0.py's VARIANTS dict + smoke gate section [3]
assert this) — this file's default invocation with no flags IS Task D's
unmodified Arm 0, run through Stage 0's instrumentation.

Run the smoke gate FIRST (seconds): python run_stage0.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
import task_d as td
from task_d import TaskDConfig, sample_batch, recovery_cosine
from rank_utils import truncate_to_rank, effective_rank, stable_rank
from model_s0 import MatrixMemoryModelS0, VARIANTS, mup_lr_mult
# Reused verbatim from Task D's audited harness (no modification) — avoids a
# second, driftable definition of the loss / recovery-stats formulas.
from run_task_d import cosine_loss, _recovery_stats, TAUS

PARAM_GROUPS = ("row_out", "row_queries", "encoder_stack")


def make_model(variant: str, d: int, h_dim: int, n_layers: int, n_heads: int,
               n_refine: int) -> MatrixMemoryModelS0:
    preset = VARIANTS[variant]["model"]
    return MatrixMemoryModelS0(d=d, h=h_dim, n_layers=n_layers, n_heads=n_heads,
                                n_refine=n_refine, **preset)


def _resolve_warmup(preset_warmup, steps: int, cli_override):
    if cli_override is not None:
        return cli_override
    if preset_warmup == "10pct":
        return max(1, steps // 10)
    return int(preset_warmup)


def _param_group(name: str) -> str:
    if name.startswith("encoder.row_out"):
        return "row_out"
    if name.startswith("encoder.row_queries"):
        return "row_queries"
    return "encoder_stack"


def _group_grad_norms(named_params) -> dict:
    """Section 6.2 item 4 (MINOR-12): per-parameter-group grad norms, called
    both before and after the flat clip_grad_norm_(1.0) so a d-dependent
    clipping pathology (candidate 6's diagnostic content, obtained "for
    free") is directly visible in the trajectory log."""
    sq = {g: 0.0 for g in PARAM_GROUPS}
    for name, p in named_params:
        if p.grad is None:
            continue
        sq[_param_group(name)] += p.grad.detach().float().pow(2).sum().item()
    return {g: v ** 0.5 for g, v in sq.items()}


def _pairwise_cosine_stats(vecs: torch.Tensor) -> dict:
    """vecs: (..., n, h) -> pairwise-cosine stats over the n*(n-1)/2 off-
    diagonal pairs (STAGE0_DESIGN.md section 6.2 item 3, H_collision
    instrumentation). Works uncached for both the unbatched row_queries
    parameter (n=d, no leading batch dim) and the per-sample post-reader `q`
    (leading batch dim B) — the max/mean/frac reduce over ALL leading dims
    either way, which for the batched case means "fraction of (sample, pair)
    entries above 0.5", a fine coarse diagnostic."""
    v = vecs / vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    gram = v @ v.transpose(-1, -2)                       # (..., n, n)
    n = gram.shape[-1]
    iu = torch.triu_indices(n, n, offset=1)
    pairs = gram[..., iu[0], iu[1]]                       # (..., n*(n-1)/2)
    return {
        "max": pairs.max().item(),
        "mean": pairs.mean().item(),
        "frac_above_0.5": (pairs > 0.5).float().mean().item(),
    }


def _build_param_groups(model, lr: float, mup_h_base, h_dim: int):
    if mup_h_base is None:
        return [{"params": list(model.parameters()), "lr": lr}]
    row_out_params, other_params = [], []
    for name, p in model.named_parameters():
        (row_out_params if _param_group(name) == "row_out" else other_params).append(p)
    mult = mup_lr_mult(h_dim, mup_h_base)
    return [{"params": other_params, "lr": lr},
            {"params": row_out_params, "lr": lr * mult}]


def _build_scheduler(opt, warmup_steps: int, total_steps: int):
    """Candidate 1 (STAGE0_DESIGN.md section 4): linear warmup then cosine
    decay to 10% of base LR. The warmup SHAPE is the cited convention
    (Vaswani et al. 2017 / Goyal et al. 2017); the exact decay-to-0.1x tail
    is a standard, undramatic default not itself pinned by either citation —
    minimal-risk choice, recorded here.

    Step-indexing note (audit MINOR-5): LambdaLR's multiplier for TRAINING
    step t is lr_lambda(last_epoch = t-1) (sched.step() runs after
    opt.step()), so lr_lambda re-indexes to step = last_epoch + 1. With
    this, peak LR is first USED at training step == warmup_steps exactly
    (not warmup_steps+1), and the cosine tail lands on the 0.1x floor at
    the final step."""
    def lr_lambda(last_epoch):
        step = last_epoch + 1              # multiplier consumed at training step `step`
        if step <= warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def _dump(path, obj):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"  (incremental write failed, non-fatal: {e!r})", flush=True)


@torch.no_grad()
def evaluate(model, cfg, gen, device, force_rank_k=None, rank_ks=None,
            n_batches=4, batch_size=256):
    """Task D's run_task_d.evaluate(), extended with the section-6.2
    row-query-vs-post-reader collision measurement on the SAME batches (so
    collision and recovery are read off identical held-out samples)."""
    model.eval()
    if rank_ks is None:
        rank_ks = tuple(range(1, cfg.d + 1))          # dense grid straddling K (M2, free on unconstrained runs)
    cos_all, er_all, sr_all, q_stats = [], [], [], []
    rk_cos = {k: [] for k in rank_ks}
    for _ in range(n_batches):
        b = sample_batch(cfg, batch_size, gen, device=device)
        Z, q = model.encode(b["keys"], b["values"], force_rank_k=force_rank_k,
                            return_intermediate=True)
        pred = model.unbind(Z, b["query_keys"])
        cos_all.append(recovery_cosine(pred, b["targets"]))
        er_all.append(effective_rank(Z))
        sr_all.append(stable_rank(Z))
        q_stats.append(_pairwise_cosine_stats(q))
        if force_rank_k is None:                        # M2 curve (unconstrained only)
            for k in rank_ks:
                Zk = truncate_to_rank(Z, k)
                pk = model.unbind(Zk, b["query_keys"])
                rk_cos[k].append(recovery_cosine(pk, b["targets"]))

    cos = torch.cat([c.reshape(-1) for c in cos_all])
    out = {"K": cfg.K, "d": cfg.d, "n_query": cfg.queries,
           "orthogonal": cfg.orthogonal, "force_rank_k": force_rank_k}
    out.update(_recovery_stats(cos))
    er, sr = torch.cat(er_all), torch.cat(sr_all)
    out["effective_rank_mean"] = er.mean().item()
    out["effective_rank_std"] = er.std().item()
    out["stable_rank_mean"] = sr.mean().item()
    out["q_collision"] = {
        k: sum(s[k] for s in q_stats) / len(q_stats) for k in ("max", "mean", "frac_above_0.5")
    }
    # Section 6.2 item 3 (audit MAJOR-2): the RAW row_queries PARAMETER's
    # collision stats at every checkpoint — not just at init and not only the
    # post-reader q. The zero-slack-drift signature (section 2.2/MINOR-9) is
    # the TRAJECTORY of this parameter's pairwise-cosine distribution during
    # training; q_collision alone can't separate parameter drift from
    # reader-induced mixing.
    out["row_query_collision"] = _pairwise_cosine_stats(
        model.encoder.row_queries.unsqueeze(0))
    if force_rank_k is None:
        out["rankk_curve"] = {                          # M2 — free on every Wave B M1 unconstrained run
            int(k): _recovery_stats(torch.cat([v.reshape(-1) for v in rk_cos[k]]))
            for k in rank_ks
        }
    model.train()
    return out


def _assemble_result(variant, cfg, steps, seed, force_rank_k, h_dim, mup_h_base,
                     warmup_steps, trajectory, checkpoints, n_skipped,
                     init_row_query_collision, t0, timed_out, steps_completed,
                     complete):
    # `complete` is the crash-safety sentinel (audit FATAL-1): every
    # INCREMENTAL checkpoint dump inside train() carries complete=False, so a
    # run killed mid-flight (orchestrator kill -9, OOM, pod eviction) leaves
    # a partial JSON that run_stage0_sweep.is_done() REJECTS and retries.
    # complete=True is set ONLY on the result train() returns after the step
    # loop genuinely ran all `steps` (never on the internal-timeout early
    # break), and reaches disk only via main()'s final write.
    result = {
        "variant": variant, "K": cfg.K, "d": cfg.d, "orthogonal": cfg.orthogonal,
        "force_rank_k": force_rank_k, "seed": seed, "steps": steps,
        "steps_completed": steps_completed,
        "complete": complete,
        "h": h_dim, "mup_h_base": mup_h_base, "warmup_steps": warmup_steps,
        "n_skipped_steps": n_skipped,
        "init_row_query_collision": init_row_query_collision,
        "trajectory": trajectory, "checkpoints": checkpoints,
        "wall_s": time.time() - t0, "timed_out": timed_out,
    }
    if checkpoints:
        final = checkpoints[-1]
        skip = {"K", "d", "orthogonal", "force_rank_k", "step"}
        for k, v in final.items():
            if k not in result and k not in skip:
                result[k] = v
        result["final_step"] = final["step"]
    return result


def train(model, cfg, device, variant, steps=3000, batch_size=256, lr=3e-4,
         seed=0, force_rank_k=None, log_every=200, ckpt_every=2000,
         out_path=None, warmup_steps=0, mup_h_base=None, h_dim=64,
         timeout_s=None):
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    param_groups = _build_param_groups(model, lr, mup_h_base, h_dim)
    opt = torch.optim.Adam(param_groups, lr=lr)
    sched = _build_scheduler(opt, warmup_steps, steps) if warmup_steps > 0 else None
    model.train()

    with torch.no_grad():
        init_collision = _pairwise_cosine_stats(model.encoder.row_queries.unsqueeze(0))

    trajectory, checkpoints = [], []
    n_skipped = 0
    timed_out = False
    steps_completed = 0   # actual last executed step (NOT trajectory[-1], which
                          # lags by up to log_every-1 when steps % log_every != 0)

    for step in range(1, steps + 1):
        steps_completed = step
        b = sample_batch(cfg, batch_size, gen, device=device)
        pred, Z = model(b, force_rank_k=force_rank_k)
        loss = cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        # Gradient hygiene (identical policy to run_task_d.py::train): skip a
        # non-finite step rather than kill the run; a few skips don't affect
        # convergence and n_skipped is reported.
        pre_clip = _group_grad_norms(model.named_parameters())
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            post_clip = _group_grad_norms(model.named_parameters())
            opt.step()
        else:
            n_skipped += 1
            post_clip = None
        if sched is not None:
            sched.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                er = effective_rank(Z).mean().item()
                sr = stable_rank(Z).mean().item()
            trajectory.append({
                "step": step, "loss": loss.item(),
                "eff_rank_trainbatch": er, "stable_rank_trainbatch": sr,
                "grad_norm_pre_clip": pre_clip, "grad_norm_post_clip": post_clip,
                "lr": opt.param_groups[0]["lr"],
            })
            extra = f"  [skipped {n_skipped} non-finite]" if n_skipped else ""
            print(f"  step {step:6d}  loss {loss.item():.4f}  eff_rank {er:.3f}{extra}", flush=True)

        if step % ckpt_every == 0 or step == steps:
            eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000 + step)
            res = evaluate(model, cfg, eval_gen, device, force_rank_k=force_rank_k,
                           n_batches=4, batch_size=batch_size)
            res["step"] = step
            checkpoints.append(res)
            # Incremental dump: ALWAYS complete=False (FATAL-1 sentinel) — only
            # main()'s post-train write may carry complete=True.
            partial = _assemble_result(variant, cfg, steps, seed, force_rank_k, h_dim,
                                       mup_h_base, warmup_steps, trajectory, checkpoints,
                                       n_skipped, init_collision, t0, timed_out=False,
                                       steps_completed=steps_completed, complete=False)
            _dump(out_path, partial)
            print(f"  [checkpoint step {step}] recovered_frac@0.9="
                  f"{res.get('recovered_frac@0.9'):.3f}  mean_cos={res.get('mean_cos'):.3f}"
                  f"  eff_rank={res['effective_rank_mean']:.3f}"
                  f"  q_collision_mean={res['q_collision']['mean']:.3f}", flush=True)

        if timeout_s is not None and time.time() - t0 > timeout_s:
            print(f"  internal timeout ({timeout_s}s) reached at step {step}; stopping early "
                  f"(partial result already written incrementally at the last checkpoint)", flush=True)
            timed_out = True
            break

    return _assemble_result(variant, cfg, steps, seed, force_rank_k, h_dim, mup_h_base,
                            warmup_steps, trajectory, checkpoints, n_skipped,
                            init_collision, t0, timed_out,
                            steps_completed=steps_completed,
                            complete=(not timed_out and steps_completed >= steps))


# ---------------------------------------------------------------------------
# Smoke gate (run FIRST, on CPU or cluster; no real training). Loops over
# EVERY variant — build requirement section 6.3: "any architecture-touching
# variant re-passes the full smoke gate (including blank-out)".
# ---------------------------------------------------------------------------

def smoke(device):
    print("=" * 60 + "\n  STAGE 0 SMOKE GATE\n" + "=" * 60)

    print("\n[1] generator + rank_utils self-tests (Task D's, unmodified)")
    td._self_test()
    Z = torch.randn(4, 16, 16, device=device)
    for k in (1, 2, 4, 8):
        er = effective_rank(truncate_to_rank(Z, k)).mean().item()
        assert er <= k + 1e-2, f"truncate_to_rank({k}) gave eff rank {er:.3f} > {k}"
    print("  OK")

    for variant in VARIANTS:
        print(f"\n[2.{variant}] forward / backward / grad-finite / force_rank_k / blank-out")
        cfg = TaskDConfig(d=8, K=4, orthogonal=True)
        gen = torch.Generator(device=device).manual_seed(1)
        h_dim = 128 if variant == "c3_mup" else 32   # h > d keeps candidate-2's QR path (n=d<=h) exercised
        model = make_model(variant, d=8, h_dim=h_dim, n_layers=2, n_heads=2, n_refine=1).to(device)
        b = sample_batch(cfg, 16, gen, device=device)

        pred, Z = model(b)
        assert pred.shape == b["targets"].shape, (pred.shape, b["targets"].shape)
        assert not torch.isnan(pred).any()
        loss = cosine_loss(pred, b["targets"])
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{variant}: no grad for {name}"
            assert not torch.isnan(p.grad).any() and not torch.isinf(p.grad).any(), \
                f"{variant}: bad grad {name}"
        print(f"  {variant}: forward {tuple(pred.shape)}, loss {loss.item():.4f}, all params finite grad")

        model.zero_grad()
        pred_k, Zk = model(b, force_rank_k=2)
        assert effective_rank(Zk).mean().item() <= 2 + 1e-2, \
            f"{variant}: force_rank_k=2 didn't constrain rank"
        cosine_loss(pred_k, b["targets"]).backward()
        for p in model.parameters():
            assert not (p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()))

        # BLANK-OUT (meaningful): decoder path touches bindings ONLY via Z —
        # mandatory re-check for architecture-touching variants (section 6.3 / attack #7).
        keys = b["keys"].clone().requires_grad_(True)
        values = b["values"].clone().requires_grad_(True)
        Zg = model.encode(keys, values)
        pred_g = model.unbind(Zg, b["query_keys"])
        gk = torch.autograd.grad(pred_g.sum(), keys, retain_graph=True, allow_unused=True)[0]
        assert gk is not None and gk.abs().sum() > 0, f"{variant}: bindings don't affect pred at all?!"
        Z_leaf = Zg.detach().clone().requires_grad_(True)
        pred_leaf = model.unbind(Z_leaf, b["query_keys"])
        g_leak = torch.autograd.grad(pred_leaf.sum(), keys, allow_unused=True)[0]
        assert g_leak is None, f"{variant}: LEAK — keys reach the decoder outside Z"
        print(f"  {variant}: force_rank_k=2 constrains rank; blank-out clean (no leak)")

    print("\n[3] baseline variant reproduces model_v4.BindingEncoder exactly (incl. bit-identity)")
    from model_v4 import BindingEncoder
    ref = BindingEncoder(d=16, h=64, n_layers=3, n_heads=4, n_refine=1)
    s0 = make_model("baseline", d=16, h_dim=64, n_layers=3, n_heads=4, n_refine=1).encoder
    ref_shapes = {n: tuple(p.shape) for n, p in ref.named_parameters()}
    s0_shapes = {n: tuple(p.shape) for n, p in s0.named_parameters()}
    assert ref_shapes == s0_shapes, f"baseline variant shape mismatch vs model_v4: {ref_shapes} vs {s0_shapes}"
    assert sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in s0.parameters())
    # Bit-identity at the SAME global seed (audit MINOR-3, auditor-verified
    # across 3 seeds; asserted here so it stays continuously verified): the
    # baseline path consumes the global RNG in exactly model_v4's sequence.
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        ref_b = BindingEncoder(d=16, h=64, n_layers=3, n_heads=4, n_refine=1)
        torch.manual_seed(seed)
        s0_b = make_model("baseline", d=16, h_dim=64, n_layers=3, n_heads=4, n_refine=1).encoder
        for (n1, p1), (n2, p2) in zip(ref_b.named_parameters(), s0_b.named_parameters()):
            assert n1 == n2 and torch.equal(p1, p2), \
                f"seed {seed}: baseline variant NOT bit-identical to model_v4 at {n1}"
    print(f"  OK: {len(ref_shapes)} param tensors match in name/shape/count AND bit-identical "
          f"values at same seed (3 seeds)")

    print("\n[4] collision + grad-norm instrumentation sanity (orthogonal init collision ~0)")
    m_normal = make_model("baseline", d=32, h_dim=64, n_layers=1, n_heads=2, n_refine=1)
    m_ortho = make_model("c2_orthogonal", d=32, h_dim=64, n_layers=1, n_heads=2, n_refine=1)
    c_normal = _pairwise_cosine_stats(m_normal.encoder.row_queries.unsqueeze(0))
    c_ortho = _pairwise_cosine_stats(m_ortho.encoder.row_queries.unsqueeze(0))
    assert c_ortho["max"] < 1e-3, f"orthogonal init not orthogonal: max pairwise cos {c_ortho['max']:.4f}"
    print(f"  normal-init max pairwise cos {c_normal['max']:.3f}  vs  "
          f"orthogonal-init max pairwise cos {c_ortho['max']:.6f} (expect ~0)")

    print("\n" + "=" * 60 + "\n  ALL STAGE 0 SMOKE CHECKS PASSED\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variant", choices=list(VARIANTS), default="baseline")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-query", type=int, default=None)
    ap.add_argument("--orthogonal", dest="orthogonal", action="store_true", default=True,
                    help="orthonormal keys+values (GATE DEFAULT, matches run_task_d.py)")
    ap.add_argument("--gaussian", dest="orthogonal", action="store_false",
                    help="near-orthogonal Gaussian keys+values (dev only)")
    ap.add_argument("--force-rank-k", type=int, default=None)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--h", type=int, default=64, dest="h_dim")
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-refine", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200,
                    help="trajectory logging cadence (section 6.2 item 1 requires <=200)")
    ap.add_argument("--ckpt-every", type=int, default=2000,
                    help="full-eval checkpoint cadence (build requirement section 6.3 requires <=2000)")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="override the variant preset's warmup length (candidate 1)")
    ap.add_argument("--mup-base-h", type=int, default=None,
                    help="override the variant preset's muP base width (candidate 3)")
    ap.add_argument("--internal-timeout", type=float, default=None,
                    help="optional self-limit (s); writes the last checkpoint's partial "
                         "result and stops gracefully instead of relying solely on the "
                         "orchestrator's external kill")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        smoke(device)
        return

    if args.log_every > 200:
        print(f"WARNING: --log-every={args.log_every} > 200: violates section 6.2 item 1's "
              f"'dense enough to date a sharp transition' requirement.", flush=True)
    if args.ckpt_every > 2000:
        print(f"WARNING: --ckpt-every={args.ckpt_every} > 2000: violates the section 6.3 "
              f"build requirement.", flush=True)
    if args.K > args.d:
        print(f"WARNING: K={args.K} > d={args.d}: no exact solution (lossy K>d regime).", flush=True)

    preset = VARIANTS[args.variant]
    cfg = TaskDConfig(d=args.d, K=args.K, n_query=args.n_query, orthogonal=args.orthogonal)
    model = make_model(args.variant, args.d, args.h_dim, args.n_layers, args.n_heads,
                       args.n_refine).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    warmup_steps = _resolve_warmup(preset["warmup_steps"], args.steps, args.warmup_steps)
    mup_h_base = args.mup_base_h if args.mup_base_h is not None else preset["mup_h_base"]

    print(f"variant={args.variant} d={args.d} K={args.K} h={args.h_dim} "
          f"orthogonal={args.orthogonal} force_rank_k={args.force_rank_k} "
          f"warmup_steps={warmup_steps} mup_h_base={mup_h_base} "
          f"params={n_params/1e6:.3f}M device={device}", flush=True)

    result = train(model, cfg, device, args.variant, steps=args.steps,
                   batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                   force_rank_k=args.force_rank_k, log_every=args.log_every,
                   ckpt_every=args.ckpt_every, out_path=args.out,
                   warmup_steps=warmup_steps, mup_h_base=mup_h_base, h_dim=args.h_dim,
                   timeout_s=args.internal_timeout)
    result["n_params"] = n_params

    summary = {k: v for k, v in result.items() if k not in ("trajectory", "checkpoints", "rankk_curve")}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
