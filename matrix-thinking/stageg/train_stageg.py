"""Stage G Regime-2 training entry point — see STAGE_G_DESIGN.md sections
6 (instrumentation) and 7 (manifest). Single-GPU (or CPU, for local
smoke), one (family, variant, seed, steps) cell per invocation, matching
run_stage0.py's "one cell per process" convention so run_stageg_sweep.py
can pack many cells across the 8 GPUs.

Per-checkpoint instrumentation (design section 6.2/6.3, required):
  trajectory   dense (<=200-step default) loss / BPB / grad-norm log,
               PLUS step-0 residual-stream entry-std (H_j free datum,
               section 4 item 1 / 6.2 item 2) logged once before any
               optimizer step.
  checkpoints  <=500-step default full eval at every T in
               --eval-iterations (mirrors round2_matrix_script.py's
               T-sweep, design section 7's provisional standard cell).
Both written INCREMENTALLY with complete=False; only the final
post-training write carries complete=True (Stage 0's crash-safety
sentinel, reused verbatim per this codebase's house style — see
run_stage0.py's `_assemble_result` docstring).

Usage:
  python train_stageg.py --smoke
  python train_stageg.py --family matrix --variant baseline --steps 3000 \
      --seed 0 --out results/w0r2_matrix_baseline_s0.json
  python train_stageg.py --family vector --variant baseline --steps 3000 \
      --seed 0 --out results/w0r2_vector_baseline_s0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MatrixModelSpec, VectorModelSpec, VARIANT_AXES, VECTOR_VARIANT_AXES
from data import train_val_datasets, resolve_data_path, VOCAB_SIZE
from flops import analytic_flops_for_cell

LOG2 = math.log(2)

# NOT IMPLEMENTED (pre-registered, audit 2026-07-02 MAJOR-3 disposition):
# design section 6.3 item 4's marginal-value-per-FLOP knockout probe
# (eval-time component replacement with cheap proxies + value-density
# computation) is deliberately not built in this pass -- it is analysis-
# stage instrumentation, not launch-blocking, and lands as a follow-on.
# Items 1-3 of section 6.3 (per-iteration rank trajectory, per-group grad
# norms, per-run analytic FLOPs) ARE implemented below.


def _autocast_ctx(device):
    """bf16 autocast on CUDA (matches round2_matrix_script.py /
    loopformer_96K_script.py's own `torch.autocast("cuda",
    dtype=torch.bfloat16)` convention) -- a no-op context on CPU/MPS, where
    bf16 autocast either isn't supported the same way or isn't the
    house-standard smoke path."""
    if isinstance(device, torch.device):
        device = device.type
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_model(family: str, variant: str, mat_dim: int, max_len: int, vocab_size: int,
                n_iterations: int):
    if family == "matrix":
        return MatrixModelSpec(mat_dim=mat_dim, n_heads=8, n_iterations=n_iterations,
                                max_len=max_len, vocab_size=vocab_size, variant=variant).build()
    elif family == "vector":
        # n_embd=80: MEASURED param match to the d=32 matrix baseline
        # (audit MAJOR-2) -- 300,976 vs 290,328 (ratio 0.965); the original
        # n_embd=64 guess was 30.5% under. See VectorReferenceModel's docstring.
        return VectorModelSpec(n_embd=80, n_head=4, n_layer=2, n_loops=n_iterations,
                                intermediate_dim=128, max_len=max_len, vocab_size=vocab_size,
                                variant=variant).build()
    raise ValueError(family)


# ---------------------------------------------------------------------------
# MAJOR-1: tau gets its own param group (excluded from the shared clip)
# ---------------------------------------------------------------------------

def split_tau_param_group(model, lr, tau_lr_mult=0.1):
    """Audit MAJOR-1: TiedBilinearHead's temperature gradient is a sum
    over B*L*vocab correlated logit terms -- measured at 99.6% of the
    model's TOTAL grad-norm (|grad(tau)|=81.03 of 81.26), which forced the
    shared clip_grad_norm_ coefficient to ~0.012 vs ~0.28 for baseline and
    silently shrank backbone updates to ~7.4% of baseline at identical
    nominal clip. Fix: log_tau (see tied_heads.py) lives in its OWN param
    group at a 10x lower LR and is clipped SEPARATELY (never inside the
    shared group).

    Returns (param_groups, main_params, tau_params) -- main_params is what
    the training loop passes to the shared clip_grad_norm_; tau_params get
    their own clip call."""
    tau_params, main_params = [], []
    for name, p in model.named_parameters():
        (tau_params if name.endswith("log_tau") else main_params).append(p)
    groups = [{"params": main_params, "lr": lr}]
    if tau_params:
        # weight_decay=0 (auditor-recommended hygiene): decaying log_tau
        # toward 0 silently pulls tau toward 1, drifting away from its
        # calibrated value for no regularization benefit on a scalar.
        groups.append({"params": tau_params, "lr": lr * tau_lr_mult, "weight_decay": 0.0})
    return groups, main_params, tau_params


# ---------------------------------------------------------------------------
# MAJOR-3(a): per-parameter-group grad norms (design section 6.3 item 3)
# ---------------------------------------------------------------------------

def _param_group(name: str, family: str) -> str:
    """Maps a parameter name to the design's four named groups (embed /
    attn-proj / thinking-MLP / output-head) plus 'cond' for the H_i /
    LoopFormer conditioning modules."""
    if family == "matrix":
        if name.startswith("embed."):
            return "embed"
        if ".attn." in name:
            return "attn_proj"
        if ".think." in name:
            return "thinking_mlp"
        if name.startswith("output_head.") or name.startswith("final_norm."):
            return "output_head"
        return "cond"          # time/dt embedders + per-layer adaLN riders
    else:
        if name.startswith("embed.") or name.startswith("lm_head."):
            return "embed"     # tied: embed IS the output head on this side
        if ".attn." in name:
            return "attn_proj"
        if ".mlp." in name:
            return "thinking_mlp"
        if name.startswith("norm_f."):
            return "output_head"
        return "cond"


GRAD_GROUPS = ("embed", "attn_proj", "thinking_mlp", "output_head", "cond")


def _group_grad_norms(model, family: str) -> dict:
    sq = {g: 0.0 for g in GRAD_GROUPS}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        sq[_param_group(name, family)] += p.grad.detach().float().pow(2).sum().item()
    return {g: round(v ** 0.5, 6) for g, v in sq.items()}


def calibrate_tied_bilinear_if_needed(model, family, vocab_size, max_len, n_iterations, device):
    """N3 (design section 4 item 4): the tied_bilinear head (H_f form i)
    MUST have its temperature calibrated from a real step-0 backbone
    forward pass before training starts -- confirmed empirically necessary
    on this build (an uncalibrated tiny GPU smoke cell showed loss ~14 /
    grad-norm ~35 at step 1 vs ~5.6/~1 for every other variant at the
    identical tiny config, exactly the H_f x H_j confound N3 predicts).
    No-op for every other head/family (duck-typed on
    `output_head.calibrate_temperature`)."""
    head = getattr(model, "output_head", None)
    if head is None or not hasattr(head, "calibrate_temperature"):
        return None
    # eval() bracketing (audit MAJOR-1 rider): dropout active during the
    # calibration forward pass added ~4% variance to the measured std --
    # calibrate deterministically, then restore training mode.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            L = min(64, max_len)
            ids = torch.randint(0, vocab_size, (4, L), device=device)
            M_n, _ = model.backbone(ids, n_iterations=n_iterations)
        measured_raw_std = head.calibrate_temperature(M_n)
    finally:
        if was_training:
            model.train()
    return measured_raw_std


def step0_entry_std(model, family: str, vocab_size: int, device) -> float:
    """H_j's free step-0 diagnostic (design section 4 item 1): residual-
    stream entry-std BEFORE any optimizer step, measured on real sampled
    ids (not hand-derived)."""
    with torch.no_grad():
        if family == "matrix":
            return model.embed.entry_std(device=device)
        else:
            # VectorEmbedding expects (B,L) and its positional table (wpe,
            # or pos_u/pos_v for the rank1 mirror) is sized to max_len --
            # cap the diagnostic sample length so this never exceeds it.
            pos_table = getattr(model.embed, "wpe", None) or getattr(model.embed, "pos_u", None)
            L = min(256, pos_table.num_embeddings)
            ids = torch.randint(0, vocab_size, (1, max(1, L)), device=device)
            x = model.embed(ids)
            return x.float().std().item()


@torch.no_grad()
def evaluate(model, val_ds, vocab_size, device, n_iterations, batch_size=64, max_eval_batches=25,
             measure_ranks=False):
    """Returns (avg_loss, avg_bpb, avg_ranks). avg_ranks: per-iteration
    effective-rank trajectory averaged over eval batches (design section
    6.3 item 1, audit MAJOR-3(b)) -- [] for the vector family / when
    measure_ranks=False."""
    model.eval()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                          drop_last=True, num_workers=0)
    total_loss, total_tokens, n_batches = 0.0, 0, 0
    all_ranks = []
    for vx, vy in loader:
        if n_batches >= max_eval_batches:
            break
        n_batches += 1
        vx, vy = vx.to(device), vy.to(device)
        fwd_kwargs = {"n_iterations": n_iterations} if hasattr(model, "mat_dim") else {"n_loops": n_iterations}
        if measure_ranks:
            fwd_kwargs["measure_ranks"] = True
        with _autocast_ctx(device):
            logits, info = model(vx, **fwd_kwargs)
            # cross_entropy INSIDE autocast (audit MINOR-1): autocast
            # routes CE to fp32, whereas outside the block it would run in
            # the logits' bf16. Both original scripts computed eval CE
            # outside autocast (bf16); Stage G deliberately diverges --
            # SYMMETRICALLY for all runs, so it cancels in every
            # comparison -- because section 8's recovered_frac bars need
            # finer loss resolution than bf16 eval provides.
            n_tok = vy.numel()
            total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1)).item() * n_tok
        total_tokens += n_tok
        if info.get("ranks"):
            all_ranks.append(info["ranks"])
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    avg_ranks = []
    if all_ranks:
        n_r = len(all_ranks[0])
        avg_ranks = [round(sum(r[i] for r in all_ranks) / len(all_ranks), 4) for i in range(n_r)]
    return avg_loss, avg_loss / LOG2, avg_ranks   # BPB is direct for byte-level


def _dump(path, obj):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"  (incremental write failed, non-fatal: {e!r})", flush=True)


def train(family, variant, mat_dim, max_len, vocab_size, n_iterations, steps, seed,
          batch_size=64, seq_len=512, lr=3e-4, warmup_steps=None, weight_decay=0.01,
          grad_clip=1.0, log_every=100, ckpt_every=500, eval_iterations=(1, 2, 4, 8),
          data_path=None, out_path=None, device="cpu", internal_timeout=None):
    t0 = time.time()
    torch.manual_seed(seed)
    device = torch.device(device)

    model = build_model(family, variant, mat_dim, max_len, vocab_size, n_iterations).to(device)
    n_params = model.count_params()
    breakdown = model.param_breakdown()
    entry_std = step0_entry_std(model, family, vocab_size, device)
    tau_calib = calibrate_tied_bilinear_if_needed(model, family, vocab_size, max_len, n_iterations, device)
    tau_msg = f" tied_bilinear_raw_std={tau_calib:.3f} tau_calibrated={model.output_head.tau.item():.4f}" \
        if tau_calib is not None else ""
    print(f"family={family} variant={variant} params={n_params:,} breakdown={breakdown} "
          f"step0_entry_std={entry_std:.4f} device={device}{tau_msg}", flush=True)

    warmup_steps = warmup_steps if warmup_steps is not None else max(1, steps // 10)
    # MAJOR-1: log_tau (tied_bilinear only) gets its own param group -- 10x
    # lower LR, and EXCLUDED from the shared clip below (its scalar grad
    # was measured at 99.6% of total grad-norm, collapsing everyone else's
    # effective clip coefficient). No-op for every other variant
    # (tau_params is empty -> single group, single clip, unchanged).
    param_groups, main_params, tau_params = split_tau_param_group(model, lr)
    opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    data_path = resolve_data_path(data_path)
    train_ds, val_ds = train_val_datasets(data_path, seq_len=seq_len)
    gen = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                                drop_last=True, num_workers=0, generator=gen)

    trajectory, checkpoints = [], []
    step, steps_completed, timed_out = 0, 0, False
    data_iter = iter(train_loader)
    model.train()

    while step < steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        fwd_kwargs = {"n_iterations": n_iterations} if family == "matrix" else {"n_loops": n_iterations}
        with _autocast_ctx(device):
            logits, info = model(x, **fwd_kwargs)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        # `step` is pre-increment here; capture group norms (pre-clip, from
        # live .grad) exactly when the post-increment trajectory log fires.
        group_norms = _group_grad_norms(model, family) if ((step + 1) % log_every == 0 or step == 0) else None
        if finite:
            # Shared clip over the MAIN params only (MAJOR-1); log_tau is
            # clipped on its own so its B*L*vocab-summed scalar gradient
            # can never drag the shared coefficient down.
            grad_norm = torch.nn.utils.clip_grad_norm_(main_params, grad_clip)
            if tau_params:
                tau_grad_norm = torch.nn.utils.clip_grad_norm_(tau_params, grad_clip)
            else:
                tau_grad_norm = None
            opt.step()
        else:
            grad_norm = float("nan")
            tau_grad_norm = None
        sched.step()
        step += 1
        steps_completed = step

        if step % log_every == 0 or step == 1:
            bpb = loss.item() / LOG2
            # Key names carry the clip semantics (coordinator polish item 2):
            # trajectory group norms are PRE-clip (captured from live .grad
            # right after backward, before either clip call); the checkpoint
            # entries below are POST-clip. Misreading one as the other
            # inverts the H_j clipping-pathology diagnostic.
            entry = {"step": step, "loss": loss.item(), "bpb": bpb,
                     "grad_norm": float(grad_norm), "lr": sched.get_last_lr()[0],
                     "grad_norm_by_group_pre_clip": group_norms}
            if tau_grad_norm is not None:
                entry["tau_grad_norm"] = float(tau_grad_norm)
                entry["tau"] = float(model.output_head.tau.item())
            trajectory.append(entry)
            print(f"  step {step:6d}  loss {loss.item():.4f}  bpb {bpb:.4f}  gn {float(grad_norm):.3f}"
                  + (f"  gn_tau {float(tau_grad_norm):.3f}" if tau_grad_norm is not None else ""), flush=True)

        if step % ckpt_every == 0 or step == steps:
            evals = {}
            for T in eval_iterations:
                # measure_ranks on checkpoint evals (MAJOR-3(b), design
                # section 6.3 item 1): per-iteration effective-rank
                # trajectory -- matrix family only (vector returns []).
                vl, vbpb, vranks = evaluate(model, val_ds, vocab_size, device, T,
                                             batch_size=batch_size, measure_ranks=(family == "matrix"))
                evals[f"T{T}"] = {"val_loss": vl, "val_bpb": vbpb, "ranks": vranks}
            # POST-clip by construction: .grad still holds the last training
            # step's gradients, already scaled by that step's clip calls
            # (evaluate() runs under no_grad and never touches .grad) -- so
            # these sum to <= the nominal clip (1.0), NOT to the raw norms;
            # read the trajectory's *_pre_clip entries for raw magnitudes.
            ckpt = {"step": step, "evals": evals,
                    "grad_norm_by_group_post_clip": _group_grad_norms(model, family)}
            checkpoints.append(ckpt)
            partial = _assemble(family, variant, mat_dim, vocab_size, n_iterations, steps, seed,
                                 n_params, breakdown, entry_std, trajectory, checkpoints, t0,
                                 timed_out=False, steps_completed=steps_completed, complete=False,
                                 seq_len=seq_len)
            _dump(out_path, partial)
            print(f"  [checkpoint step {step}] " +
                  "  ".join(f"T={T}:bpb={evals[f'T{T}']['val_bpb']:.4f}" for T in eval_iterations), flush=True)

        if internal_timeout is not None and time.time() - t0 > internal_timeout:
            print(f"  internal timeout ({internal_timeout}s) reached at step {step}; stopping early", flush=True)
            timed_out = True
            break

    return _assemble(family, variant, mat_dim, vocab_size, n_iterations, steps, seed,
                      n_params, breakdown, entry_std, trajectory, checkpoints, t0, timed_out,
                      steps_completed=steps_completed,
                      complete=(not timed_out and steps_completed >= steps), seq_len=seq_len)


def _assemble(family, variant, mat_dim, vocab_size, n_iterations, steps, seed, n_params,
              breakdown, entry_std, trajectory, checkpoints, t0, timed_out, steps_completed,
              complete, seq_len=512):
    # analytic_flops_per_token (MAJOR-3(c), design build requirement (ii)):
    # per-run analytic FLOPs so any reader can re-derive "matched" under
    # tokens-seen / params / FLOPs conventions post-hoc (section 9 attack #1).
    flops = analytic_flops_for_cell(family, variant, mat_dim, seq_len, n_iterations, vocab_size)
    result = {
        "family": family, "variant": variant, "mat_dim": mat_dim, "vocab_size": vocab_size,
        "n_iterations": n_iterations, "steps": steps, "seed": seed, "seq_len": seq_len,
        "n_params": n_params, "param_breakdown": breakdown, "step0_entry_std": entry_std,
        "analytic_flops_per_token": flops,
        "steps_completed": steps_completed, "complete": complete, "timed_out": timed_out,
        "trajectory": trajectory, "checkpoints": checkpoints, "wall_s": time.time() - t0,
    }
    if checkpoints:
        final = checkpoints[-1]
        result["final_step"] = final["step"]
        result["final_evals"] = final["evals"]
    return result


def smoke(device="cpu"):
    print(f"{'=' * 60}\n  STAGE G TRAIN SMOKE (device={device}, tiny)\n{'=' * 60}")
    import tempfile
    tmp_bin = os.path.join(tempfile.gettempdir(), "stageg_smoke_bytes.bin")
    if not os.path.exists(tmp_bin) or os.path.getsize(tmp_bin) < 200_000:
        with open(tmp_bin, "wb") as f:
            f.write(bytes((i * 37) % 256 for i in range(200_000)))
    for family, variants in (("matrix", list(VARIANT_AXES)), ("vector", list(VECTOR_VARIANT_AXES))):
        for variant in variants:
            # vocab_size=256: MUST match raw-byte range (data.ByteDataset yields
            # unrestricted byte values 0-255) -- a smaller vocab_size here would
            # silently IndexError on out-of-range byte values, exactly the bug
            # this comment now heads off.
            res = train(family, variant, mat_dim=8, max_len=32, vocab_size=256, n_iterations=2,
                        steps=4, seed=0, batch_size=4, seq_len=16, log_every=2, ckpt_every=2,
                        eval_iterations=(1, 2), data_path=tmp_bin, out_path=None, device=device)
            assert res["complete"] is True
            assert res["steps_completed"] == 4
            assert len(res["checkpoints"]) >= 1
            print(f"  {family}/{variant}: OK, final_evals={res['final_evals']}")
    print(f"\nSMOKE PASSED (device={device})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--family", choices=["matrix", "vector"], default="matrix")
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--mat-dim", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--n-iterations", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-iterations", type=str, default="1,2,4,8")
    ap.add_argument("--data-path", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--internal-timeout", type=float, default=None)
    args = ap.parse_args()

    if args.smoke:
        smoke(device=args.device)
        return

    registry = VARIANT_AXES if args.family == "matrix" else VECTOR_VARIANT_AXES
    if args.variant not in registry:
        print(f"ERROR: --variant {args.variant!r} not in {args.family} registry {list(registry)}",
              file=sys.stderr)
        sys.exit(1)

    eval_iterations = tuple(int(x) for x in args.eval_iterations.split(","))
    result = train(args.family, args.variant, args.mat_dim, args.max_len, args.vocab_size,
                    args.n_iterations, args.steps, args.seed, batch_size=args.batch_size,
                    seq_len=args.seq_len, lr=args.lr, warmup_steps=args.warmup_steps,
                    log_every=args.log_every, ckpt_every=args.ckpt_every,
                    eval_iterations=eval_iterations, data_path=args.data_path, out_path=args.out,
                    device=args.device, internal_timeout=args.internal_timeout)

    summary = {k: v for k, v in result.items() if k not in ("trajectory", "checkpoints")}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2), flush=True)
    if args.out:
        _dump(args.out, result)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
