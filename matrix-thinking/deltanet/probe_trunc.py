"""Skip-rate + per-step wall-clock probe (audit round-1 FINDINGS 1c + 3).

One (d, K, fr, impl) cell per invocation, single GPU (pass
CUDA_VISIBLE_DEVICES explicitly -- check nvidia-smi first, this box runs
concurrent waves). Replicates run_deltanet.train()'s exact step logic
(sample -> forward -> cosine loss -> backward -> finite-grad check ->
skip-or-step) minus checkpoint evals, so per-step time is the pure
training-step cost and the skip rate is measured under identical numerics
to a real run. The measured results this probe produced (2026-07-02, GPU 7)
are baked into run_deltanet_sweep.py's _PER_STEP_S constants -- see the
table + methodology comment there; this script is kept so the audit (and
any future re-pricing) can re-run the measurement rather than trust it.

Usage:
  CUDA_VISIBLE_DEVICES=7 python probe_trunc.py --d 64  --K 32                              # unconstrained timing
  CUDA_VISIBLE_DEVICES=7 python probe_trunc.py --d 64  --K 32 --fr 32 --impl eigh
  CUDA_VISIBLE_DEVICES=7 python probe_trunc.py --d 64  --K 32 --fr 32 --impl svd_lowrank
  CUDA_VISIBLE_DEVICES=7 python probe_trunc.py --d 128 --K 64 --fr 64 --impl eigh
  CUDA_VISIBLE_DEVICES=7 python probe_trunc.py --d 128 --K 64 --fr 64 --impl svd_lowrank
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
import task_dn as tdn
from model_dn import DeltaNetGateModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--fr", type=int, default=None)
    ap.add_argument("--impl", choices=["eigh", "svd_lowrank"], default="eigh")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10,
                     help="steps excluded from the per-step timing (CUDA warmup)")
    args = ap.parse_args()

    dev = "cuda"
    torch.manual_seed(args.seed)
    cfg = tdn.DeltaNetTaskConfig(d=args.d, K=args.K)
    model = DeltaNetGateModel(d=args.d, trunc_impl=args.impl).to(dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    n_skip = 0
    n_skip_at_150 = None
    t0 = None
    for step in range(1, args.steps + 1):
        if step == args.warmup + 1:
            torch.cuda.synchronize()
            t0 = time.time()
        b = tdn.sample_batch(cfg, args.batch_size, gen, hop_set=cfg.H_train, device=dev,
                             assert_injective=(step == 1))
        pred, S_T = model(b, force_rank_k=args.fr)
        loss = (1.0 - F.cosine_similarity(pred, b["targets"], dim=-1)).mean()
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skip += 1
        if step == 150 and args.steps > 150:
            n_skip_at_150 = n_skip
    torch.cuda.synchronize()
    per_step_ms = (time.time() - t0) / (args.steps - args.warmup) * 1000.0

    at150 = (f"{n_skip_at_150}/150 = {n_skip_at_150 / 150:.2%}"
             if n_skip_at_150 is not None else "n/a")
    print(f"PROBE d={args.d} K={args.K} fr={args.fr} impl={args.impl} "
          f"steps={args.steps} batch={args.batch_size} seed={args.seed}: "
          f"skip@150 {at150} | skip@{args.steps} {n_skip}/{args.steps} = {n_skip/args.steps:.2%} "
          f"| per-step {per_step_ms:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
