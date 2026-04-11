#!/usr/bin/env python3
"""
LoopFormer Baseline: ICLR 2026 SOTA looped transformer
=======================================================
Direct comparison against our Matrix Thinker.
Same paradigm: shared blocks applied T times with timestep conditioning.
Same data, same compute, same eval. The ONLY difference: vectors vs matrices.

Based on https://github.com/armenjeddi/loopformer (ICLR 2026)
Adapted to our training framework for apples-to-apples comparison.

Usage:
  torchrun --standalone --nproc_per_node=8 run_loopformer_baseline.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math, time, json, os, shutil
from pathlib import Path
import datetime as dt_module
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — matched to our matrix thinker
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Model — tuned to ~5M params to match matrix thinker
    "n_embd": 96,
    "n_head": 4,               # head_dim=24
    "n_layer": 2,              # 2 shared blocks
    "n_loops": 8,              # repeated 8x = 16 effective layers
    "intermediate_dim": 240,   # ~5.3M params to match matrix thinker's ~5.2M
    "max_len": 2048,
    "dropout": 0.1,
    # Training — identical to matrix thinker
    "batch_size_per_gpu": 96,     # match matrix thinker exactly
    "seq_len": 512,
    "max_steps": 3000,
    "lr": 3e-4,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    # Eval
    "eval_interval": 500,
    "eval_batch_size": 96,
    "eval_loops": [1, 2, 4, 8],   # test at different loop counts
    "log_interval": 50,
    # Paths
    "data_dir": "/toy_story_slam/data",
    "fallback_data_dir": "/toy_story_slam/data_quick",
    "results_dir": "/toy_story_slam/results/loopformer_baseline",
    "num_workers": 8,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOOPFORMER MODEL (adapted from github.com/armenjeddi/loopformer)
# ═══════════════════════════════════════════════════════════════════════════════

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.
    From LoopFormer: tells each loop iteration "you are step t of T"."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LoopFormerBlock(nn.Module):
    """Single transformer block with timestep-conditioned modulation.
    The key LoopFormer innovation: AdaLN-style conditioning tells each block
    which iteration it's on."""
    def __init__(self, n_embd, n_head, intermediate_dim, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.RMSNorm(n_embd, elementwise_affine=False)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.norm_2 = nn.RMSNorm(n_embd, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, n_embd),
            nn.Dropout(dropout),
        )
        # AdaLN modulation: timestep conditions the norms
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(n_embd, 4 * n_embd, bias=True),
        )

    def forward(self, x, c):
        # c is timestep conditioning vector (B, n_embd)
        shift_attn, scale_attn, shift_mlp, scale_mlp = self.adaLN(c).chunk(4, dim=-1)

        # Conditioned attention
        h = self.norm_1(x)
        h = h * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        x = x + self.attn(h)

        # Conditioned MLP
        h = self.norm_2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(h)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class LoopFormer(nn.Module):
    """LoopFormer: shared blocks applied T times with timestep conditioning.
    ICLR 2026 (arxiv 2602.11451)."""
    def __init__(self, n_embd=128, n_head=4, n_layer=2, n_loops=8,
                 intermediate_dim=320, max_len=2048, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.n_loops = n_loops

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(max_len, n_embd)
        self.drop = nn.Dropout(dropout)

        # Shared blocks (applied n_loops times)
        self.blocks = nn.ModuleList([
            LoopFormerBlock(n_embd, n_head, intermediate_dim, dropout)
            for _ in range(n_layer)
        ])

        # Timestep embedders (the LoopFormer innovation)
        self.time_embedder = TimestepEmbedder(n_embd)
        self.dt_embedder = TimestepEmbedder(n_embd)

        self.norm_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _one_loop(self, x, c):
        for block in self.blocks:
            x = block(x, c)
        return x

    def forward(self, token_ids, n_loops=None, **kwargs):
        n_loops = n_loops or self.n_loops
        B, L = token_ids.shape
        device = token_ids.device

        tok_emb = self.wte(token_ids)
        pos_emb = self.wpe(torch.arange(L, device=device))
        x = self.drop(tok_emb + pos_emb)

        # LoopFormer looping with timestep conditioning
        steps = [1.0 / n_loops] * n_loops
        ti = torch.zeros(B, dtype=x.dtype, device=device)
        for dt in steps:
            dt_base = torch.ones_like(ti) * dt
            c = self.time_embedder(ti) + self.dt_embedder(dt_base)
            if self.training and n_loops > 1:
                x = torch.utils.checkpoint.checkpoint(
                    self._one_loop, x, c, use_reentrant=False)
            else:
                x = self._one_loop(x, c)
            ti = ti + dt

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits, {'ranks': [], 'model_type': 'loopformer'}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA + EVAL + TRAINING (identical framework to matrix thinker)
# ═══════════════════════════════════════════════════════════════════════════════

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len):
        self.data = torch.load(path, weights_only=True)
        self.seq_len = seq_len
        self.n = len(self.data) // seq_len - 1
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i * self.seq_len
        c = self.data[s:s + self.seq_len + 1]
        return c[:-1], c[1:]


def evaluate(model, val_ds, vocab_size, device, n_loops, eval_batch_size=96, max_eval_batches=50):
    model.eval()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=eval_batch_size,
                                          shuffle=False, drop_last=True, num_workers=4)
    total_loss, total_tokens, n_batches = 0, 0, 0
    with torch.no_grad():
        for vx, vy in loader:
            if n_batches >= max_eval_batches:
                break
            n_batches += 1
            vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(vx, n_loops=n_loops)
            n_tok = vy.numel()
            total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1)).item() * n_tok
            total_tokens += n_tok
    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return avg_loss, math.exp(min(avg_loss, 20))


def run():
    cfg = CONFIG.copy()

    dist.init_process_group("nccl", timeout=dt_module.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    results_dir = Path(cfg["results_dir"])
    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), results_dir / "script.py")

    class Logger:
        def __init__(self, path):
            self.f = open(path, 'w') if is_main else None
        def log(self, msg):
            if not is_main: return
            ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            if self.f: self.f.write(line + '\n'); self.f.flush()
        def close(self):
            if self.f: self.f.close()

    dist.barrier()
    logger = Logger(results_dir / "train.log")
    logger.log(f"LoopFormer Baseline (ICLR 2026) | GPUs: {world_size}")
    logger.log(f"Config: {json.dumps(cfg, indent=2)}")

    data_dir = cfg["data_dir"]
    if not os.path.exists(os.path.join(data_dir, "meta.json")):
        data_dir = cfg["fallback_data_dir"]
        logger.log(f"Big data not ready, using fallback: {data_dir}")
    meta = json.load(open(os.path.join(data_dir, "meta.json")))
    vocab_size = meta["vocab_size"]
    logger.log(f"Data: {meta.get('train_tokens', '?'):,} train tokens")

    train_ds = TokenDataset(os.path.join(data_dir, "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(data_dir, "val.pt"), cfg["seq_len"])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=sampler,
        drop_last=True, num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)

    raw_model = LoopFormer(
        n_embd=cfg["n_embd"], n_head=cfg["n_head"], n_layer=cfg["n_layer"],
        n_loops=cfg["n_loops"], intermediate_dim=cfg["intermediate_dim"],
        max_len=cfg["max_len"], vocab_size=vocab_size, dropout=cfg["dropout"]
    ).to(device)

    n_params = raw_model.count_params()
    logger.log(f"LoopFormer params: {n_params:,}")
    logger.log(f"Architecture: {cfg['n_layer']} blocks x {cfg['n_loops']} loops = {cfg['n_layer']*cfg['n_loops']} effective layers")
    logger.log(f"n_embd={cfg['n_embd']}, n_head={cfg['n_head']}, intermediate={cfg['intermediate_dim']}")

    model = DDP(raw_model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    def lr_lambda(step):
        if step < cfg["warmup_steps"]: return step / cfg["warmup_steps"]
        p = (step - cfg["warmup_steps"]) / max(cfg["max_steps"] - cfg["warmup_steps"], 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, best_val_loss = 0, float("inf")
    start_time = time.time()
    training_curve = []
    data_iter = iter(train_loader)
    epoch = 0
    model.train()
    logger.log("--- Training starts ---")

    while step < cfg["max_steps"]:
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, n_loops=cfg["n_loops"])
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            ppl = math.exp(min(loss.item(), 20))
            tps = step * cfg["batch_size_per_gpu"] * world_size * cfg["seq_len"] / elapsed
            logger.log(f"Step {step:5d} | Loss {loss.item():.4f} | PPL {ppl:8.1f} | "
                       f"GN {grad_norm:.3f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                       f"{tps:.0f} tok/s | {elapsed:.0f}s")

        if step % cfg["eval_interval"] == 0:
            if is_main:
                logger.log(f"--- Eval step {step} ---")
                eval_results = {}
                for T in cfg["eval_loops"]:
                    vl, vp = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
                    marker = ""
                    if T == cfg["n_loops"] and vl < best_val_loss:
                        best_val_loss = vl
                        torch.save(raw_model.state_dict(), results_dir / "best.pt")
                        marker = " *BEST*"
                    logger.log(f"  Loops={T:2d}: PPL {vp:8.1f}{marker}")
                    eval_results[f"L{T}"] = {"val_loss": vl, "val_ppl": vp}
                training_curve.append({"step": step, "train_loss": loss.item(), "evals": eval_results})
            dist.barrier()

    elapsed = time.time() - start_time
    if is_main:
        logger.log(f"\n{'='*70}")
        logger.log(f"DONE: LoopFormer | {step} steps | {elapsed/60:.1f} min")

        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        final = {}
        for T in cfg["eval_loops"]:
            vl, vp = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
            logger.log(f"  Loops={T:2d}: PPL {vp:8.1f}")
            final[f"L{T}"] = {"val_loss": vl, "val_ppl": vp}

        ppls = {T: final[f"L{T}"]["val_ppl"] for T in cfg["eval_loops"]}
        t1, tn = cfg["eval_loops"][0], cfg["eval_loops"][-1]
        delta = ppls[t1] - ppls[tn]
        pct = delta / ppls[t1] * 100 if ppls[t1] > 0 else 0

        summary = "\n".join([
            "", "=" * 70,
            "  LOOPFORMER BASELINE (ICLR 2026)",
            "=" * 70,
            f"  Params: {n_params:,} | {elapsed/60:.1f} min on {world_size}x H100",
            f"  Architecture: {cfg['n_layer']} blocks x {cfg['n_loops']} loops, "
            f"n_embd={cfg['n_embd']}, timestep-conditioned",
            f"  Data: {meta.get('train_tokens', '?'):,} tokens | Steps: {step}",
            "",
            *[f"  Loops={T:2d}: PPL {ppls[T]:8.1f}" for T in cfg["eval_loops"]],
            "",
            f"  Loop benefit: {delta:.1f} PPL ({pct:.1f}%)",
            "=" * 70, ""
        ])
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"experiment": "loopformer_baseline", "params": n_params,
                    "config": cfg, "final_evals": final, "training_curve": training_curve,
                    "loop_benefit_pct": pct, "time_min": elapsed / 60},
                   open(results_dir / "results.json", 'w'), indent=2, default=float)
        logger.log(f"Saved to {results_dir}")
        logger.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
