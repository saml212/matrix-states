#!/usr/bin/env python3
"""
Outer-Product Embedding in LoopFormer at 10M+ Params
=====================================================
THE TEST: Does outer-product matrix embedding improve a standard architecture?

Model A: LoopFormer with outer-product embedding (d=16 → flatten to 256-dim)
Model B: LoopFormer with standard embedding (256-dim flat vector)

Same total params (~10M). Same data. Same training. Same everything.
Only difference: how bytes become numbers.

If A beats B → outer-product embedding is a real, portable improvement.
If A ≈ B → the T=1 advantage was from matrix operations, not embedding.
If A worse → outer-product is actually harmful at scale.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math, time, json, os, shutil, argparse
from pathlib import Path
import datetime as dt_module
from datetime import datetime, timezone

BASE_CONFIG = {
    "d_model": 256, "n_layers": 12, "n_heads": 8, "ff_mult": 4,
    "n_loops": 4,               # LoopFormer: shared layers applied 4 times
    "max_len": 2048, "dropout": 0.1,
    "batch_size_per_gpu": 128, "seq_len": 512, "max_steps": 5000,
    "lr": 3e-4, "warmup_steps": 500, "weight_decay": 0.01, "grad_clip": 1.0,
    "eval_interval": 500, "eval_batch_size": 128, "eval_loops": [1, 2, 4],
    "log_interval": 50, "data_dir": "/toy_story_slam/data_bytes",
    "num_workers": 8, "max_eval_batches": 50,
}

# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════

class OuterProductEmbed(nn.Module):
    """Byte → two d-dim vectors → outer product → d×d matrix → flatten to d²-dim."""
    def __init__(self, d, vocab_size, max_len):
        super().__init__()
        self.d = d
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d * d)  # position in d²-dim space directly

    def forward(self, token_ids):
        B, L = token_ids.shape
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)  # (B, L, d, d)
        h = M.reshape(B, L, self.d * self.d)         # flatten to (B, L, d²)
        pos = self.pos(torch.arange(L, device=token_ids.device))
        return h + pos * 0.1

class StandardEmbed(nn.Module):
    """Byte → d_model-dim vector. Standard embedding."""
    def __init__(self, d_model, vocab_size, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, token_ids):
        B, L = token_ids.shape
        h = self.embed(token_ids)
        pos = self.pos(torch.arange(L, device=token_ids.device))
        return h + pos

# ═══════════════════════════════════════════════════════════════════════════
# LOOPFORMER (from the ICLR 2026 paper, simplified)
# ═══════════════════════════════════════════════════════════════════════════

class TimestepEmbedder(nn.Module):
    """Tells each loop iteration where it is in the trajectory."""
    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.freq_dim = freq_dim

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))

class LoopFormerBlock(nn.Module):
    """Transformer block with AdaLN timestep conditioning."""
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 4 * d_model, bias=True))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c):
        shift_a, scale_a, shift_f, scale_f = self.adaLN(c).chunk(4, dim=-1)
        # Conditioned attention
        h = self.norm1(x) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        B, L, D = h.shape; H = self.n_heads; hd = self.head_dim
        qkv = self.attn_qkv(h).reshape(B, L, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        x = x + self.attn_out(attn.transpose(1, 2).reshape(B, L, D))
        # Conditioned FFN
        h = self.norm2(x) * (1 + scale_f.unsqueeze(1)) + shift_f.unsqueeze(1)
        x = x + self.ffn(h)
        return x

class LoopFormer(nn.Module):
    def __init__(self, embed_type='outer_product', d_model=256, mat_dim=16,
                 n_layers=12, n_heads=8, ff_mult=4, n_loops=4,
                 max_len=2048, vocab_size=256, dropout=0.1):
        super().__init__()
        self.n_loops = n_loops
        self.embed_type = embed_type

        if embed_type == 'outer_product':
            self.embed = OuterProductEmbed(mat_dim, vocab_size, max_len)
        else:
            self.embed = StandardEmbed(d_model, vocab_size, max_len)

        ff_dim = d_model * ff_mult
        self.blocks = nn.ModuleList([LoopFormerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.time_embed = TimestepEmbedder(d_model)
        self.dt_embed = TimestepEmbedder(d_model)
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, std=0.02)

    def _one_loop(self, x, c):
        for block in self.blocks:
            x = block(x, c)
        return x

    def forward(self, token_ids, n_loops=None):
        n_loops = n_loops or self.n_loops
        x = self.drop(self.embed(token_ids))
        steps = [1.0 / n_loops] * n_loops
        ti = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for dt in steps:
            dt_base = torch.ones_like(ti) * dt
            c = self.time_embed(ti) + self.dt_embed(dt_base)
            if self.training and n_loops > 1:
                x = torch.utils.checkpoint.checkpoint(self._one_loop, x, c, use_reentrant=False)
            else:
                x = self._one_loop(x, c)
            ti = ti + dt
        return self.lm_head(self.norm(x)), {}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ═══════════════════════════════════════════════════════════════════════════
# DATA + EVAL + TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len):
        self.data = torch.load(path, weights_only=True); self.seq_len = seq_len
        self.n = len(self.data) // seq_len - 1
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i * self.seq_len; c = self.data[s:s+self.seq_len+1]; return c[:-1], c[1:]

def evaluate(model, val_ds, device, n_loops, batch_size=128, max_batches=50):
    model.eval()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    total_loss, total_tokens, n_batches = 0, 0, 0
    with torch.no_grad():
        for vx, vy in loader:
            if n_batches >= max_batches: break
            n_batches += 1; vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(vx, n_loops=n_loops)
            total_loss += F.cross_entropy(logits.reshape(-1, 256), vy.reshape(-1)).item() * vy.numel()
            total_tokens += vy.numel()
    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return avg_loss, avg_loss / math.log(2)  # loss, BPB

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", choices=["outer_product", "standard"], required=True)
    args = parser.parse_args()

    cfg = BASE_CONFIG.copy()
    cfg["embed_type"] = args.embed
    cfg["results_dir"] = f"/toy_story_slam/results/loopformer_{args.embed}"

    dist.init_process_group("nccl", timeout=dt_module.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank); device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    results_dir = Path(cfg["results_dir"])
    if is_main: results_dir.mkdir(parents=True, exist_ok=True); shutil.copy2(Path(__file__).resolve(), results_dir / "script.py")

    class Logger:
        def __init__(self, path): self.f = open(path, 'w') if is_main else None
        def log(self, msg):
            if not is_main: return
            line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"; print(line, flush=True)
            if self.f: self.f.write(line + '\n'); self.f.flush()
        def close(self):
            if self.f: self.f.close()

    dist.barrier(); logger = Logger(results_dir / "train.log")
    logger.log(f"LoopFormer + {args.embed} embedding | GPUs: {world_size}")

    data_dir = cfg["data_dir"]
    meta = json.load(open(os.path.join(data_dir, "meta.json"))); vocab_size = meta["vocab_size"]
    logger.log(f"Data: {meta.get('train_tokens', '?'):,} bytes")

    train_ds = TokenDataset(os.path.join(data_dir, "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(data_dir, "val.pt"), cfg["seq_len"])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=sampler, drop_last=True, num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)

    raw_model = LoopFormer(
        embed_type=args.embed, d_model=cfg["d_model"], mat_dim=16,
        n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], ff_mult=cfg["ff_mult"],
        n_loops=cfg["n_loops"], max_len=cfg["max_len"], vocab_size=vocab_size, dropout=cfg["dropout"]
    ).to(device)

    n_params = raw_model.count_params()
    embed_p = sum(p.numel() for n, p in raw_model.named_parameters() if 'embed' in n)
    layer_p = sum(p.numel() for n, p in raw_model.named_parameters() if 'block' in n)
    logger.log(f"Params: {n_params:,} (embed={embed_p:,} layers={layer_p:,})")
    logger.log(f"Architecture: d={cfg['d_model']}, {cfg['n_layers']} layers × {cfg['n_loops']} loops, {cfg['n_heads']} heads")

    model = DDP(raw_model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    def lr_lambda(step):
        if step < cfg["warmup_steps"]: return step / cfg["warmup_steps"]
        return 0.5 * (1 + math.cos(math.pi * (step - cfg["warmup_steps"]) / max(cfg["max_steps"] - cfg["warmup_steps"], 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, best_val_loss, start_time = 0, float("inf"), time.time()
    training_curve, data_iter, epoch = [], iter(train_loader), 0
    model.train(); logger.log("--- Training starts ---")

    while step < cfg["max_steps"]:
        try: x, y = next(data_iter)
        except StopIteration: epoch += 1; sampler.set_epoch(epoch); data_iter = iter(train_loader); x, y = next(data_iter)
        x, y = x.to(device), y.to(device); optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, n_loops=cfg["n_loops"])
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step(); scheduler.step(); step += 1
        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time; bpb = loss.item() / math.log(2)
            tps = step * cfg["batch_size_per_gpu"] * world_size * cfg["seq_len"] / elapsed
            logger.log(f"Step {step:5d} | BPB {bpb:.3f} | GN {grad_norm:.3f} | {tps:.0f} tok/s | {elapsed:.0f}s")
        if step % cfg["eval_interval"] == 0:
            if is_main:
                logger.log(f"--- Eval step {step} ---")
                eval_results = {}
                for L in cfg["eval_loops"]:
                    vl, vbpb = evaluate(raw_model, val_ds, device, L, cfg["eval_batch_size"], cfg["max_eval_batches"])
                    marker = ""
                    if L == cfg["n_loops"] and vl < best_val_loss:
                        best_val_loss = vl; torch.save(raw_model.state_dict(), results_dir / "best.pt"); marker = " *BEST*"
                    logger.log(f"  L={L}: BPB {vbpb:.3f}{marker}")
                    eval_results[f"L{L}"] = {"loss": vl, "bpb": vbpb}
                training_curve.append({"step": step, "loss": loss.item(), "evals": eval_results})
            dist.barrier()

    elapsed = time.time() - start_time
    if is_main:
        logger.log(f"\n{'='*70}"); logger.log(f"DONE | {step} steps | {elapsed/60:.1f} min")
        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        final = {}
        for L in cfg["eval_loops"]:
            vl, vbpb = evaluate(raw_model, val_ds, device, L, cfg["eval_batch_size"], cfg["max_eval_batches"])
            logger.log(f"  L={L}: BPB {vbpb:.3f}")
            final[f"L{L}"] = {"loss": vl, "bpb": vbpb}
        bpbs = {L: final[f"L{L}"]["bpb"] for L in cfg["eval_loops"]}
        summary = "\n".join([
            "", "="*70,
            f"  LOOPFORMER + {args.embed.upper()} EMBEDDING",
            "="*70,
            f"  Params: {n_params:,} (embed={embed_p:,} layers={layer_p:,})",
            f"  d={cfg['d_model']}, {cfg['n_layers']} layers × {cfg['n_loops']} loops",
            f"  Time: {elapsed/60:.1f} min on {world_size}x H100", "",
            *[f"  L={L}: BPB {bpbs[L]:.3f}" for L in cfg["eval_loops"]],
            "", f"  Loop benefit: BPB {bpbs[1]:.3f} → {bpbs[cfg['n_loops']]:.3f}",
            "="*70, ""])
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"experiment": f"loopformer_{args.embed}", "embed_type": args.embed,
                    "params": n_params, "param_breakdown": {"embed": embed_p, "layers": layer_p},
                    "config": cfg, "final_evals": final, "training_curve": training_curve,
                    "time_min": elapsed/60},
                   open(results_dir / "results.json", 'w'), indent=2, default=float)
        logger.close()
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": run()
