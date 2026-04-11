#!/usr/bin/env python3
"""
CRITICAL ABLATION: Outer-product embed + standard vector ops
=============================================================
Embed tokens as d×d matrix via outer product (same as matrix thinker),
then FLATTEN to d²-dim vector and use standard transformer attention + FFN.
If this matches matrix thinker quality → the win is the EMBEDDING, not the ops.
If matrix ops beat this → the matrix OPERATIONS genuinely help.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math, time, json, os, shutil
from pathlib import Path
import datetime as dt_module
from datetime import datetime, timezone

CONFIG = {
    "mat_dim": 16, "n_layers": 12, "n_heads": 4, "n_iterations": 8,
    "max_len": 2048, "dropout": 0.1, "batch_size_per_gpu": 96, "seq_len": 512,
    "max_steps": 3000, "lr": 3e-4, "warmup_steps": 300, "weight_decay": 0.01,
    "grad_clip": 1.0, "eval_interval": 500, "eval_batch_size": 96,
    "eval_iterations": [1, 2, 4, 8], "log_interval": 50,
    "data_dir": "/toy_story_slam/data", "fallback_data_dir": "/toy_story_slam/data_quick",
    "results_dir": "/toy_story_slam/results/ablation_flat", "num_workers": 8,
}

class FlatVectorBlock(nn.Module):
    """Standard pre-norm transformer block operating on d²-dim vectors."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(), nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout))
    def forward(self, h):
        h_n = self.norm1(h)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=h.device)
        attn_out, _ = self.attn(h_n, h_n, h_n, attn_mask=mask)
        h = h + attn_out
        h = h + self.ffn(self.norm2(h))
        return h

class OuterProductFlatModel(nn.Module):
    """
    Outer-product matrix EMBEDDING + standard vector OPERATIONS.
    Embed: token → u ⊗ v → 16×16 matrix → flatten to 256-dim vector.
    Process: standard transformer layers on 256-dim vectors.
    Iterate: shared layers applied T times (same as matrix thinker).
    Output: Linear(256, vocab) — standard vector output.
    """
    def __init__(self, mat_dim=16, n_layers=12, n_heads=4,
                 max_len=2048, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim
        d = mat_dim
        d_model = d * d  # 256 for d=16
        self.d_model = d_model
        # Same outer-product embedding as matrix thinker
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)
        # Standard transformer layers on flattened d²-dim vectors
        self.layers = nn.ModuleList([FlatVectorBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def _one_iteration(self, h):
        for layer in self.layers:
            if self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return h

    def forward(self, token_ids, n_iterations=1, measure_ranks=False):
        B, L = token_ids.shape
        d = self.mat_dim
        # Outer-product embedding — IDENTICAL to matrix thinker
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1
        # FLATTEN to vector — this is what we're testing
        h = M.reshape(B, L, d * d)  # (B, L, 256)
        # Iterative refinement with shared layers
        for t in range(n_iterations):
            h = self._one_iteration(h)
        h = self.norm(h)
        logits = self.output(h)
        return logits, {'ranks': []}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len):
        self.data = torch.load(path, weights_only=True); self.seq_len = seq_len
        self.n = len(self.data) // seq_len - 1
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i * self.seq_len; c = self.data[s:s+self.seq_len+1]; return c[:-1], c[1:]

def evaluate(model, val_ds, vocab_size, device, n_iterations, eval_batch_size=96, max_eval_batches=50):
    model.eval()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=True, num_workers=4)
    total_loss, total_tokens, n_batches = 0, 0, 0
    with torch.no_grad():
        for vx, vy in loader:
            if n_batches >= max_eval_batches: break
            n_batches += 1; vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(vx, n_iterations=n_iterations)
            n_tok = vy.numel()
            total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1)).item() * n_tok
            total_tokens += n_tok
    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return avg_loss, math.exp(min(avg_loss, 20)), avg_loss / math.log(2) / 3.7

def run():
    cfg = CONFIG.copy()
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
    logger.log(f"CRITICAL ABLATION: Outer-product embed + flat vector ops | GPUs: {world_size}")
    data_dir = cfg["data_dir"]
    if not os.path.exists(os.path.join(data_dir, "meta.json")): data_dir = cfg["fallback_data_dir"]
    meta = json.load(open(os.path.join(data_dir, "meta.json"))); vocab_size = meta["vocab_size"]
    logger.log(f"Data: {meta.get('train_tokens', '?'):,} tokens")

    train_ds = TokenDataset(os.path.join(data_dir, "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(data_dir, "val.pt"), cfg["seq_len"])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=sampler, drop_last=True, num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)

    raw_model = OuterProductFlatModel(
        mat_dim=cfg["mat_dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        max_len=cfg["max_len"], vocab_size=vocab_size, dropout=cfg["dropout"]).to(device)
    n_params = raw_model.count_params()
    ep = sum(p.numel() for n,p in raw_model.named_parameters() if 'embed' in n or 'pos' in n)
    tp = sum(p.numel() for n,p in raw_model.named_parameters() if 'layers' in n)
    hp = sum(p.numel() for n,p in raw_model.named_parameters() if 'output' in n or 'norm' in n)
    logger.log(f"Params: {n_params:,} (embed={ep:,} layers={tp:,} head={hp:,})")
    logger.log(f"Architecture: outer-product embed d={cfg['mat_dim']} -> flatten to {cfg['mat_dim']**2}-dim -> {cfg['n_layers']} transformer layers x T={cfg['n_iterations']} iterations")

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
            logits, _ = model(x, n_iterations=cfg["n_iterations"])
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step(); scheduler.step(); step += 1
        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time; bpb = loss.item() / math.log(2) / 3.7
            tps = step * cfg["batch_size_per_gpu"] * world_size * cfg["seq_len"] / elapsed
            logger.log(f"Step {step:5d} | BPB {bpb:.3f} | PPL {math.exp(min(loss.item(),20)):8.1f} | GN {grad_norm:.3f} | {tps:.0f} tok/s | {elapsed:.0f}s")
        if step % cfg["eval_interval"] == 0:
            if is_main:
                logger.log(f"--- Eval step {step} ---"); eval_results = {}
                for T in cfg["eval_iterations"]:
                    vl, vp, vbpb = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
                    marker = ""
                    if T == cfg["n_iterations"] and vl < best_val_loss:
                        best_val_loss = vl; torch.save(raw_model.state_dict(), results_dir / "best.pt"); marker = " *BEST*"
                    logger.log(f"  T={T:2d}: BPB {vbpb:.3f} | PPL {vp:8.1f}{marker}")
                    eval_results[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "val_bpb": vbpb}
                training_curve.append({"step": step, "train_loss": loss.item(), "evals": eval_results})
            dist.barrier()

    elapsed = time.time() - start_time
    if is_main:
        logger.log(f"\n{'='*70}"); logger.log(f"DONE | {step} steps | {elapsed/60:.1f} min")
        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        final = {}
        for T in cfg["eval_iterations"]:
            vl, vp, vbpb = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
            logger.log(f"  T={T:2d}: BPB {vbpb:.3f} | PPL {vp:8.1f}")
            final[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "val_bpb": vbpb}
        bpbs = {T: final[f"T{T}"]["val_bpb"] for T in cfg["eval_iterations"]}
        t1, tn = cfg["eval_iterations"][0], cfg["eval_iterations"][-1]
        summary = "\n".join([
            "", "="*70, "  CRITICAL ABLATION: OUTER-PRODUCT EMBED + FLAT VECTOR OPS", "="*70,
            f"  Same embedding as matrix thinker (u ⊗ v), then FLATTEN to {cfg['mat_dim']**2}-dim",
            f"  Standard transformer layers on flat vectors",
            f"  Params: {n_params:,} (embed={ep:,} layers={tp:,} head={hp:,})",
            f"  Time: {elapsed/60:.1f} min on {world_size}x H100", "",
            *[f"  T={T:2d}: BPB {bpbs[T]:.3f}" for T in cfg["eval_iterations"]], "",
            f"  Thinking benefit: BPB {bpbs[t1]:.3f} -> {bpbs[tn]:.3f}",
            f"  vs Matrix d=16 v2:    BPB 1.906 at T=8",
            f"  vs Matrix d=32 R2:    BPB 1.670 at T=8",
            f"  vs LoopFormer FLOPs:  BPB 0.870 at T=8", "",
            "  INTERPRETATION:",
            "  If BPB ≈ matrix thinker → the win is the EMBEDDING, not the ops",
            "  If BPB >> matrix thinker → matrix OPERATIONS genuinely help",
            "="*70, ""])
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"experiment": "ablation_flat", "params": n_params,
                    "param_breakdown": {"embed": ep, "layers": tp, "head": hp},
                    "config": cfg, "final_evals": final, "training_curve": training_curve,
                    "time_min": elapsed/60}, open(results_dir / "results.json", 'w'), indent=2, default=float)
        logger.log(f"Saved to {results_dir}"); logger.close()
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": run()
