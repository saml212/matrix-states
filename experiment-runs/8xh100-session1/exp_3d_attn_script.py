#!/usr/bin/env python3
"""
Exp 3D: Restore True Matrix Product Attention at d=16
======================================================
The original novelty: rows at position s couple with columns at position t.
Score is a MATRIX per pair, not a scalar. At d=16 this is affordable:
memory = O(L² × hd × hd) = O(L² × 4 × 4) = 16× scalar attention.
At L=512: 512² × 16 × 4 bytes = 67MB per head. 4 heads = 268MB. Feasible.

This tests whether matrix-structured attention (the thing we removed) was
actually the key ingredient we were missing.
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
    "mat_dim": 16, "n_thinking_layers": 4, "n_heads": 4, "n_probes": 16,
    "n_iterations": 4, "max_len": 2048, "dropout": 0.1,
    "batch_size_per_gpu": 16, "seq_len": 512, "max_steps": 1000,
    "lr": 3e-4, "warmup_steps": 300, "weight_decay": 0.01, "grad_clip": 1.0,
    "eval_interval": 200, "eval_batch_size": 16, "eval_iterations": [1, 2, 4],
    "log_interval": 50, "data_dir": "/toy_story_slam/data",
    "fallback_data_dir": "/toy_story_slam/data_quick",
    "results_dir": "/toy_story_slam/results/exp_3d_attn", "num_workers": 8,
}

class MatrixRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, d)); self.eps = eps
    def forward(self, M):
        return M / torch.sqrt(M.pow(2).mean(dim=(-2,-1), keepdim=True) + self.eps) * self.weight

class RowThenColProjection(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
    def forward(self, M):
        return torch.einsum('...ij,jk->...ik', F.silu(torch.einsum('ij,...jk->...ik', self.A, M)), self.B)

class Matrix3DAttention(nn.Module):
    """THE ORIGINAL NOVELTY: 3D matrix product attention.

    Q_s @ K_t^T produces a MATRIX per (s,t) pair.
    Row i of Q at position s couples with column j of K at position t.
    The trace gives a scalar weight. The matrix transforms V before aggregation.

    At d=16, n_heads=4: head_dim=4, score matrix is 4×4.
    Memory: L² × H × hd × hd × 4 bytes = 512² × 4 × 16 × 4 = 268MB. Feasible.
    """
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.d, self.n_heads, self.head_dim = d, n_heads, d // n_heads
        self.norm = MatrixRMSNorm(d)
        self.q_proj = RowThenColProjection(d)
        self.k_proj = RowThenColProjection(d)
        self.v_proj = RowThenColProjection(d)
        self.o_proj = RowThenColProjection(d)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, M):
        B, L, d, _ = M.shape
        H, hd = self.n_heads, self.head_dim

        M_n = self.norm(M)
        Q = self.q_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)  # (B,H,L,hd,d)
        K = self.k_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)
        V = self.v_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4)

        # 3D matrix product scores: Q_s @ K_t^T → (B,H,L,L,hd,hd) matrix per pair
        mat_scores = torch.einsum('bhlik,bhmjk->bhlmij', Q, K)
        # Trace for scalar weight
        scalar_scores = torch.einsum('bhlmii->bhlm', mat_scores) / math.sqrt(hd)

        # Causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=M.device, dtype=torch.bool), diagonal=1)
        scalar_scores = scalar_scores.masked_fill(causal_mask[None, None], float('-inf'))
        attn_weights = self.attn_dropout(F.softmax(scalar_scores, dim=-1))

        # Structured aggregation: matrix score transforms V before weighting
        transformed_V = torch.einsum('bhlmij,bhmjk->bhlmik', mat_scores, V)
        out = torch.einsum('bhlm,bhlmij->bhlij', attn_weights, transformed_V)

        out = out.permute(0, 2, 1, 3, 4).reshape(B, L, d, d)
        return M + self.o_proj(out)

class MultiplicativeThinkingLayer(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.norm = MatrixRMSNorm(d)
        self.delta_gate = RowThenColProjection(d); self.delta_value = RowThenColProjection(d)
        self.delta_up = RowThenColProjection(d)
        self.gamma_gate = RowThenColProjection(d); self.gamma_value = RowThenColProjection(d)
        self.gamma_up = RowThenColProjection(d)
        self.key_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.val_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.gate_mult_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_mult_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_write_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_write_bias = nn.Parameter(torch.tensor(-2.0))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('I', torch.eye(d))
    def forward(self, M):
        M_n = self.norm(M); s = self.scale.clamp(0.01, 0.5)
        delta = self.delta_up(F.silu(self.delta_gate(M_n)) * self.delta_value(M_n)) * s
        gamma = self.gamma_up(F.silu(self.gamma_gate(M_n)) * self.gamma_value(M_n)) * s
        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)
        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)
        g_m = torch.sigmoid((self.gate_mult_W * M_n).sum(dim=(-2,-1), keepdim=True) + self.gate_mult_bias)
        g_w = torch.sigmoid((self.gate_write_W * M_n).sum(dim=(-2,-1), keepdim=True) + self.gate_write_bias)
        return M + self.dropout(g_m * (M_mult - M_n) + g_w * M_write)

class ThinkingBlock(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = Matrix3DAttention(d, n_heads, dropout)  # THE KEY CHANGE
        self.think = MultiplicativeThinkingLayer(d, dropout)
    def forward(self, M): return self.think(self.attn(M))

class MultiProbeHead(nn.Module):
    def __init__(self, d, vocab_size, n_probes=None):
        super().__init__()
        K = n_probes or d
        self.U = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.V = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.out = nn.Linear(K, vocab_size, bias=False)
    def forward(self, M):
        MV = torch.einsum('blij, kj -> blik', M, self.V)
        return self.out(torch.einsum('ki, blik -> blk', self.U, MV))

class MatrixThinker3D(nn.Module):
    def __init__(self, mat_dim=16, n_layers=12, n_heads=4, n_probes=16,
                 max_len=2048, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim; d = mat_dim
        self.embed_u = nn.Embedding(vocab_size, d); self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d); self.pos_v = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([ThinkingBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.final_norm = MatrixRMSNorm(d)
        self.output_head = MultiProbeHead(d, vocab_size, n_probes)
    def _one_iteration(self, M):
        for layer in self.layers:
            if self.training:
                M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else:
                M = layer(M)
        return M
    def forward(self, token_ids, n_iterations=1, measure_ranks=False):
        B, L = token_ids.shape; d = self.mat_dim
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1
        ranks = []
        for t in range(n_iterations):
            M = self._one_iteration(M)
            if measure_ranks:
                with torch.no_grad():
                    Mf = M.detach().float().reshape(-1, d, d)
                    if Mf.shape[0] > 512: Mf = Mf[torch.randperm(Mf.shape[0], device=Mf.device)[:512]]
                    S = torch.linalg.svdvals(Mf).clamp(min=1e-10); Sn = S / S.sum(-1, keepdim=True)
                    ranks.append((-(Sn * Sn.log()).sum(-1)).exp().mean().item())
        return self.output_head(self.final_norm(M)), {'ranks': ranks}
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
    all_ranks = []
    with torch.no_grad():
        for vx, vy in loader:
            if n_batches >= max_eval_batches: break
            n_batches += 1; vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, info = model(vx, n_iterations=n_iterations, measure_ranks=True)
            n_tok = vy.numel()
            total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1)).item() * n_tok
            total_tokens += n_tok
            if info['ranks']: all_ranks.append(info['ranks'])
    avg_loss = total_loss / max(total_tokens, 1); avg_bpb = avg_loss / math.log(2) / 3.7
    avg_ranks = []
    if all_ranks:
        n_r = len(all_ranks[0]); avg_ranks = [sum(r[i] for r in all_ranks) / len(all_ranks) for i in range(n_r)]
    model.train()
    return avg_loss, math.exp(min(avg_loss, 20)), avg_bpb, avg_ranks

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
    logger.log(f"Exp 3D: Matrix Thinker with TRUE 3D attention at d=16 | GPUs: {world_size}")
    data_dir = cfg["data_dir"]
    if not os.path.exists(os.path.join(data_dir, "meta.json")): data_dir = cfg["fallback_data_dir"]
    meta = json.load(open(os.path.join(data_dir, "meta.json"))); vocab_size = meta["vocab_size"]
    logger.log(f"Data: {meta.get('train_tokens', '?'):,} tokens")

    train_ds = TokenDataset(os.path.join(data_dir, "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(data_dir, "val.pt"), cfg["seq_len"])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=sampler, drop_last=True, num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)

    raw_model = MatrixThinker3D(mat_dim=cfg["mat_dim"], n_layers=cfg["n_thinking_layers"],
        n_heads=cfg["n_heads"], n_probes=cfg["n_probes"], max_len=cfg["max_len"],
        vocab_size=vocab_size, dropout=cfg["dropout"]).to(device)
    n_params = raw_model.count_params()
    tp = sum(p.numel() for n,p in raw_model.named_parameters() if 'layers' in n)
    logger.log(f"Params: {n_params:,} (think={tp:,})")
    logger.log(f"3D attention: score is {cfg['mat_dim']//cfg['n_heads']}x{cfg['mat_dim']//cfg['n_heads']} matrix per pair (not scalar)")

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
        do_log = (step + 1) % cfg["log_interval"] == 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, info = model(x, n_iterations=cfg["n_iterations"], measure_ranks=do_log)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step(); scheduler.step(); step += 1
        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time; bpb = loss.item() / math.log(2) / 3.7
            tps = step * cfg["batch_size_per_gpu"] * world_size * cfg["seq_len"] / elapsed
            ranks = info.get('ranks', []); rank_str = ", ".join(f"{r:.2f}" for r in ranks) if ranks else "-"
            logger.log(f"Step {step:5d} | BPB {bpb:.3f} | PPL {math.exp(min(loss.item(),20)):8.1f} | GN {grad_norm:.3f} | Rank [{rank_str}] | {tps:.0f} tok/s | {elapsed:.0f}s")
        if step % cfg["eval_interval"] == 0:
            if is_main:
                logger.log(f"--- Eval step {step} ---"); eval_results = {}
                for T in cfg["eval_iterations"]:
                    vl, vp, vbpb, vr = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
                    rank_str = ", ".join(f"{r:.2f}" for r in vr) if vr else "-"; marker = ""
                    if T == cfg["n_iterations"] and vl < best_val_loss:
                        best_val_loss = vl; torch.save(raw_model.state_dict(), results_dir / "best.pt"); marker = " *BEST*"
                    logger.log(f"  T={T:2d}: BPB {vbpb:.3f} | Rank [{rank_str}]{marker}")
                    eval_results[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "val_bpb": vbpb, "ranks": vr}
                training_curve.append({"step": step, "train_loss": loss.item(), "evals": eval_results})
            dist.barrier()

    elapsed = time.time() - start_time
    if is_main:
        logger.log(f"\n{'='*70}"); logger.log(f"DONE | {step} steps | {elapsed/60:.1f} min")
        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        final = {}
        for T in cfg["eval_iterations"]:
            vl, vp, vbpb, vr = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
            rank_str = ", ".join(f"{r:.2f}" for r in vr) if vr else "-"
            logger.log(f"  T={T:2d}: BPB {vbpb:.3f} | Rank [{rank_str}]")
            final[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "val_bpb": vbpb, "ranks": vr}
        bpbs = {T: final[f"T{T}"]["val_bpb"] for T in cfg["eval_iterations"]}
        t1, tn = cfg["eval_iterations"][0], cfg["eval_iterations"][-1]
        summary = "\n".join([
            "", "="*70, "  EXP 3D: TRUE MATRIX PRODUCT ATTENTION d=16", "="*70,
            f"  3D attention: score is 4x4 matrix per pair (rows couple with cols)",
            f"  Params: {n_params:,} (think={tp:,})",
            f"  Time: {elapsed/60:.1f} min on {world_size}x H100", "",
            *[f"  T={T:2d}: BPB {bpbs[T]:.3f}" for T in cfg["eval_iterations"]], "",
            f"  Thinking: BPB {bpbs[t1]:.3f} -> {bpbs[tn]:.3f}",
            f"  vs Frobenius d=16 v2: BPB 1.906 at T=8",
            f"  vs d=32 Round 2:      BPB 1.670 at T=8",
            f"  KEY QUESTION: does 3D attention beat Frobenius at d=16?",
            "="*70, ""])
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"experiment": "exp_3d_attn", "params": n_params,
                    "config": cfg, "final_evals": final, "training_curve": training_curve,
                    "time_min": elapsed/60}, open(results_dir / "results.json", 'w'), indent=2, default=float)
        logger.log(f"Saved to {results_dir}"); logger.close()
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": run()
