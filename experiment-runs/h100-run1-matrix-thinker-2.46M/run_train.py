import torch, sys, time, math, json
import torch.nn.functional as F

sys.path.insert(0, "/root/src")
from matrix_thinker import AutoregressiveMatrixThinker

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len):
        self.data = torch.load(path, weights_only=True)
        self.seq_len = seq_len
        self.n = len(self.data) // seq_len - 1
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i * self.seq_len
        c = self.data[s:s+self.seq_len+1]
        return c[:-1], c[1:]

meta = json.load(open("/root/data/reasoning/meta.json"))
vocab_size = meta["vocab_size"]

train_ds = TokenDataset("/root/data/reasoning/train.pt", 256)
val_ds = TokenDataset("/root/data/reasoning/val.pt", 256)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, drop_last=True)

model = AutoregressiveMatrixThinker(
    mat_dim=16, n_thinking_layers=2, max_thoughts=20,
    n_heads=4, max_len=1024, vocab_size=vocab_size, dropout=0.1
).cuda()

print(f"Model: {model.count_parameters():,} params")
print(f"Data: {meta['train_tokens']:,} tokens")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Training with bfloat16 autocast, 8 thinking steps")
print()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.98))
max_steps = 5000
warmup = 200

def lr_lambda(step):
    if step < warmup: return step / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / (max_steps - warmup)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

step = 0
start = time.time()
best_val = float("inf")
data_iter = iter(train_loader)
model.train()

while step < max_steps:
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x, y = next(data_iter)
    x, y = x.cuda(), y.cuda()

    optimizer.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, info = model(x, n_thoughts=8)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    step += 1

    if step % 50 == 0:
        elapsed = time.time() - start
        ppl = math.exp(min(loss.item(), 20))
        ranks = info["mean_ranks_per_step"]
        rank_str = " ".join([f"{r:.2f}" for r in ranks])
        solidifying = "YES" if len(ranks) >= 2 and ranks[-1] < ranks[0] else "no"
        print(f"[{elapsed:6.0f}s] Step {step:5d} | Loss {loss.item():.3f} | PPL {ppl:.0f} | Ranks [{rank_str}] | Solid: {solidifying}")

    if step % 500 == 0:
        model.eval()
        vl = 0; vn = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                if vn >= 20: break
                vx, vy = vx.cuda(), vy.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vlogits, _ = model(vx, n_thoughts=8)
                vl += F.cross_entropy(vlogits.reshape(-1, vocab_size), vy.reshape(-1)).item()
                vn += 1
        val_loss = vl / max(vn, 1)
        val_ppl = math.exp(min(val_loss, 20))
        marker = " *" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "/root/results/best.pt")
        print(f"  Val Loss {val_loss:.3f} | Val PPL {val_ppl:.0f}{marker}")
        model.train()

elapsed = time.time() - start
print(f"\nDone: {step} steps in {elapsed/60:.0f} min | Best Val PPL: {math.exp(min(best_val, 20)):.0f}")
results = {"params": model.count_parameters(), "best_val_loss": best_val,
           "best_val_ppl": math.exp(min(best_val, 20)), "steps": step, "time_s": elapsed}
json.dump(results, open("/root/results/result.json", "w"), indent=2, default=float)
