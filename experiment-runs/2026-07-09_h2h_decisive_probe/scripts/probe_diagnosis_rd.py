"""probe_diagnosis_rd.py -- PROBE-INSTRUMENT DIAGNOSIS (HEAD_TO_HEAD sec 1.3.1
redesign round, diagnosis stage; 2026-07-09). READ-ONLY on all checkpoints and
results; writes ONLY under results/h2h_rung1/probe_diagnosis/.

Question: the joint-trained sec 1.3.1 probe plateaus at probe_cos_mean
0.12-0.22 in ALL THREE arms (rf@0.9=0.0) while CE learns. Is the answer
information (a) in the tap but under-fit by the joint-trained linear probe
(training dynamics), (b) in the tap but not LINEARLY decodable
(representation), (c) not in the tap at all (task routing), or (d) bounded by
the task itself not being solved (task difficulty)?

Per-arm experiments on the SAVED task1 _auxrev2 checkpoints (frozen backbone,
offline fits on extracted (state_summary_raw, answer) pairs):
  0. REPRODUCE the reported plateau with the checkpoint's own joint-trained
     rig on the pinned EVAL_SEED episode set (sanity anchor).
  1a. Pinned-shape linear (adapter->probe) refit: SGD-on-cosine-loss to
      convergence, cold start AND warm start from the checkpoint rig.
  1b. MLP probe (1 hidden layer, identical hidden width across arms).
  1c. Ridge regression closed form (affine, lambda grid, best on heldout).
  1d. SHUFFLED-TAP negative control for 1c (fit machinery must NOT solve it).
  1e. Linear softmax classifier tap -> answer ENTITY ID (identity
      decodability, separate from T_val-codebook encoding).
  3.  TARGET-SPACE CONTROL: ridge + SGD-cosine with targets = the arm's OWN
      tied-embedding rows of the answer token (is the failure the ARBITRARY
      CODEBOOK property of T_val?).
  4.  Cosine DISTRIBUTION for the best T_val probe (quantiles, rf@bars,
      correct-vs-confusion separation, nearest-row codebook top-1 --
      DIAGNOSTIC ONLY, argmax decode is never the registered metric).
  5.  LM-HEAD ROUTE: [ctx ++ query_window] full forward, logits at the REL
      position (grammar-congruent answer slot) and at <Q>; top-1 accuracy
      full-vocab / pool-restricted / episode-restricted.

Fit set: fresh-seed episodes (DIAG_FIT_SEED). Eval set: the pinned EVAL_SEED
episodes (identical to the calibration curves' own eval set) -> numbers are
directly comparable to the reported plateaus.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_cell_train_rd import (load_h2h_checkpoint, get_pools, task_cfg,     # noqa: E402
                               make_eval_episodes, AUDITED_TAP, probe_targets,
                               eval_recovered_frac, VALUE_DIM, TAP_DIM,
                               GRAMMAR_BATCH, _transformer_episode_chunks)
import probe_head_rd as ph                                                    # noqa: E402
from grammar_rd import sample_batch_rd                                        # noqa: E402

DIAG_FIT_SEED = 20260711          # fresh; distinct from EVAL_SEED(20260710)/PROBE_TARGET_SEED
N_FIT_BATCHES = 40                # 40 x 32 episodes x 32 queries = 40,960 fit points
N_LMHEAD_BATCHES = 8              # 8 x 32 episodes x 32 queries = 8,192 decode points
LOGIT_ROW_CHUNK = 128             # (B*Q)-row chunk for full-vocab logits (~3.6 GB fp32 peak)
SGD_STEPS = 3000
SGD_BATCH = 4096
MLP_HIDDEN = 256                  # identical across arms
BARS = (0.5, 0.6, 0.7, 0.8, 0.9)
QUANTS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def cos_stats(pred: torch.Tensor, tgt: torch.Tensor) -> dict:
    cos = F.cosine_similarity(pred.float(), tgt.float(), dim=-1)
    return {"cos_mean": cos.mean().item(),
            **{f"rf@{b}": (cos >= b).float().mean().item() for b in BARS}}


# ---------------------------------------------------------------------------
# Tap extraction (frozen backbone, audited tap functions)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_taps(arch, model, batches) -> dict:
    """Returns flattened (N, tap_dim) taps + (N,) answer token ids +
    per-point episode entity sets (N, K) for restricted decodes."""
    model.eval()      # eval_recovered_frac leaves the model in train mode; freeze it here
    taps, ans, ep_ents = [], [], []
    for b_full in batches:
        chunks = (_transformer_episode_chunks(
                      b_full, b_full["token_ids"].shape[1] + b_full["query_tokens"].shape[-1])
                  if arch == "transformer" else (b_full,))
        for b in chunks:
            tap = AUDITED_TAP[arch](model, b["token_ids"], b["query_tokens"])   # (B,Q,D)
            B, Q, D = tap.shape
            taps.append(tap.reshape(B * Q, D).float().cpu())
            a = torch.gather(b["entity_ids"], 1, b["tgt_slot"])                 # (B,Q) answer ids
            ans.append(a.reshape(-1).cpu())
            ep_ents.append(b["entity_ids"].unsqueeze(1).expand(B, Q, -1).reshape(B * Q, -1).cpu())
    return {"tap": torch.cat(taps), "answer_id": torch.cat(ans), "ep_entities": torch.cat(ep_ents)}


# ---------------------------------------------------------------------------
# Offline fits
# ---------------------------------------------------------------------------

def ridge_fit(X, Y, lambdas=(1e-6, 1e-4, 1e-2, 1.0, 100.0)):
    """Closed-form affine ridge on GPU float64; returns per-lambda weights."""
    Xd = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1).double()
    Yd = Y.double()
    G = Xd.T @ Xd
    XtY = Xd.T @ Yd
    out = {}
    for lam in lambdas:
        W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=X.device, dtype=torch.float64), XtY)
        out[lam] = W.float()
    return out


def ridge_pred(W, X):
    Xd = torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], dim=1)
    return Xd @ W


def sgd_fit(module: nn.Module, X, Y, steps=None, batch=SGD_BATCH, lr=1e-2,
            gen: torch.Generator | None = None) -> list:
    """Adam on the exact aux cosine loss, cosine LR decay to 1e-4. Returns the
    trailing-100-step loss trace tail so convergence is verifiable."""
    steps = SGD_STEPS if steps is None else steps      # late-bound: smoke mode rewrites the global
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    N = X.shape[0]
    trace = []
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = 1e-4 + 0.5 * (lr - 1e-4) * (1 + torch.cos(torch.tensor(s / steps * 3.14159)).item())
        idx = torch.randint(0, N, (batch,), generator=gen, device=X.device)
        loss = ph.probe_aux_loss(module(X[idx]), Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if s >= steps - 100:
            trace.append(loss.item())
    return trace


class PinnedLinear(nn.Module):
    """adapter (no bias) -> probe (bias): the exact sec 1.3.1.2 shape."""
    def __init__(self, tap_dim, out_dim):
        super().__init__()
        self.adapter = ph.build_adapter_arm(tap_dim, out_dim)
        self.probe = nn.Linear(out_dim, out_dim, bias=True)
    def forward(self, x):
        return self.probe(self.adapter(x))


class MLPProbe(nn.Module):
    def __init__(self, tap_dim, out_dim, hidden=MLP_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(tap_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)


def classifier_fit(X, y_class, n_class, steps=None, batch=SGD_BATCH, lr=1e-2, device="cuda"):
    steps = SGD_STEPS if steps is None else steps      # late-bound: smoke mode rewrites the global
    lin = nn.Linear(X.shape[1], n_class).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    N = X.shape[0]
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = 1e-4 + 0.5 * (lr - 1e-4) * (1 + torch.cos(torch.tensor(s / steps * 3.14159)).item())
        idx = torch.randint(0, N, (batch,), device=X.device)
        loss = F.cross_entropy(lin(X[idx]), y_class[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return lin


# ---------------------------------------------------------------------------
# LM-head route
# ---------------------------------------------------------------------------

@torch.no_grad()
def lm_head_route(arch, model, batches, pool_ids: torch.Tensor, device) -> dict:
    """Full forward over [ctx ++ query_window] (query DOES enter the context
    here -- a DIAGNOSIS of information routes, not the registered readout).
    Decode at the REL position (whose next token is the VALUE slot under the
    bind grammar) and at <Q> (last). Top-1 over: full vocab, the 106-name
    train pool, the episode's own K entities."""
    model.eval()
    stats = {k: {"full": 0, "pool": 0, "episode": 0} for k in ("rel_pos", "q_pos")}
    for b in batches:
        ctx, qt = b["token_ids"], b["query_tokens"]
        B, T_ctx = ctx.shape
        _, Q, qlen = qt.shape
        seq = torch.cat([ctx.unsqueeze(1).expand(B, Q, T_ctx).reshape(B * Q, T_ctx),
                         qt.reshape(B * Q, qlen)], dim=1)
        ans = torch.gather(b["entity_ids"], 1, b["tgt_slot"]).reshape(-1)        # (B*Q,)
        ep_ents = b["entity_ids"].unsqueeze(1).expand(B, Q, -1).reshape(B * Q, -1)
        for i in range(0, seq.shape[0], LOGIT_ROW_CHUNK):
            rows = seq[i:i + LOGIT_ROW_CHUNK]
            logits = model(rows)                                                 # (n,T,V)
            # window layout [buf..., KEY, REL, <Q>]: REL at -2 (its logits predict
            # the next token = the VALUE slot under the bind grammar), <Q> at -1.
            for key, pos in (("rel_pos", -2), ("q_pos", -1)):
                lg = logits[:, pos, :]                                           # (n,V)
                a = ans[i:i + rows.shape[0]]
                ee = ep_ents[i:i + rows.shape[0]]
                stats[key]["full"] += (lg.argmax(-1) == a).sum().item()
                lp = lg[:, pool_ids]
                stats[key]["pool"] += (pool_ids[lp.argmax(-1)] == a).sum().item()
                le = torch.gather(lg, 1, ee)
                stats[key]["episode"] += (torch.gather(ee, 1, le.argmax(-1, keepdim=True)).squeeze(1) == a).sum().item()
    return stats


# ---------------------------------------------------------------------------
# Per-arm driver
# ---------------------------------------------------------------------------

def diagnose_arm(arch: str, ckpt_path: str, device: str) -> dict:
    t_arm0 = time.time()
    out = {"arch": arch, "ckpt": ckpt_path}
    model, rig, doc = load_h2h_checkpoint(ckpt_path, device)
    assert doc["task"].startswith("task1") and doc.get("K", 32) == 32
    pools = get_pools(device)
    cfg_eval = task_cfg("task1_calib", 32, n_query=None)
    hop_set = tuple(cfg_eval.H_train)

    # --- episode sets ---
    eval_batches = make_eval_episodes(cfg_eval, pools, device, hop_set)          # pinned EVAL_SEED set
    gen = torch.Generator(device=device); gen.manual_seed(DIAG_FIT_SEED)
    fit_batches = [sample_batch_rd(cfg_eval, GRAMMAR_BATCH, gen, hop_set, pools, device=device)
                   for _ in range(N_FIT_BATCHES)]

    # --- 0: reproduce the reported plateau with the checkpoint's own rig ---
    rf0, cos0 = eval_recovered_frac(arch, model, rig, eval_batches)
    out["repro_ckpt_rig"] = {"recovered_frac@0.9": rf0, "probe_cos_mean": cos0}
    print(f"[{arch}] repro ckpt rig on pinned eval set: rf={rf0:.4f} cos={cos0:.4f}", flush=True)

    # --- tap extraction (frozen backbone) ---
    fit = extract_taps(arch, model, fit_batches)
    ev = extract_taps(arch, model, eval_batches)
    Xf, Xe = fit["tap"].to(device), ev["tap"].to(device)
    T_val = rig.T_val                                                            # (50259,64) frozen
    Yf, Ye = T_val[fit["answer_id"].to(device)], T_val[ev["answer_id"].to(device)]
    out["n_fit_points"], out["n_eval_points"] = Xf.shape[0], Xe.shape[0]
    out["tap_norm_fit_mean"] = Xf.norm(dim=-1).mean().item()

    results = {}

    # --- 1c: ridge closed form (+ shuffled-tap negative control 1d) ---
    best = None
    for lam, W in ridge_fit(Xf, Yf).items():
        st = cos_stats(ridge_pred(W, Xe), Ye)
        if best is None or st["cos_mean"] > best[1]["cos_mean"]:
            best = (lam, st, W)
    results["ridge_Tval"] = {"lambda": best[0], **best[1]}
    perm = torch.randperm(Xf.shape[0], device=device)
    Wsh = ridge_fit(Xf, Yf[perm], lambdas=(best[0],))[best[0]]
    results["ridge_SHUFFLED_control"] = cos_stats(ridge_pred(Wsh, Xe), Ye)

    # --- 1a: pinned-shape linear, SGD on cosine loss, cold + warm start ---
    g = torch.Generator(device=device); g.manual_seed(0)
    lin_cold = PinnedLinear(TAP_DIM[arch], VALUE_DIM).to(device)
    tr = sgd_fit(lin_cold, Xf, Yf, gen=g)
    results["pinned_linear_sgd_cold"] = {**cos_stats(lin_cold(Xe).detach(), Ye),
                                         "fit_set": cos_stats(lin_cold(Xf).detach(), Yf),
                                         "loss_tail_mean": sum(tr) / len(tr)}
    lin_warm = PinnedLinear(TAP_DIM[arch], VALUE_DIM).to(device)
    lin_warm.adapter.load_state_dict(rig.adapter.state_dict())
    lin_warm.probe.load_state_dict(rig.probe.state_dict())
    sgd_fit(lin_warm, Xf, Yf, gen=g)
    results["pinned_linear_sgd_warm"] = cos_stats(lin_warm(Xe).detach(), Ye)

    # --- 1b: MLP probe ---
    mlp = MLPProbe(TAP_DIM[arch], VALUE_DIM).to(device)
    sgd_fit(mlp, Xf, Yf, gen=g)
    results["mlp_probe"] = {**cos_stats(mlp(Xe).detach(), Ye),
                            "fit_set": cos_stats(mlp(Xf).detach(), Yf)}

    # --- 1e: linear classifier tap -> answer entity id (identity decodability) ---
    pool_ids = pools.train_name_ids.to(device)                                    # ~106 names
    id2class = torch.full((int(pool_ids.max()) + 1,), -1, dtype=torch.int64, device=device)
    id2class[pool_ids] = torch.arange(pool_ids.numel(), device=device)
    yf_c, ye_c = id2class[fit["answer_id"].to(device)], id2class[ev["answer_id"].to(device)]
    assert (yf_c >= 0).all() and (ye_c >= 0).all()
    clf = classifier_fit(Xf, yf_c, pool_ids.numel(), device=device)
    with torch.no_grad():
        results["linear_classifier_identity"] = {
            "top1_heldout": (clf(Xe).argmax(-1) == ye_c).float().mean().item(),
            "top1_fit": (clf(Xf).argmax(-1) == yf_c).float().mean().item(),
            "chance": 1.0 / pool_ids.numel(), "n_classes": pool_ids.numel()}
    # MLP classifier (same hidden as 1b -- identity decodability, nonlinear)
    mlp_clf_head = nn.Sequential(nn.Linear(TAP_DIM[arch], MLP_HIDDEN), nn.ReLU(),
                                 nn.Linear(MLP_HIDDEN, pool_ids.numel())).to(device)
    opt = torch.optim.Adam(mlp_clf_head.parameters(), lr=1e-2)
    for s in range(SGD_STEPS):
        for gpg in opt.param_groups:
            gpg["lr"] = 1e-4 + 0.5 * (1e-2 - 1e-4) * (1 + torch.cos(torch.tensor(s / SGD_STEPS * 3.14159)).item())
        idx = torch.randint(0, Xf.shape[0], (SGD_BATCH,), device=device)
        loss = F.cross_entropy(mlp_clf_head(Xf[idx]), yf_c[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        results["mlp_classifier_identity"] = {
            "top1_heldout": (mlp_clf_head(Xe).argmax(-1) == ye_c).float().mean().item()}

    # --- 3: target-space control (tied-embedding rows of the answer token) ---
    E = model.embed.weight.detach()                                               # (V, d_model=256)
    Yf_e, Ye_e = E[fit["answer_id"].to(device)], E[ev["answer_id"].to(device)]
    beste = None
    for lam, W in ridge_fit(Xf, Yf_e).items():
        st = cos_stats(ridge_pred(W, Xe), Ye_e)
        if beste is None or st["cos_mean"] > beste[1]["cos_mean"]:
            beste = (lam, st)
    results["ridge_tied_embed_target"] = {"lambda": beste[0], **beste[1]}
    lin_e = nn.Linear(TAP_DIM[arch], E.shape[1]).to(device)
    sgd_fit(lin_e, Xf, Yf_e, gen=g)
    results["sgd_linear_tied_embed_target"] = cos_stats(lin_e(Xe).detach(), Ye_e)
    # matched control: SAME machinery, targets swapped back to T_val (rules out
    # "the tied-embed fit only looks better because 256-dim output/lr happened to fit better")
    lin_t = nn.Linear(TAP_DIM[arch], VALUE_DIM).to(device)
    sgd_fit(lin_t, Xf, Yf, gen=g)
    results["sgd_linear_Tval_target_matched"] = cos_stats(lin_t(Xe).detach(), Ye)

    # --- 4: distribution for the best T_val probe ---
    with torch.no_grad():
        candidates = {"ridge": ridge_pred(best[2], Xe), "pinned_sgd": lin_cold(Xe), "mlp": mlp(Xe)}
    name, pred = max(candidates.items(),
                     key=lambda kv: F.cosine_similarity(kv[1], Ye, dim=-1).mean().item())
    cos = F.cosine_similarity(pred, Ye, dim=-1)
    # nearest-T_val-row over the train pool (codebook top-1 -- DIAGNOSTIC ONLY,
    # never the registered metric; CLAUDE.md argmax rule acknowledged)
    predn = F.normalize(pred, dim=-1)
    pool_rows = F.normalize(T_val[pool_ids], dim=-1)
    nn_idx = (predn @ pool_rows.T).argmax(-1)
    nn_correct = (pool_ids[nn_idx] == ev["answer_id"].to(device))
    results["best_probe_distribution"] = {
        "which": name,
        "quantiles": {str(q): torch.quantile(cos, q).item() for q in QUANTS},
        **{f"rf@{b}": (cos >= b).float().mean().item() for b in BARS},
        "codebook_top1_DIAGNOSTIC_ONLY": nn_correct.float().mean().item(),
        "cos_mean_when_codebook_correct": cos[nn_correct].mean().item() if nn_correct.any() else None,
        "cos_mean_when_codebook_wrong": cos[~nn_correct].mean().item() if (~nn_correct).any() else None,
    }

    # --- constant-predictor floor (context for all cos numbers) ---
    mean_t = Yf.mean(0, keepdim=True).expand_as(Ye)
    results["constant_mean_predictor_floor"] = cos_stats(mean_t, Ye)

    # --- 5: LM-head route ---
    lm_batches = [sample_batch_rd(cfg_eval, GRAMMAR_BATCH, gen, hop_set, pools, device=device)
                  for _ in range(N_LMHEAD_BATCHES)]
    stats = lm_head_route(arch, model, lm_batches, pool_ids, device)
    n_pts = sum(b["query_tokens"].shape[0] * b["query_tokens"].shape[1] for b in lm_batches)
    results["lm_head_route"] = {k: {kk: v / n_pts for kk, v in d.items()} for k, d in stats.items()}
    results["lm_head_route"]["n_points"] = n_pts
    results["lm_head_route"]["chance_episode"] = 1.0 / 32

    out["results"] = results
    out["wall_s"] = time.time() - t_arm0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="results/h2h_rung1/probe_diagnosis")
    ap.add_argument("--arms", nargs="+", default=["contender", "ablation", "transformer"])
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, wiring check only")
    args = ap.parse_args()
    global N_FIT_BATCHES, N_LMHEAD_BATCHES, SGD_STEPS
    if args.smoke:
        N_FIT_BATCHES, N_LMHEAD_BATCHES, SGD_STEPS = 2, 1, 50
    ckpts = {a: f"/data/h2h_rung1_ckpts/h2h_calib_{a}_task1_calib_primary_K32_auxrev2.pt"
             for a in args.arms}
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    for arch, ck in ckpts.items():
        r = diagnose_arm(arch, ck, args.device)
        suffix = "_smoke" if args.smoke else ""
        path = os.path.join(args.out_dir, f"diagnosis_{arch}{suffix}.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[{arch}] DONE in {r['wall_s']:.1f}s -> {path}", flush=True)
        print(json.dumps(r["results"], indent=2), flush=True)
    print(f"TOTAL wall {time.time() - t0:.1f}s (~{(time.time() - t0) / 3600:.3f} GPU-h)", flush=True)


if __name__ == "__main__":
    main()
