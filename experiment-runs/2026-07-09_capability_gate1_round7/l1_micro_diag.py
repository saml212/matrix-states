"""ADJUDICATION ROUND 7 -- L=1 mechanism micro-diagnostic (authorized <=0.1 GPU-h).

The L=1 dip (0.78-0.89 direct mean-cos in 6/7 calibration cells, while L=2-4 are
near-perfect) -- three candidate mechanisms, each probed with the cheapest
discriminating test on the EXISTING gate1_diagnosis checkpoints:

  H-EVAL  eval-sample/protocol artifact: the L=1 eval batch contains only |gens|
          (3-4) distinct words; or degauging fit/eval degeneracy.
          [Static fact checked in code: gate 1(a)'s metric is the DIRECT cosine
          (gate1_diag.per_L_curve) -- degauging is never applied to it.]
  H-ENC   encoder degeneracy at L=1: the reader's attention over a single memory
          position is query-independent (softmax over one key), so all d_state
          row-reads are identical -- is that function class the ceiling?
  H-MIX   training-mix / slow-learning: L=1 gets ~1/8 of batches but only |gens|
          distinct words; gradient conflict on shared params, not a hard ceiling.

Probes:
  P1  per-generator direct cosine at L=1 + block-error decomposition (all ckpts).
      Uniform depression => scale/gauge artifact; heterogeneous => representational.
  P2  degauging probe (S4@8K, A6@20K): centered-U pipeline, fit INCLUDING vs
      EXCLUDING L=1 words on the SAME eval set; plus degauged scores OF the L=1
      words themselves.
  P3  read-vector degeneracy: numerical check that at L=1 the reader output is
      identical across all d_state row queries (and not at L=2).
  P4  frozen-downstream capacity: optimize a free (1,1,h) input embedding per
      generator against the FROZEN model to max cosine. High ceiling => capacity
      exists (favors H-MIX); low ceiling => architectural (favors H-ENC).
  P5  L=1-only fine-tune, 300 steps, lr 3e-4, batch 256, on CLONES of S4@8K and
      A6@20K, vs a 300-step standard-mix control. Fast convergence => H-MIX.
      Followed by the full per-L curve (seed 424242, same as the report) to
      price the interference on L>=2.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch

CODE_DIR = os.environ.get("GATE1_CODE_DIR", "/home/nvidia/chapter2/capability_separation")
DIAG_DIR = os.path.join(CODE_DIR, "results", "gate1_diagnosis")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, DIAG_DIR)

from groups import (GROUP_NAMES, D_MIN, D_STATE, generating_set, group_seed_salt,
                    word_product, batched_targets, rho_G_embedded)
from group_task import sample_train_batch
from group_word_encoder import GroupWordModel, cosine_loss, recovery_cosine
from readout import dump_Z, restrict, degauge_and_score
from gate1_diag import per_L_curve  # same seed/machinery as the report's curves

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_T0 = time.time()
REPORT = {}

CKPTS = [("S3", 8000), ("S4", 8000), ("A5", 8000), ("S5", 8000), ("A6", 8000),
         ("A5", 20000), ("A6", 20000)]


def load(name, steps):
    model = GroupWordModel(D_STATE[name], len(generating_set(name)), L_max=16, h=32).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(DIAG_DIR, f"ckpt_{name}_{steps}.pt"),
                                     map_location=DEVICE))
    model.eval()
    return model


def flat_cos(A, B):
    a, b = A.flatten(), B.flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------------------
print("=" * 100)
print("P1 -- per-generator direct cosine at L=1 + block decomposition (all 7 ckpts)")
print("     [distinct words in any L=1 eval batch == |gens|: S3/S5=3, S4/A5/A6=4]")
print("=" * 100, flush=True)
p1 = {}
for name, steps in CKPTS:
    model = load(name, steps)
    d, ds = D_MIN[name], D_STATE[name]
    gens = generating_set(name)
    rows = []
    with torch.no_grad():
        for i, g in enumerate(gens):
            tok = torch.tensor([[i]], dtype=torch.long, device=DEVICE)
            Z = model.encode(tok).squeeze(0).cpu().numpy().astype(np.float64)
            T = rho_G_embedded(g, ds)
            rows.append(dict(
                gen=i,
                cos_full=flat_cos(Z, T),
                cos_rho_block=flat_cos(Z[:d, :d], g),
                cos_compl_block=flat_cos(Z[d:, d:], np.eye(ds - d)),
                fro_rho=float(np.linalg.norm(Z[:d, :d])),
                fro_compl=float(np.linalg.norm(Z[d:, d:])),
                fro_off=float(np.sqrt(np.linalg.norm(Z[:d, d:]) ** 2 +
                                      np.linalg.norm(Z[d:, :d]) ** 2)),
            ))
    cs = [r["cos_full"] for r in rows]
    p1[f"{name}@{steps}"] = rows
    print(f"[{name}@{steps:>5d}] per-gen cos_full: " + " ".join(f"{c:.4f}" for c in cs)
          + f"   mean={np.mean(cs):.4f} min={min(cs):.4f} max={max(cs):.4f} spread={max(cs)-min(cs):.4f}")
    print(f"           per-gen cos_rho_block: "
          + " ".join(f"{r['cos_rho_block']:.4f}" for r in rows)
          + "  |  cos_compl: " + " ".join(f"{r['cos_compl_block']:.3f}" for r in rows))
    print(f"           fro rho/compl/off (gen-mean): "
          f"{np.mean([r['fro_rho'] for r in rows]):.3f}/"
          f"{np.mean([r['fro_compl'] for r in rows]):.3f}/"
          f"{np.mean([r['fro_off'] for r in rows]):.3f}", flush=True)
    del model
REPORT["P1_per_generator"] = p1


# ---------------------------------------------------------------------------
def sample_words_range(name, seed, n_words, lo, hi):
    rng = np.random.default_rng(seed + group_seed_salt(name))
    gens = generating_set(name)
    idx_list, prod_list = [], []
    for _ in range(n_words):
        L = int(rng.integers(lo, hi + 1))
        idx_list.append(rng.integers(0, len(gens), size=L))
        prod_list.append(word_product(gens, idx_list[-1]))
    return idx_list, prod_list


def centered_U(Z_words, d_min):
    Zbar = Z_words.mean(axis=0)
    cov = np.zeros_like(Zbar)
    for Z in Z_words:
        D = Z - Zbar
        cov += D @ D.T
    U_full, _, _ = np.linalg.svd(cov)
    return U_full[:, :d_min]


print("\n" + "=" * 100)
print("P2 -- degauging probe: fit INCL vs EXCL L=1 words (same eval set) + L=1 words scored")
print("=" * 100, flush=True)
p2 = {}
for name, steps in (("S4", 8000), ("A6", 20000)):
    model = load(name, steps)
    d, ds = D_MIN[name], D_STATE[name]
    gens = generating_set(name)
    idx_list, prod_list = sample_words_range(name, 909_000, 50, 1, 8)
    Ls = np.array([len(ix) for ix in idx_list])
    n_l1 = int((Ls == 1).sum())
    Z_words = dump_Z(model, idx_list, DEVICE)
    U_all = centered_U(Z_words, d)
    U_no1 = centered_U(Z_words[Ls >= 2], d)
    pc = np.linalg.svd(U_all.T @ U_no1, compute_uv=False)  # principal cosines U_all vs U_no1

    rng = np.random.default_rng(777_001)
    perm = rng.permutation(50)
    fit_i, eval_i = perm[:30].tolist(), perm[30:50].tolist()
    fit_ii = [i for i in fit_i if Ls[i] >= 2]              # EXCLUDE L=1 from fit

    def score(fit_idx, eval_idx, U):
        A = np.stack([restrict(Z, U) for Z in Z_words])
        return degauge_and_score([A[i] for i in fit_idx], [A[i] for i in eval_idx],
                                 [prod_list[i] for i in fit_idx],
                                 [prod_list[i] for i in eval_idx], d)

    s_incl = score(fit_i, eval_i, U_all)
    s_excl = score(fit_ii, eval_i, U_all)
    # the L=1 words themselves, degauged with an L>=2-only fit:
    A_all = np.stack([restrict(Z, U_all) for Z in Z_words])
    Zg = dump_Z(model, [np.array([i]) for i in range(len(gens))], DEVICE)
    Ag = [restrict(Z, U_all) for Z in Zg]
    s_l1words = degauge_and_score([A_all[i] for i in fit_ii], Ag,
                                  [prod_list[i] for i in fit_ii], gens, d)
    p2[f"{name}@{steps}"] = dict(
        n_l1_in_sample=n_l1,
        fit_incl=dict(n_fit=len(fit_i), mean_cos=s_incl["mean_cos"], rec90=s_incl["recovered_frac_90"]),
        fit_excl=dict(n_fit=len(fit_ii), mean_cos=s_excl["mean_cos"], rec90=s_excl["recovered_frac_90"]),
        l1_words_degauged=dict(mean_cos=s_l1words["mean_cos"],
                               coses=[float(c) for c in s_l1words["coses"]]),
        principal_cos_Uall_vs_UnoL1=[float(x) for x in pc],
    )
    print(f"[{name}@{steps}] L=1 words in N=50 sample: {n_l1}")
    print(f"  degauged eval mean_cos: fit-INCL-L1 (n=30) {s_incl['mean_cos']:+.4f} | "
          f"fit-EXCL-L1 (n={len(fit_ii)}) {s_excl['mean_cos']:+.4f} | "
          f"delta {s_excl['mean_cos'] - s_incl['mean_cos']:+.4f}")
    print(f"  the {len(gens)} L=1 words, degauged (L>=2 fit): "
          + " ".join(f"{c:.4f}" for c in s_l1words["coses"])
          + f"  (mean {s_l1words['mean_cos']:.4f})")
    print(f"  principal cosines U(all-50) vs U(L>=2 only): "
          + " ".join(f"{x:.4f}" for x in pc), flush=True)
    del model
REPORT["P2_degauging"] = p2


# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("P3 -- reader degeneracy at L=1: read-vector std across the d_state row queries")
print("=" * 100, flush=True)
p3 = {}
model = load("S4", 8000)
enc = model.encoder
with torch.no_grad():
    for L in (1, 2):
        tok = torch.randint(0, 4, (8, L), generator=torch.Generator().manual_seed(7)).to(DEVICE)
        emb = enc.embed_tokens(tok)
        mem = enc.encoder(emb)
        q = enc.row_queries.unsqueeze(0).expand(8, enc.d_state, enc.h)
        read, _ = enc.reader(q, mem, mem, need_weights=False)
        std_across_queries = float(read.std(dim=1).max().item())
        p3[f"L={L}"] = std_across_queries
        print(f"  L={L}: max std of reader output across the {enc.d_state} row queries "
              f"= {std_across_queries:.3e}  ({'DEGENERATE (query-independent)' if std_across_queries < 1e-6 else 'query-dependent'})")
del model
REPORT["P3_reader_degeneracy"] = p3


# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("P4 -- frozen-downstream capacity: optimize a free (1,1,h) embedding per generator")
print("=" * 100, flush=True)
p4 = {}
for name, steps in (("S4", 8000), ("A6", 20000)):
    model = load(name, steps)
    for p in model.parameters():
        p.requires_grad_(False)
    enc = model.encoder
    ds = D_STATE[name]
    gens = generating_set(name)
    best = []
    for i, g in enumerate(gens):
        tok = torch.tensor([[i]], dtype=torch.long, device=DEVICE)
        T = torch.tensor(rho_G_embedded(g, ds), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        e = torch.nn.Parameter(enc.embed_tokens(tok).detach().clone())
        opt = torch.optim.Adam([e], lr=1e-2)
        b = -1.0
        for it in range(400):
            Z = enc.encode_from_embedding(e)
            loss = cosine_loss(Z, T)
            opt.zero_grad(); loss.backward(); opt.step()
            b = max(b, 1.0 - float(loss.item()))
        best.append(b)
    p4[f"{name}@{steps}"] = best
    print(f"[{name}@{steps}] best achievable cos per generator (frozen downstream, 400 Adam steps): "
          + " ".join(f"{b:.4f}" for b in best) + f"   min={min(best):.4f}", flush=True)
    del model
REPORT["P4_frozen_capacity"] = p4


# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("P5 -- L=1-only fine-tune (300 steps) vs standard-mix control, clones of S4@8K / A6@20K")
print("=" * 100, flush=True)
p5 = {}


def eval_l1(model, name):
    """Equal-weight mean direct cosine over the |gens| distinct L=1 words."""
    ds = D_STATE[name]
    n_gens = len(generating_set(name))
    tok = torch.arange(n_gens, dtype=torch.long, device=DEVICE).unsqueeze(1)  # (n_gens, 1)
    with torch.no_grad():
        target = batched_targets(name, tok, ds)
        cos = recovery_cosine(model.encode(tok), target)
    return float(cos.mean().item())


for name, steps in (("S4", 8000), ("A6", 20000)):
    base = load(name, steps)
    ds = D_STATE[name]
    n_gens = len(generating_set(name))
    out = {}
    for mode in ("l1_only", "mix_control"):
        model = copy.deepcopy(base)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        gen = torch.Generator().manual_seed(31337 + group_seed_salt(name))
        traj = {0: eval_l1(model, name)}
        for step in range(1, 301):
            if mode == "l1_only":
                tok = torch.randint(0, n_gens, (256, 1), generator=gen).to(DEVICE)
                batch = {"token_idx": tok, "target": batched_targets(name, tok, ds)}
            else:
                batch = sample_train_batch(name, 256, gen, device=DEVICE)
            loss = cosine_loss(model.encode(batch["token_idx"]), batch["target"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step in (25, 50, 100, 200, 300):
                traj[step] = eval_l1(model, name)
        out[mode] = dict(l1_trajectory=traj)
        print(f"[{name}@{steps}][{mode:11s}] L=1 equal-weight cos: "
              + " -> ".join(f"{s}:{c:.4f}" for s, c in traj.items()), flush=True)
        if mode == "l1_only":
            curve = per_L_curve(model, name)     # same seed 424242 as the report
            out[mode]["per_L_after"] = {L: curve[L]["mean_cos"] for L in range(1, 9)}
            print(f"             per-L after L=1-only FT: "
                  + " ".join(f"{L}:{curve[L]['mean_cos']:.3f}" for L in range(1, 9)), flush=True)
        del model
    p5[f"{name}@{steps}"] = out
    del base
REPORT["P5_finetune"] = p5

REPORT["wall_seconds_total"] = time.time() - GPU_T0
with open(os.path.join(DIAG_DIR, "l1_micro_diag.json"), "w") as f:
    json.dump(REPORT, f, indent=2)
print(f"\nwall total: {REPORT['wall_seconds_total']:.1f}s  -> "
      f"{REPORT['wall_seconds_total']/3600:.4f} GPU-h (upper bound; incl CPU-bound degauging)")
print(f"report -> {os.path.join(DIAG_DIR, 'l1_micro_diag.json')}")
