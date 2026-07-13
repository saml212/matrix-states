#!/usr/bin/env python3
"""JOB 1 RECEIPT: the A-3 residual clears the DEPLOYED T2a-4 gate.

T2a-4's gate is check_t2a4_positive_control(records), which PASSES iff, on the
oracle's records, argmax_intact_at_k == b AND argmax_intact != argmax_keyswap on
EVERY row. We feed it REAL PerfectCopyOracle records (the DEPLOYED oracle, the
DEPLOYED plant_and_verify_t2_window hard assertion) built over a corpus that is:

  correct        : coherent in-range tokens.
  A3i_perm       : content-preserving mis-tokenization -- a vocab BIJECTION
                   (in-range wrong tokens); the bridge's <vocab_size + collider
                   asserts do NOT catch it.
  A3i_xtok       : content-preserving mis-tokenization -- a genuine wrong
                   tokenizer (falcon on gpt2 text, mod vocab).
  A3ii_wrong_eot : the plant window carries a REAL eos (50256) that a wrong
                   probe-time eot_override would fail to exclude.

If the DEPLOYED check_t2a4_positive_control returns passes=True on the mis-aimed
adapters, T2a-4 is blind to them -- the Job-1 attack.  Because the wrapped model
is PERFECT (mechanism by fiat), the OTHER legs (t2a1/t2a3) also pass on such an
adapter, so the full gate certifies INSTRUMENT_VALID.  The whole question of Job
2 is then whether that blind spot MATTERS -- see a3_adjudication_measure.py.
"""
import os, sys, json, random
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_recall_gap_probe_v2_rd as probe
from lm_recall_gap_probe_v2_rd import plant_and_verify_t2_window, PlantContestedError, PerfectCopyOracle

DEVICE = "cuda"
SEQ, N, VOCAB, EOS, FILLER = 512, 512, 50257, 50256, 220
DATA, CORPUS, SEED = "/data/deltanet_rd_data", "openr1-mix-ext", 424242
rng = random.Random(SEED); torch.manual_seed(SEED)

oracle = PerfectCopyOracle(VOCAB).to(DEVICE)   # the DEPLOYED oracle, mechanism BY FIAT
_, val, _, _, _ = probe.load_corpus(DATA, CORPUS, "cpu"); val = val.tolist()
from transformers import AutoTokenizer
g = AutoTokenizer.from_pretrained("gpt2-large"); f = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b", trust_remote_code=True)
pi = list(range(VOCAB)); rng.shuffle(pi)

starts = list(range(1000, 1000 + N * 700, 700))[:N]
wins = [val[s:s + SEQ + 1] for s in starts]

def mistok(w, mode):
    if mode == "correct": return list(w)
    if mode == "A3i_perm": return [pi[t] for t in w]
    if mode == "A3i_xtok":
        ids = [i % VOCAB for i in f(g.decode([t for t in w if t != EOS]), add_special_tokens=False)["input_ids"]]
        return (ids + [FILLER] * len(w))[:len(w)]
    if mode == "A3ii_wrong_eot": return list(w)   # real eos left in context (see splice below)
    raise ValueError(mode)

def build_records(mode):
    intact, keyswap, k0s, bs, deltas = [], [], [], [], []
    for w in wins:
        a, ap, b = rng.sample([t for t in range(300, VOCAB) if t not in (EOS, FILLER)], 3)
        delta = rng.randint(2, SEQ - 6); k0 = rng.randint(delta + 2, SEQ - 2); j0 = k0 - delta; p = j0 + 1
        ctx = mistok(w, mode)
        if mode == "A3ii_wrong_eot" and k0 - p > 2:
            ctx[rng.randint(p + 1, k0 - 1)] = EOS   # a real doc boundary a wrong eot won't exclude
        ctx = [FILLER if (idx not in (j0, p, k0) and t in (a, ap, b)) else t for idx, t in enumerate(ctx)]
        try:
            wi = plant_and_verify_t2_window(ctx, j0, k0, a, b)
        except PlantContestedError:
            continue
        wk = list(wi); wk[j0] = ap
        intact.append(wi); keyswap.append(wk); k0s.append(k0); bs.append(b); deltas.append(delta)
    def argmax_at(windows):
        out = []
        xb = torch.tensor([w[:-1] for w in windows], dtype=torch.long, device=DEVICE)
        k0t = torch.tensor(k0s, dtype=torch.long, device=DEVICE)
        for s in range(0, xb.shape[0], 8):
            with torch.no_grad():
                am = oracle(xb[s:s + 8]).argmax(dim=-1)
            out.extend(am[torch.arange(am.shape[0], device=DEVICE), k0t[s:s + am.shape[0]]].tolist())
        return out
    ai, ak = argmax_at(intact), argmax_at(keyswap)
    return [{"b": bs[i], "argmax_intact_at_k": ai[i], "argmax_keyswap_at_k": ak[i], "delta": deltas[i]}
            for i in range(len(bs))]

print(f"{'mode':16s} {'n':>4s} {'passes':>7s} {'n_miss':>7s} {'n_aim_unchanged':>16s}")
res = {}
for mode in ("correct", "A3i_perm", "A3i_xtok", "A3ii_wrong_eot"):
    recs = build_records(mode)
    v = probe.check_t2a4_positive_control(recs)          # THE DEPLOYED GATE
    nmiss = sum(1 for r in recs if r["argmax_intact_at_k"] != r["b"])
    naim = sum(1 for r in recs if r["argmax_intact_at_k"] == r["argmax_keyswap_at_k"])
    res[mode] = {"n": len(recs), "passes": v["passes"], "n_miss": nmiss, "n_aim_unchanged": naim}
    print(f"{mode:16s} {len(recs):4d} {str(v['passes']):>7s} {nmiss:7d} {naim:16d}")
print(json.dumps(res, indent=2))
