#!/usr/bin/env python3
"""A-3 ADJUDICATION MEASUREMENT (sec 29 dispatch, Job 2).

Question the whole verdict turns on: do A-3's two UNCAUGHT adapter mis-aims
actually CORRUPT a REAL induction model's acc_copy, or do they only miss
corruptions that leave the copy measurement intact?

  A-3(i)  content-preserving CONTEXT mis-tokenization (in-range wrong tokens).
  A-3(ii) non-collider probe-time eot_override mis-aim.

METHOD. Real model: gpt2-large (the W2 witness, a documented induction-head
transformer), loaded bf16 and wrapped in the DEPLOYED HFLogitsWrapper (fp32
upcast + finite check) -- the exact adapter every witness verdict travels.
Real plant: the DEPLOYED plant_and_verify_t2_window hard assertion. Real
readout: argmax at k0 == b, the exact acc_copy definition.

M1 is a PAIRED isolation: identical plant (a,b,a at j0,p,k0), identical delta,
identical (a,a',b) -- ONLY the CONTEXT tokenization differs across conditions.
That is the precise variable the dispatch asks about:
  "does the model's copy of the plant depend on the context being correctly
   tokenized, or only on the plant pattern being present?"

Conditions:
  correct   : real coherent gpt2 context.
  perm      : context = pi(real token), a fixed vocab BIJECTION -> systematic
              in-range gibberish preserving per-token frequency + repeat
              structure (the faithful "wrong tokenizer" analog).
  rand      : context = uniform random in-range tokens (worst-case OOD).
  eosbound  : real context with a REAL eos (50256) spliced between p and k0 --
              the ONLY mechanism by which a wrong non-collider eot_override can
              reach acc_copy (real doc boundaries no longer excluded => windows
              span them). A-3(ii)'s worst concrete mechanism.

acc_copy (arm1) and aiming (arm1 vs arm4 key-swap) reported per condition.
NO GPU TRAINING TOUCHED. gpt2-large forwards only, on the pinned GPU.
"""
import os, sys, json, math, random
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_recall_gap_probe_v2_rd as probe
import t2a_reference_driver_v2_rd as driver
from lm_recall_gap_probe_v2_rd import plant_and_verify_t2_window, PlantContestedError

DEVICE = "cuda"
SEQ = 512                 # SEQ_LEN_DEFAULT
N = 640                   # paired windows
VOCAB = 50257             # gpt2
EOS = 50256               # GPT2_EOT
FILLER = 220              # ' ' token, used to clear accidental a/a'/b collisions in gibberish
MICRO = 32
DATA = "/data/deltanet_rd_data"
CORPUS = "openr1-mix-ext"
SEED = 20260713

rng = random.Random(SEED)
torch.manual_seed(SEED)

print(f"[load] gpt2-large bf16 + DEPLOYED HFLogitsWrapper on {DEVICE}", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
hf = AutoModelForCausalLM.from_pretrained("gpt2-large", dtype=torch.bfloat16).to(DEVICE).eval()
model = driver.HFLogitsWrapper(hf).to(DEVICE).eval()
assert next(hf.parameters()).dtype == torch.bfloat16

# tokenizers for the FAITHFUL cross-tokenizer mis-tokenization (xtok): decode gpt2 text,
# re-encode with a DIFFERENT real tokenizer (falcon-mamba), map ids into gpt2 range -> the
# model reads its-own-embeddings-in-wrong-positions = STRUCTURED gibberish a real wrong
# tokenizer would produce (NOT uniform-random, which no tokenizer emits).
print("[load] gpt2-large(=gpt2 BPE) + falcon-mamba tokenizers for xtok", flush=True)
gpt2_tok = AutoTokenizer.from_pretrained("gpt2-large")   # same BPE as gpt2, and fully cached
falcon_tok = AutoTokenizer.from_pretrained("tiiuae/falcon-mamba-7b", trust_remote_code=True)

print(f"[load] corpus {CORPUS}", flush=True)
train, val, meta, _, _ = probe.load_corpus(DATA, CORPUS, "cpu")
train = train.tolist()
val = val.tolist()
print(f"  train={len(train):,} val={len(val):,}", flush=True)

# token frequency -> pick rare-but-trained plant tokens (mimics V5 p_train(b)<=1e-4, count>=2)
counts = torch.bincount(torch.tensor(train), minlength=VOCAB).tolist()
Ntrain = len(train)
rare = [t for t in range(VOCAB) if 2 <= counts[t] <= Ntrain * 1e-4 and t != EOS and t != FILLER]
print(f"  rare-token pool (2<=count<=1e-4*N): {len(rare):,}", flush=True)

# sample N real coherent context windows of length SEQ+1 from val (non-eos-heavy, spaced)
starts = []
step = max(1, (len(val) - SEQ - 2) // (N + 5))
i = 1000
while len(starts) < N and i + SEQ + 1 < len(val):
    starts.append(i)
    i += step
real_windows = [val[s:s + SEQ + 1] for s in starts]

# one fixed vocab bijection pi (derangement-ish; identity-free not required)
pi = list(range(VOCAB))
rng.shuffle(pi)

def build_specs():
    specs = []
    for w in real_windows:
        # draw (a,a',b) distinct rare tokens
        a, ap, b = rng.sample(rare, 3)
        # delta / positions
        delta = rng.randint(2, SEQ - 6)
        k0 = rng.randint(delta + 2, SEQ - 2)
        j0 = k0 - delta
        specs.append({"w": w, "a": a, "ap": ap, "b": b, "delta": delta, "j0": j0, "k0": k0})
    return specs

def clear_collisions(ctx, a, ap, b, protect):
    """Replace any occurrence of a/ap/b OUTSIDE protected positions with FILLER,
    so plant_and_verify's exact-count hard assertion holds. protect = {j0,p,k0}."""
    bad = {a, ap, b}
    return [FILLER if (idx not in protect and t in bad) else t for idx, t in enumerate(ctx)]

def make_context(w, cond, a, ap, b, j0, k0):
    p = j0 + 1
    protect = {j0, p, k0}
    if cond == "correct":
        ctx = list(w)
    elif cond == "perm":
        ctx = [pi[t] for t in w]
    elif cond == "rand":
        ctx = [rng.randrange(VOCAB) for _ in w]
    elif cond == "xtok":
        # FAITHFUL wrong-tokenizer: decode gpt2 window -> text -> falcon-mamba re-tokenize ->
        # ids mod VOCAB (in-range wrong tokens). Structured gibberish, NOT uniform-random.
        text = gpt2_tok.decode([t for t in w if t != EOS])
        fids = falcon_tok(text, add_special_tokens=False)["input_ids"]
        fids = [i % VOCAB for i in fids]
        if len(fids) < len(w):
            fids = fids + [FILLER] * (len(w) - len(fids))
        ctx = fids[:len(w)]
    elif cond == "eosbound":
        # DIAGNOSTIC (NOT A-3(ii) itself): a REAL eos spliced strictly between p and k0.
        # Measures the induction cost of a doc boundary between key and query -- a mechanism
        # that is PRESENT EQUALLY under correct and wrong eot_override (the probe never excludes
        # eos from plant windows), so it is NOT differential in the eot mis-aim. Kept only to
        # quantify the boundary-crossing penalty in isolation.
        ctx = list(w)
        if k0 - p > 2:
            pos = rng.randint(p + 1, k0 - 1)
            ctx[pos] = EOS
    else:
        raise ValueError(cond)
    return clear_collisions(ctx, a, ap, b, protect)

def read_argmax_at_k(windows, k0s):
    """windows: list[list[int]] length SEQ+1. Feed w[:-1] (len SEQ), argmax at k0."""
    out = []
    xb = torch.tensor([w[:-1] for w in windows], dtype=torch.long, device=DEVICE)
    k0t = torch.tensor(k0s, dtype=torch.long, device=DEVICE)
    for s in range(0, xb.shape[0], MICRO):
        with torch.no_grad():
            lg = model(xb[s:s + MICRO])                 # (b, SEQ, V) fp32
        am = lg.argmax(dim=-1)                          # (b, SEQ)
        idx = torch.arange(am.shape[0], device=DEVICE)
        out.extend(am[idx, k0t[s:s + am.shape[0]]].tolist())
        del lg
    return out

CONDITIONS = ["correct", "perm", "xtok", "rand", "eosbound"]
specs = build_specs()
results = {}
for cond in CONDITIONS:
    intact, keyswap, k0s, bs, dropped = [], [], [], [], 0
    for sp in specs:
        a, ap, b, j0, k0 = sp["a"], sp["ap"], sp["b"], sp["j0"], sp["k0"]
        ctx = make_context(sp["w"], cond, a, ap, b, j0, k0)
        try:
            w_intact = plant_and_verify_t2_window(ctx, j0, k0, a, b)     # a at j0,k0 ; b at p
            # key-swap arm: w[j0] := a' (a now only at k0)
            w_ks = list(w_intact); w_ks[j0] = ap
        except PlantContestedError:
            dropped += 1
            continue
        intact.append(w_intact); keyswap.append(w_ks); k0s.append(k0); bs.append(b)
    am_intact = read_argmax_at_k(intact, k0s)
    am_keyswap = read_argmax_at_k(keyswap, k0s)
    n = len(bs)
    acc_copy = sum(int(am_intact[i] == bs[i]) for i in range(n)) / n
    aiming = sum(int(am_intact[i] != am_keyswap[i]) for i in range(n)) / n
    # KS = P(hit intact) - P(hit keyswap)  (both == b)
    hit_i = sum(int(am_intact[i] == bs[i]) for i in range(n)) / n
    hit_k = sum(int(am_keyswap[i] == bs[i]) for i in range(n)) / n
    ks = hit_i - hit_k
    results[cond] = {"n": n, "dropped": dropped, "acc_copy": round(acc_copy, 4),
                     "aiming": round(aiming, 4), "KS_hit_intact_minus_keyswap": round(ks, 4),
                     "hit_keyswap": round(hit_k, 4)}
    print(f"[M1 {cond:9s}] n={n} dropped={dropped}  acc_copy={acc_copy:.4f}  "
          f"aiming={aiming:.4f}  KS={ks:.4f}  hit_keyswap={hit_k:.4f}", flush=True)

print("\n=== M1 SUMMARY (paired isolation; same plant, same delta, same (a,a',b)) ===")
c = results["correct"]["acc_copy"]
for cond in CONDITIONS:
    r = results[cond]
    rel = (r["acc_copy"] - c)
    print(f"  {cond:9s}: acc_copy={r['acc_copy']:.4f}  (delta vs correct = {rel:+.4f})  "
          f"aiming={r['aiming']:.4f}  KS={r['KS_hit_intact_minus_keyswap']:.4f}")
print(json.dumps({"config": {"model": "gpt2-large", "seq_len": SEQ, "N": N, "corpus": CORPUS,
                             "seed": SEED}, "M1": results}, indent=2))
