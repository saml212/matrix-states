#!/usr/bin/env python3
"""
DeltaNet-RD Wave 2 (Waves C+D) analysis.
Descriptive + interventional claim tier ONLY (DELTANET_REALDATA_DESIGN.md sec 6.3, 14.7).
CPU-only, reads result JSONs, no training/eval launched.
"""
import json, glob, statistics as st
from collections import defaultdict

WC_DIR = "/home/nvidia/chapter2/deltanet_rd/results/lm_rd/waveC"
WD_DIR = "/home/nvidia/chapter2/deltanet_rd/results/lm_rd/waveD"

def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
def sd(xs):
    if len(xs) < 2: return float('nan')
    m = mean(xs)
    return (sum((x-m)**2 for x in xs)/(len(xs)-1))**0.5

def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float('nan')
    mx, my = mean(xs), mean(ys)
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = sum((x-mx)**2 for x in xs)**0.5
    dy = sum((y-my)**2 for y in ys)**0.5
    if dx == 0 or dy == 0: return float('nan')
    return num/(dx*dy)

CORPORA = ["openr1", "wikitext"]
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------------------
# Load Wave C
# ---------------------------------------------------------------------------
wc = {}
for corpus in CORPORA:
    for seed in SEEDS:
        path = f"{WC_DIR}/wC_lm_{corpus}_dm256_ds64_L2_s{seed}.json"
        wc[(corpus, seed)] = json.load(open(path))

print("=" * 90)
print("LOADED Wave C:", len(wc), "runs")
print("=" * 90)

# ---------------------------------------------------------------------------
# Q1: state-rank dynamics, reasoning vs narrative, contamination-conditioned
# ---------------------------------------------------------------------------
# Pool rank_stats_raw across checkpoints (all 7) and seeds (all 3), per corpus
# per layer, filtered to window_within_doc == True, frac == 1.0 (full-window
# occupancy -- the "most developed state" reading per window).
# Also compute the UNFILTERED (all windows, any frac) pool as a contrast to
# show the size of the contamination effect.

def collect_raw(corpus, frac_filter=None, within_doc_only=False, layer_filter=None,
                 step_filter=None):
    out = []  # each item: dict with step, layer, effective_rank, stable_rank, window_within_doc
    for seed in SEEDS:
        run = wc[(corpus, seed)]
        for ck in run["checkpoints"]:
            step = ck["step"]
            if step_filter is not None and step != step_filter:
                continue
            for r in ck["rank_stats_raw"][corpus]:
                if frac_filter is not None and r["frac"] != frac_filter:
                    continue
                if within_doc_only and not r["window_within_doc"]:
                    continue
                if layer_filter is not None and r["layer"] != layer_filter:
                    continue
                out.append({**r, "step": step, "seed": seed})
    return out

print("\n--- Q1: rank by corpus/layer, pooled across checkpoints+seeds ---")
print("(frac=1.0, window_within_doc filter as noted; d_state=64 is max rank)\n")

q1_summary = {}
for corpus in CORPORA:
    for layer in [0, 1]:
        filt = collect_raw(corpus, frac_filter=1.0, within_doc_only=True, layer_filter=layer)
        unfilt = collect_raw(corpus, frac_filter=1.0, within_doc_only=False, layer_filter=layer)
        er_f = [r["effective_rank"] for r in filt]
        sr_f = [r["stable_rank"] for r in filt]
        er_u = [r["effective_rank"] for r in unfilt]
        sr_u = [r["stable_rank"] for r in unfilt]
        q1_summary[(corpus, layer)] = dict(
            n_filt=len(filt), n_unfilt=len(unfilt),
            er_filt_mean=mean(er_f), er_filt_sd=sd(er_f),
            sr_filt_mean=mean(sr_f), sr_filt_sd=sd(sr_f),
            er_unfilt_mean=mean(er_u), sr_unfilt_mean=mean(sr_u),
        )
        print(f"corpus={corpus:9s} layer={layer}  "
              f"n(within-doc)={len(filt):4d}/{len(unfilt):4d}  "
              f"eff_rank(within-doc)={mean(er_f):6.2f}+/-{sd(er_f):5.2f}  "
              f"eff_rank(ALL windows)={mean(er_u):6.2f}  "
              f"stable_rank(within-doc)={mean(sr_f):5.3f}+/-{sd(sr_f):5.3f}  "
              f"stable_rank(ALL)={mean(sr_u):5.3f}")

print("\n--- Q1b: contamination severity by corpus (fraction of frac=1.0 windows crossing doc boundary) ---")
for corpus in CORPORA:
    all_w = collect_raw(corpus, frac_filter=1.0, within_doc_only=False, layer_filter=0)
    n_cross = sum(1 for r in all_w if not r["window_within_doc"])
    print(f"corpus={corpus:9s}  n={len(all_w)}  frac_crossing={n_cross/len(all_w):.3f}")

# ---------------------------------------------------------------------------
# Q1 / Q4: trajectory over training, per checkpoint step
# ---------------------------------------------------------------------------
print("\n--- Q1/Q4: rank trajectory over training (frac=1.0, within-doc filter), pooled across seeds ---")
traj = {}  # (corpus, layer) -> list of (step, mean_er, mean_sr, n, val_loss_mean)
steps_all = [ck["step"] for ck in wc[("openr1", 0)]["checkpoints"]]
for corpus in CORPORA:
    for layer in [0, 1]:
        rows = []
        for step in steps_all:
            filt = collect_raw(corpus, frac_filter=1.0, within_doc_only=True,
                                layer_filter=layer, step_filter=step)
            er = [r["effective_rank"] for r in filt]
            sr = [r["stable_rank"] for r in filt]
            vlosses = [wc[(corpus, s)]["checkpoints"][steps_all.index(step)]["val_loss"][corpus]
                       for s in SEEDS]
            rows.append((step, mean(er), mean(sr), len(er), mean(vlosses)))
        traj[(corpus, layer)] = rows
        print(f"\ncorpus={corpus} layer={layer}")
        print(f"{'step':>6s} {'n':>4s} {'eff_rank':>9s} {'stable_rank':>12s} {'val_loss(home)':>15s}")
        for step, erm, srm, n, vlm in rows:
            print(f"{step:6d} {n:4d} {erm:9.2f} {srm:12.3f} {vlm:15.4f}")

# correlation between rank trajectory and val loss trajectory (within corpus/layer)
print("\n--- Q4: correlation of eff_rank trajectory with home val_loss trajectory (pearson, n=7 checkpoints) ---")
for (corpus, layer), rows in traj.items():
    ers = [r[1] for r in rows]
    vls = [r[4] for r in rows]
    r = pearson(ers, vls)
    print(f"corpus={corpus:9s} layer={layer}  corr(eff_rank, val_loss)={r:+.3f}  "
          f"(eff_rank range {min(ers):.2f}->{max(ers):.2f}, val_loss range {max(vls):.3f}->{min(vls):.3f})")

# ---------------------------------------------------------------------------
# Q3: cross-corpus val loss asymmetry (final checkpoint, step 6103)
# ---------------------------------------------------------------------------
print("\n--- Q3: cross-corpus val loss, final checkpoint, mean +/- sd over 3 seeds ---")
q3 = {}
for train_c in CORPORA:
    for eval_c in CORPORA:
        vals = [wc[(train_c, s)]["checkpoints"][-1]["val_loss"][eval_c] for s in SEEDS]
        q3[(train_c, eval_c)] = (mean(vals), sd(vals), vals)
        tag = "HOME" if train_c == eval_c else "CROSS"
        print(f"train={train_c:9s} eval={eval_c:9s} [{tag:5s}]  "
              f"val_loss={mean(vals):.4f} +/- {sd(vals):.4f}   raw={[round(v,3) for v in vals]}")

# ---------------------------------------------------------------------------
# Load Wave D
# ---------------------------------------------------------------------------
wd = {}
for corpus in CORPORA:
    for seed in SEEDS:
        path = f"{WD_DIR}/wD_lm_{corpus}_s{seed}.json"
        wd[(corpus, seed)] = json.load(open(path))

print("\n" + "=" * 90)
print("LOADED Wave D:", len(wd), "runs")
print("=" * 90)

K_GRID = [8, 16, 24, 32, 48, 64]

# ---------------------------------------------------------------------------
# Q2 (headline): damage vs k, home-corpus (train==eval) comparison
# ---------------------------------------------------------------------------
print("\n--- Q2 HEADLINE: truncation damage vs k, HOME-corpus (train==eval), raw + frequency-balanced ---")
print(f"{'k':>4s} {'openr1 raw':>12s} {'openr1 bal':>12s} {'wikitext raw':>13s} {'wikitext bal':>13s} "
      f"{'raw diff(o-w)':>14s} {'bal diff(o-w)':>14s}")
q2_home = {}
for k in K_GRID:
    ks = str(k)
    o_raw = [wd[("openr1", s)]["damage_by_corpus_and_k"]["openr1"]["by_k"][ks]["raw_mean"] for s in SEEDS]
    o_bal = [wd[("openr1", s)]["damage_by_corpus_and_k"]["openr1"]["by_k"][ks]["frequency_balanced_mean"] for s in SEEDS]
    w_raw = [wd[("wikitext", s)]["damage_by_corpus_and_k"]["wikitext"]["by_k"][ks]["raw_mean"] for s in SEEDS]
    w_bal = [wd[("wikitext", s)]["damage_by_corpus_and_k"]["wikitext"]["by_k"][ks]["frequency_balanced_mean"] for s in SEEDS]
    q2_home[k] = dict(o_raw=o_raw, o_bal=o_bal, w_raw=w_raw, w_bal=w_bal)
    print(f"{k:4d} {mean(o_raw):8.4f}+/-{sd(o_raw):.4f} {mean(o_bal):8.4f}+/-{sd(o_bal):.4f} "
          f"{mean(w_raw):9.4f}+/-{sd(w_raw):.4f} {mean(w_bal):9.4f}+/-{sd(w_bal):.4f} "
          f"{mean(o_raw)-mean(w_raw):14.4f} {mean(o_bal)-mean(w_bal):14.4f}")

# ---------------------------------------------------------------------------
# Q2 full 2x2: train_corpus x eval_corpus, to separate "eval text needs rank"
# from "training regime shapes state usage"
# ---------------------------------------------------------------------------
print("\n--- Q2 full 2x2 (train_corpus x eval_corpus) damage vs k, raw_mean, mean over 3 seeds ---")
header = f"{'k':>4s}"
combos = [(tc, ec) for tc in CORPORA for ec in CORPORA]
for tc, ec in combos:
    header += f" {'tr='+tc[:4]+'/ev='+ec[:4]:>18s}"
print(header)
q2_full = {}
for k in K_GRID:
    ks = str(k)
    row = f"{k:4d}"
    for tc, ec in combos:
        vals = [wd[(tc, s)]["damage_by_corpus_and_k"][ec]["by_k"][ks]["raw_mean"] for s in SEEDS]
        q2_full[(tc, ec, k)] = vals
        row += f" {mean(vals):18.4f}"
    print(row)

print("\n--- Q2 full 2x2, frequency-balanced_mean ---")
print(header)
for k in K_GRID:
    ks = str(k)
    row = f"{k:4d}"
    for tc, ec in combos:
        vals = [wd[(tc, s)]["damage_by_corpus_and_k"][ec]["by_k"][ks]["frequency_balanced_mean"] for s in SEEDS]
        row += f" {mean(vals):18.4f}"
    print(row)

# ---------------------------------------------------------------------------
# Q2: k* -- first k at which damage collapses to near-baseline noise
# ---------------------------------------------------------------------------
print("\n--- Q2: k* (first k where |raw_mean| and |balanced_mean| both < 0.005 nats, home-corpus) ---")
NOISE_THRESH = 0.005
for corpus in CORPORA:
    kstar = None
    for k in K_GRID:
        ks = str(k)
        raws = [wd[(corpus, s)]["damage_by_corpus_and_k"][corpus]["by_k"][ks]["raw_mean"] for s in SEEDS]
        bals = [wd[(corpus, s)]["damage_by_corpus_and_k"][corpus]["by_k"][ks]["frequency_balanced_mean"] for s in SEEDS]
        if abs(mean(raws)) < NOISE_THRESH and abs(mean(bals)) < NOISE_THRESH:
            kstar = k
            break
    print(f"corpus={corpus:9s} k*={kstar} (d_state=64)")

# ---------------------------------------------------------------------------
# Q2: symbol vs word stratification, home corpus, k=8 and k=16
# ---------------------------------------------------------------------------
print("\n--- Q2: symbol vs word class damage, home-corpus, k=8 and k=16 (mean over 3 seeds, raw_mean per class) ---")
for k in [8, 16]:
    ks = str(k)
    print(f"\nk={k}")
    for corpus in CORPORA:
        classes = defaultdict(list)
        counts = defaultdict(list)
        for s in SEEDS:
            d = wd[(corpus, s)]["damage_by_corpus_and_k"][corpus]["by_k"][ks]
            for cls, val in d["by_symbol_word_class"].items():
                classes[cls].append(val)
            for cls, cnt in d["by_symbol_word_class_counts"].items():
                counts[cls].append(cnt)
        parts = []
        for cls in ["word", "symbol", "other"]:
            if cls in classes:
                parts.append(f"{cls}={mean(classes[cls]):.4f}(n~{mean(counts[cls]):.0f})")
        print(f"  corpus={corpus:9s} " + "  ".join(parts))

# ---------------------------------------------------------------------------
# Q2: frequency band stratification, home corpus, k=8,16 (bands 0=common..3=rare, unequal counts)
# ---------------------------------------------------------------------------
print("\n--- Q2: frequency-band damage, home-corpus, k=8 and k=16 (band idx 0..3, mean over 3 seeds) ---")
for k in [8, 16]:
    ks = str(k)
    print(f"\nk={k}")
    for corpus in CORPORA:
        bands = defaultdict(list)
        counts = defaultdict(list)
        for s in SEEDS:
            d = wd[(corpus, s)]["damage_by_corpus_and_k"][corpus]["by_k"][ks]
            for b, val in d["by_frequency_band"].items():
                bands[b].append(val)
            for b, cnt in d["by_frequency_band_counts"].items():
                counts[b].append(cnt)
        parts = []
        for b in sorted(bands.keys(), key=int):
            parts.append(f"band{b}={mean(bands[b]):+.4f}(n~{mean(counts[b]):.0f})")
        print(f"  corpus={corpus:9s} " + "  ".join(parts))

# ---------------------------------------------------------------------------
# Boundary contamination on Wave D eval windows themselves
# ---------------------------------------------------------------------------
print("\n--- Wave D eval-window boundary contamination (scored continuation) ---")
for corpus in CORPORA:
    for s in SEEDS:
        d = wd[(corpus, s)]["damage_by_corpus_and_k"][corpus]["windows"]["boundary_scored_continuation"]
        print(f"corpus={corpus:9s} seed={s}  frac_windows_crossing={d['frac_windows_crossing']:.4f}  "
              f"frac_tokens_cross_doc={d['frac_tokens_cross_doc']:.4f}")

print("\nDONE")
