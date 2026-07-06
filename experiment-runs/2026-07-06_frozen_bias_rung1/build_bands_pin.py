"""Driver: extract Arm-1 val loss + Arm-1'/Arm-1''/kraw span_frac per corpus per seed
from Phase-A's 18 retrofit JSONs + Arm-1's own 6 training JSONs, then call
bands_pinned_frozenbias.write_bands_pinned_frozenbias and validate. Run ON THE BOX.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bands_pinned_frozenbias import write_bands_pinned_frozenbias, validate_bands_pinned_frozenbias

# Reuse the CANONICAL pooling helper (n-weighted gram_deviation pooling across
# checkpoints/layers, THEN span_frac computed once from the pooled gram_deviation_mean --
# not a re-implementation) -- the SAME file frozen_bias_retrofit_eval_rd.py itself
# dynamically loads for its own ANCHORS_FN/SPAN_FRAC_FN.
_ANALYZE_PATH = "/home/nvidia/experiment-runs/2026-07-05_trackc_rung2/analyze_probe_wave2.py"
import importlib.util
_spec = importlib.util.spec_from_file_location("analyze_probe_wave2_pin", _ANALYZE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pooled_subset = _mod.pooled_subset
span_frac = _mod.span_frac

RESULTS_DIR = "results/frozen_bias_lm"
CORPORA = ["openr1-mix-ext", "wikitext-mix-ext"]
SEEDS = [0, 1, 2]

# Arm-1 (off) training result JSON naming convention.
def arm1_train_path(corpus, seed):
    return os.path.join(RESULTS_DIR, f"frozenbias_lm_off_lam0p00_{corpus}_dm256_ds64_L2_s{seed}.json")

def retrofit_path(mode, corpus_short, seed):
    return os.path.join(RESULTS_DIR, f"retrofit_{mode}_arm1_{corpus_short}_s{seed}.json")

CORPUS_SHORT = {"openr1-mix-ext": "openr1", "wikitext-mix-ext": "wikitext"}

def extract_val_loss():
    out = {}
    paths = {}
    for corpus in CORPORA:
        vals, ps = [], []
        for seed in SEEDS:
            p = arm1_train_path(corpus, seed)
            d = json.load(open(p))
            assert d["complete"], f"{p} not complete"
            vl = d["checkpoints"][-1]["val_loss"][corpus]
            vals.append(vl)
            ps.append(p)
        out[corpus] = vals
        paths[corpus] = ps
    return out, paths

def extract_span_frac(mode):
    """mode in {arm1prime, arm1double, kraw}. Each retrofit JSON covers ONE checkpoint,
    measured against BOTH corpora (per_corpus). We take the checkpoint's OWN TRAINED corpus
    slice, pooled via analyze_probe_wave2.py's own canonical pooled_subset (n-weighted
    gram_deviation pooling across layers, THEN span_frac computed once from the pooled
    gram_deviation_mean -- the SAME convention every archived span_frac figure uses, not a
    re-derivation)."""
    out = {}
    paths = {}
    for corpus in CORPORA:
        short = CORPUS_SHORT[corpus]
        vals, ps = [], []
        for seed in SEEDS:
            p = retrofit_path(mode, short, seed)
            d = json.load(open(p))
            # single checkpoint per call -> one entry in per_checkpoint
            assert len(d["per_checkpoint"]) == 1, (
                f"{p}: expected 1 checkpoint, got {len(d['per_checkpoint'])}")
            ckpt_entry = list(d["per_checkpoint"].values())[0]
            assert ckpt_entry["corpus_trained_on"] == corpus, (
                f"{p}: corpus_trained_on={ckpt_entry['corpus_trained_on']!r} != expected {corpus!r}")
            pooled = pooled_subset(d, [corpus])
            assert pooled is not None and pooled["n_scored"] > 0, (
                f"{p}: no scored episodes for corpus {corpus}")
            cfg = ckpt_entry["config"]
            K = d["chunk_size"]
            head_dim = cfg["d_state"] // cfg.get("num_heads", 1)
            sf = span_frac(pooled["gram_deviation_mean"], K, head_dim)
            vals.append(sf)
            ps.append(p)
        out[corpus] = vals
        paths[corpus] = ps
    return out, paths

if __name__ == "__main__":
    val_loss_vals, val_loss_paths = extract_val_loss()
    arm1prime_vals, arm1prime_paths = extract_span_frac("arm1prime")
    arm1double_vals, arm1double_paths = extract_span_frac("arm1double")
    kraw_vals, kraw_paths = extract_span_frac("kraw")

    print("=== Arm 1 val loss per corpus per seed ===")
    print(json.dumps(val_loss_vals, indent=2))
    print("=== Arm 1' (arm1prime) post-blend span_frac per corpus per seed ===")
    print(json.dumps(arm1prime_vals, indent=2))
    print("=== Arm 1'' (arm1double) post-blend span_frac per corpus per seed ===")
    print(json.dumps(arm1double_vals, indent=2))
    print("=== Arm 1 kraw (pre-blend) span_frac per corpus per seed ===")
    print(json.dumps(kraw_vals, indent=2))

    out_path = os.path.join(RESULTS_DIR, "BANDS_PINNED-FrozenBias.json")
    doc = write_bands_pinned_frozenbias(
        out_path,
        arm1_val_loss_per_corpus=val_loss_vals,
        arm1prime_span_frac_per_corpus=arm1prime_vals,
        arm1double_span_frac_per_corpus=arm1double_vals,
        arm1_kraw_span_frac_per_corpus=kraw_vals,
        arm1_result_paths=val_loss_paths,
        arm1prime_result_paths=arm1prime_paths,
        arm1double_result_paths=arm1double_paths,
        arm1_kraw_result_paths=kraw_paths,
    )
    print(f"\nwrote {out_path}, pinned_at_iso={doc['pinned_at_iso']}")

    # Validate immediately.
    revalidated = validate_bands_pinned_frozenbias(
        out_path, val_loss_vals, arm1prime_vals, arm1double_vals, kraw_vals)
    print(f"\nvalidate_bands_pinned_frozenbias -> {'OK (doc returned)' if revalidated is not None else 'FAILED (None)'}")
    assert revalidated is not None, "BANDS_PINNED-FrozenBias.json failed its own validator immediately after writing"

    print("\n=== PINNED BANDS SUMMARY ===")
    for corpus in CORPORA:
        print(f"\n-- {corpus} --")
        print(f"  val_loss_tolerance: {doc['arm1_val_loss_tolerance'][corpus]}")
        print(f"  arm1prime_bands (post-blend primary ref): {doc['arm1prime_span_frac_bands'][corpus]}")
        print(f"  arm1double_bands (post-blend control ref): {doc['arm1double_span_frac_bands'][corpus]}")
        print(f"  kraw_bands (pre-blend co-primary ref): {doc['arm1_kraw_span_frac_bands'][corpus]}")
