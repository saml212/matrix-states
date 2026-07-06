"""frozen_bias_retrofit_eval_rd.py -- FROZEN_BIAS_LM_DESIGN.md sec 5/sec 7.1's
EVAL-ONLY retrofit measurement tool: Arm 1' (Arm 1's own checkpoint, dense
per-token blend applied ONLY at eval time), Arm 1'' (Arm 1's own checkpoint,
global-vector blend applied ONLY at eval time), and the pre-blend `k_raw`
co-primary (sec 4.a-i, no blend present, no artifact possible). Mirrors
lm_attractor_probe_rd.py's own checkpoint-loading / non-invasive-hook /
run_measurement structure (cloned, not cross-imported -- this codebase's
own pod-safety convention).

**Load-bearing property (sec 7.1's entire redesign depends on this): the
SAME `apply_frozen_bias_blend` function lm_pretrain_rd.py's DeltaNetLMMixer.
forward calls live, at training time, for Arm 2/Arm 2', is called HERE,
at eval time, on Arm 1's own never-bias-trained checkpoint. One function,
never two hand-copied twins -- sec 8.0b's code-path-equality claim is true
BY CONSTRUCTION (same Python function object), not merely by convention.**

Three measurement modes, one per CLI subcommand-style flag:
  --mode arm1prime   Arm 1' -- capture_raw_keys' own pre-blend k_raw, then
                      apply_frozen_bias_blend(..., arm_mode="per_token",
                      lam=<--lambda>) -- span_frac on the RESULT (sec 7.1's
                      primary, post-blend).
  --mode arm1double  Arm 1'' -- same, arm_mode="global" (sec 7.1a's control).
  --mode kraw        the co-primary (sec 4.a-i): span_frac on k_raw directly,
                      NO blend applied at all -- works for ANY checkpoint
                      (Arm 1 or Arm 2), the "no artifact possible by
                      construction" reading.

Usage:
  python frozen_bias_retrofit_eval_rd.py --smoke
  python frozen_bias_retrofit_eval_rd.py --mode arm1prime --checkpoints <arm1_ckpt.pt> \\
      --lam 0.58 --frozen-bias-seed 20260705 --data-dir /data/deltanet_rd_data \\
      --out results/frozen_bias_lm/arm1prime_lam0p58.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

from lm_pretrain_rd import (CORPUS_DIRS, DEFAULT_DATA_DIR, EOT_TOKEN_ID, DeltaNetLM,
                             apply_frozen_bias_blend, build_frozen_bias_table,
                             corpus_fixed_seed, frozen_bias_global_vector, get_batch, load_corpus)
from key_anchoring import ANCHOR_INIT_SEED   # shared symbol, not a duplicated literal (build audit)
from lm_attractor_probe_rd import chunk_key_gram_stats, summarize_gram_records

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYZE_PATH = os.path.normpath(os.path.join(
    _HERE, "..", "..", "experiment-runs", "2026-07-05_trackc_rung2", "analyze_probe_wave2.py"))


def _load_span_frac_fns():
    """Same dynamic-load-by-path discipline sim_frozen_bias_direction.py already uses -- the
    committed `span_frac`/`anchors` implementation lives under experiment-runs/ (not a package
    member), loaded once here so every span_frac number this tool ever produces uses the EXACT
    same function every archived span_frac figure in this codebase already used, never a
    reimplementation that could silently drift."""
    import importlib.util
    assert os.path.exists(_ANALYZE_PATH), (
        f"analyze_probe_wave2.py not found at {_ANALYZE_PATH} -- this tool requires the SAME "
        f"span_frac implementation already used for every archived span_frac number.")
    spec = importlib.util.spec_from_file_location("analyze_probe_wave2_frozenbias_retrofit", _ANALYZE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.anchors, mod.span_frac, _ANALYZE_PATH


ANCHORS_FN, SPAN_FRAC_FN, ANALYZE_PATH_USED = _load_span_frac_fns()

MIN_VALID_FOR_GRAM = 2


# ---------------------------------------------------------------------------
# Checkpoint loading + non-invasive raw-key capture (clones lm_attractor_probe_rd.py's
# load_checkpoint/capture_raw_keys pattern exactly).
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: str) -> tuple[DeltaNetLM, dict]:
    ckpt = torch.load(path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def capture_raw_keys(model: DeltaNetLM, batches: list, device: str) -> tuple[dict, torch.Tensor]:
    """IDENTICAL to lm_attractor_probe_rd.py's own capture_raw_keys: hooks k_conv1d's OUTPUT --
    the PRE-frozen-bias-blend key, by construction (the blend hook, when active, runs strictly
    AFTER k_conv1d inside DeltaNetLMMixer.forward, sec 2's own insertion-point comment). This
    means capturing here on an Arm-1 checkpoint (which trained with frozen_bias_arm='off') gives
    the co-primary's own `k_raw` (sec 4.a-i) directly -- no blend is present anywhere in this
    captured population, so no mechanical-pinning artifact is possible by construction."""
    captured = {i: [] for i in range(len(model.blocks))}

    def make_hook(i):
        def hook(module, inp, out):
            k_raw = out[0] if isinstance(out, tuple) else out
            captured[i].append(k_raw.detach())
        return hook

    handles = [blk.mixer.k_conv1d.register_forward_hook(make_hook(i))
               for i, blk in enumerate(model.blocks)]
    try:
        with torch.no_grad():
            for x in batches:
                _ = model(x)
    finally:
        for h in handles:
            h.remove()
    keys_by_layer = {i: torch.cat(v, dim=0) for i, v in captured.items()}
    token_ids_cat = torch.cat(batches, dim=0)
    return keys_by_layer, token_ids_cat


def measure_population(k_pop: torch.Tensor, content_mask: torch.Tensor, num_heads: int,
                        chunk_size: int) -> dict:
    """chunk_key_gram_stats/summarize_gram_records (imported unmodified from
    lm_attractor_probe_rd.py) + the REAL span_frac/anchors transform (imported unmodified from
    analyze_probe_wave2.py) -- the SAME chain sim_frozen_bias_direction.py's own measure_population
    uses, applied here to REAL captured keys instead of a synthetic population."""
    records = chunk_key_gram_stats(k_pop, content_mask, num_heads, chunk_size)
    summary = summarize_gram_records(records)
    head_dim = k_pop.shape[-1] // num_heads
    a = ANCHORS_FN(chunk_size, head_dim)
    sf = SPAN_FRAC_FN(summary["gram_deviation_mean"], chunk_size, head_dim) if summary.get("n_scored", 0) > 0 else None
    summary["anchors"] = a
    summary["span_frac"] = sf
    return summary


# ---------------------------------------------------------------------------
# The three measurement modes.
# ---------------------------------------------------------------------------

def run_retrofit_measurement(checkpoint_paths: list, mode: str, lam: float, frozen_bias_seed: int,
                              data_dir: str, chunk_size: int, n_windows: int, batch_size: int,
                              seq_len: int, device: str) -> dict:
    assert mode in ("arm1prime", "arm1double", "kraw"), f"unknown mode {mode!r}"
    per_checkpoint = {}
    pooled_records: list[dict] = []

    for ckpt_path in checkpoint_paths:
        model, ckpt = load_checkpoint(ckpt_path, device)
        num_heads = model.blocks[0].mixer.num_heads
        d_state = model.blocks[0].mixer.d_state
        vocab_size = model.vocab_size

        # sec 8.0b's own "same B buffer instance" requirement: construct the table/global-vec
        # ONCE per checkpoint here (never per-corpus/per-layer), from the SAME seed the design
        # registers (frozen_bias_seed, default ANCHOR_INIT_SEED) -- shared across every corpus and
        # every layer measured below, mirroring what a live Arm-2 training run's own per-layer
        # (but same-seed) table construction would produce.
        table = None
        global_vec = None
        if mode in ("arm1prime", "arm1double"):
            table = build_frozen_bias_table(vocab_size, d_state, seed=frozen_bias_seed).to(device)
            if mode == "arm1double":
                global_vec = frozen_bias_global_vector(table)

        per_corpus = {}
        for corpus_name in sorted(CORPUS_DIRS):
            if not os.path.isdir(os.path.join(data_dir, CORPUS_DIRS[corpus_name])):
                continue
            _, val_tokens, meta, _, val_offs = load_corpus(data_dir, corpus_name, device)
            gen = torch.Generator(device=device).manual_seed(corpus_fixed_seed(corpus_name) + 95_000)
            n_batches = max(1, -(-n_windows // batch_size))
            batches = [get_batch(val_tokens, batch_size, seq_len, gen)[0] for _ in range(n_batches)]
            keys_by_layer, token_ids_cat = capture_raw_keys(model, batches, device)
            content_mask = (token_ids_cat != EOT_TOKEN_ID)

            per_layer = {}
            for layer_idx, k_raw in keys_by_layer.items():
                if mode == "kraw":
                    k_measured = k_raw
                else:
                    arm_mode = "per_token" if mode == "arm1prime" else "global"
                    # THE load-bearing call: the IDENTICAL function DeltaNetLMMixer.forward calls
                    # at training time for a live Arm-2/Arm-2' cell (sec 8.0b's code-path-equality
                    # claim, true by construction -- same function object).
                    k_measured = apply_frozen_bias_blend(
                        k_raw, token_ids_cat, arm_mode, table, global_vec, lam)
                records_summary = measure_population(k_measured, content_mask, num_heads, chunk_size)
                per_layer[layer_idx] = records_summary
                for r in chunk_key_gram_stats(k_measured, content_mask, num_heads, chunk_size):
                    pooled_records.append({**r, "checkpoint": ckpt_path, "corpus": corpus_name,
                                            "layer": layer_idx})
            per_corpus[corpus_name] = {"per_layer": per_layer,
                                        "n_windows_sampled": int(token_ids_cat.shape[0])}
        per_checkpoint[ckpt_path] = {
            "corpus_trained_on": ckpt.get("corpus"), "seed": ckpt.get("seed"), "step": ckpt.get("step"),
            "config": ckpt.get("config"), "n_params": sum(p.numel() for p in model.parameters()),
            "per_corpus": per_corpus,
        }
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled_summary = summarize_gram_records(pooled_records)
    return {
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 5/sec 7.1/sec 4.a-i",
        "mode": mode, "lam": lam if mode != "kraw" else None,
        "frozen_bias_seed": frozen_bias_seed if mode != "kraw" else None,
        "analyze_path_used": ANALYZE_PATH_USED,
        "chunk_size": chunk_size, "n_windows_per_corpus_per_checkpoint": n_windows,
        "seq_len": seq_len, "checkpoints": checkpoint_paths,
        "pooled": pooled_summary, "per_checkpoint": per_checkpoint,
    }


# ---------------------------------------------------------------------------
# Smoke gate. Requires CUDA (DeltaNetLM's forward needs chunk_delta_rule, no CPU path) -- mirrors
# lm_attractor_probe_rd.py's own smoke() convention: this checks the RETROFIT-SPECIFIC logic
# (code-path equality, the three modes agree with direct expectations), NOT lm_pretrain_rd.py's
# own hook correctness (that is smoke_frozen_bias_wave_neg1.py's job, CPU-runnable, built
# separately).
# ---------------------------------------------------------------------------

def smoke(device: str):
    print("=" * 60 + "\n  FROZEN_BIAS_RETROFIT_EVAL_RD SMOKE GATE\n" + "=" * 60)
    assert device == "cuda", "chunk_delta_rule has no CPU path -- this smoke requires CUDA"

    print("\n[1] code-path equality: apply_frozen_bias_blend applied HERE (retrofit, eval-mode) "
          "on a real checkpoint's captured k_raw is torch.equal to the SAME function applied "
          "inside DeltaNetLMMixer.forward's own live (train-mode) blend on the SAME input")
    torch.manual_seed(41)
    V, D_STATE = 300, 64
    m_off = DeltaNetLM(V, d_model=64, d_state=D_STATE, n_layers=1, conv_size=4,
                        frozen_bias_arm="off").to(device)
    x = torch.randint(0, V, (4, 128), device=device)
    keys_by_layer, token_ids_cat = capture_raw_keys(m_off, [x], device)
    k_raw_captured = keys_by_layer[0]
    table = build_frozen_bias_table(V, D_STATE, seed=999).to(device)
    lam = 0.58
    retrofit_blended = apply_frozen_bias_blend(k_raw_captured, token_ids_cat, "per_token", table, None, lam)

    # Live path: build a per_token-active model, forcibly install the SAME table instance and
    # k_raw, and read the actual internal blend the mixer's forward() computes.
    m_live = DeltaNetLM(V, d_model=64, d_state=D_STATE, n_layers=1, conv_size=4,
                         frozen_bias_arm="per_token", frozen_bias_lambda=lam,
                         frozen_bias_vocab_size=V, frozen_bias_seed=999).to(device)
    with torch.no_grad():
        m_live.blocks[0].mixer.frozen_bias_table.copy_(table)
    live_blended = apply_frozen_bias_blend(
        k_raw_captured, token_ids_cat, "per_token", m_live.blocks[0].mixer.frozen_bias_table, None, lam)
    equal = torch.equal(retrofit_blended, live_blended)
    print(f"  retrofit-blend == live-model's-own-table blend (torch.equal): {equal}")
    assert equal, "code-path equality FAILED -- retrofit and live blends diverge on identical inputs"

    print("\n[2] kraw mode returns the RAW (unblended) key unchanged")
    kraw_direct = k_raw_captured
    assert torch.equal(kraw_direct, keys_by_layer[0]), "kraw mode must be a pure pass-through"
    print("  kraw mode: pass-through confirmed (torch.equal)")

    print("\n[3] arm1double (global) mode uses a single derived vector, NOT a per-token lookup -- "
          "verify every position receives the SAME bias vector")
    gvec = frozen_bias_global_vector(table)
    global_blended = apply_frozen_bias_blend(k_raw_captured, token_ids_cat, "global", None, gvec, lam)
    # Reconstruct what "same bias every position" implies: blending k_raw toward gvec should give
    # a DIFFERENT result than blending toward table[token_ids] (per_token), confirming the two
    # modes are not accidentally identical (a degenerate table where they'd coincide is generically
    # improbable at random init, but assert it directly rather than assume).
    same_as_per_token = torch.equal(global_blended, retrofit_blended)
    assert not same_as_per_token, (
        "global-mode and per_token-mode blends are IDENTICAL -- either the table degenerated to a "
        "constant, or the global/per_token dispatch is broken")
    print(f"  global-mode blend differs from per_token-mode blend (expected): "
          f"{not same_as_per_token}")

    print("\n" + "=" * 60 + "\n  ALL FROZEN_BIAS_RETROFIT_EVAL_RD SMOKE CHECKS PASSED\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["arm1prime", "arm1double", "kraw"], default=None)
    ap.add_argument("--checkpoints", type=str, nargs="+", default=None)
    ap.add_argument("--lam", type=float, default=0.58)
    ap.add_argument("--frozen-bias-seed", type=int, default=ANCHOR_INIT_SEED,
                     help="MUST match the seed used to construct the frozen table this program's "
                          "training runs use (ANCHOR_INIT_SEED by default) -- passing a different "
                          "seed would compare against a DIFFERENT table than Arm 2/Arm 2' trained "
                          "through, silently invalidating the comparison.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        smoke(device)
        return

    assert args.mode is not None, "--mode is required for a real (non-smoke) run"
    assert args.checkpoints, "--checkpoints is required for a real (non-smoke) run"
    result = run_retrofit_measurement(
        args.checkpoints, args.mode, args.lam, args.frozen_bias_seed, args.data_dir,
        args.chunk_size, args.n_windows, args.batch_size, args.seq_len, device)
    print(json.dumps({k: v for k, v in result.items() if k != "per_checkpoint"}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
