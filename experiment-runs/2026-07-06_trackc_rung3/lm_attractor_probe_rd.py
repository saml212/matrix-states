"""lm_attractor_probe_rd.py -- SCALE_TRANSFER_DESIGN.md sec 5.5 item 1
(Track C's write-geometry attractor probe, one of this build's three
required measurement-battery items alongside rank instrumentation and val
losses, both already in the harness -- see below): measures per-chunk RAW
(non-orthogonalized) key-Gram deviation on TRAINED checkpoints -- the SAME
"write-geometry attractor" instrument the exactness-mechanism study used
(model_rd.py's `gram_deviation`, C16's own instrument: "||Eff^T Eff - I||_F
... Applied to the (already L2-normalized) key side"), ported to
free-running LM text via a non-invasive forward hook -- Track B's own
lm_geo3_wave_neg1_gate.py pattern, cloned here (works on ANY archived
checkpoint, geo3-active or not, no model-code coupling needed, no
backward pass anywhere in this file).

Unlike Track B's geo3-in-LM construction (sec 4), this probe does NOT
select/orthogonalize/rewrite anything -- it is purely descriptive (Tier 2,
sec 2: "the trained state's measured geometry differs under condition X"),
run POST-HOC on already-trained checkpoints (Track C's rung-1 runs and the
required same-mix control cell, sec 5.5/5.6). It answers: "how far are
this checkpoint's own raw per-chunk write keys from mutually orthonormal"
-- the SAME attractor Wave 0/1 found at 13-14M params on a synthetic
grammar, and Wave C/exactness confirmed persists on real text at that one
fixed scale; Track C asks whether it persists across the scaling ladder
(rung 1 this session; rungs 2/3 are registered but out of this session's
build/launch scope, sec 5.3/lm_rd_rung_configs.py).

Scope note (sec 5.5's OTHER two items, both explicitly NOT built here):
item 2 (the frontier-probe transplant -- splicing the synthetic grammar's
K-cycle recovery task onto a pretrained LM's own embedding table) is a
separate, substantially larger, genuinely-untested-at-any-scale build item
with its own "mandatory validation step" gate (sec 5.5's own words) --
this session's task brief lists this probe's scope as "write-geometry
attractor probe ... rank instrumentation ... val losses", NOT the
transplant, so it is deliberately deferred, not silently dropped. Item 3
(the fix-effect / geo3-at-scale wave) is GATED OUT entirely this session:
Track B's own Wave -1 gate returned a HARD no-launch (EXPERIMENT_LOG.md,
"SCALE-TRANSFER Track B ... HARD NO-LAUNCH"), so there is no validated
geo3-in-LM construction to transplant to any rung. Rank instrumentation
(effective_rank/stable_rank of the WHOLE recurrent state) is ALREADY in
the harness (lm_pretrain_rd.py's sample_state_rank_stats, run
automatically at every training checkpoint) -- not duplicated here.

Usage:
  python lm_attractor_probe_rd.py --smoke
  python lm_attractor_probe_rd.py --checkpoints <path1.pt> [<path2.pt> ...] \\
      --data-dir /data/deltanet_rd_data --out results/lm_rd_trackc/attractor_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

from lm_pretrain_rd import (CORPUS_DIRS, DEFAULT_DATA_DIR, EOT_TOKEN_ID, GEO3_LM_CHUNK_SIZE_DEFAULT,
                             DeltaNetLM, corpus_fixed_seed, get_batch, load_corpus, set_and_log_tf32)
from model_rd import gram_deviation
from rank_utils import effective_rank, stable_rank

MIN_VALID_FOR_GRAM = 2   # a single item's own Gram-vs-identity deviation is trivially 0 by
                          # construction (not informative) -- episodes below this are reported with
                          # gram_deviation=None (excluded, counted), never silently averaged as 0


# ---------------------------------------------------------------------------
# Measurement: per-chunk raw-key Gram/rank statistics
# ---------------------------------------------------------------------------

def chunk_key_gram_stats(k_raw: torch.Tensor, content_mask: torch.Tensor, num_heads: int,
                          chunk_size: int, min_valid: int = MIN_VALID_FOR_GRAM) -> list[dict]:
    """k_raw: (B,T,d_state) RAW post-conv keys (pre-kernel-internal L2-norm
    -- the kernel's own use_qk_l2norm_in_kernel=True normalization happens
    INSIDE chunk_delta_rule and is not interceptable at this hook point;
    this probe L2-normalizes itself before computing Gram deviation,
    mirroring model_rd.py's own documented convention ("Applied to the
    ALREADY L2-normalized key side") and Track B's own
    _geo3_lm_select_and_orthogonalize's identical pre-normalization
    rationale -- a deliberate, documented measurement choice, not an
    oversight). content_mask: (B,T) bool, True=eligible (non-EOT/pad).

    Returns one record per (batch-row, chunk, head) episode -- a Python
    loop (not vectorized): episode counts here are O(hundreds) per
    (checkpoint, corpus, layer) cell (n_windows x n_chunks x num_heads),
    cheap for a post-hoc measurement tool, and the loop keeps EOT-exclusion
    (a variable-K-per-episode operation) trivially correct and auditable --
    mirrors sample_geo3_diagnostics' own per-(b,c,h,s) Python loop in this
    same codebase, not a novel pattern.

    Each record: {b, chunk, head, n_valid, gram_deviation, effective_rank,
    stable_rank} -- the latter three are None when n_valid < min_valid
    (excluded, COUNTED in the output, never silently treated as 0)."""
    B, T, D = k_raw.shape
    assert T % chunk_size == 0, f"T={T} not a multiple of chunk_size={chunk_size}"
    assert D % num_heads == 0, f"d_state={D} not divisible by num_heads={num_heads}"
    n_chunks = T // chunk_size
    head_dim = D // num_heads
    k_h = k_raw.view(B, n_chunks, chunk_size, num_heads, head_dim)
    mask_c = content_mask.view(B, n_chunks, chunk_size)

    records = []
    for b in range(B):
        for c in range(n_chunks):
            valid_idx = mask_c[b, c].nonzero(as_tuple=True)[0]
            n_valid = int(valid_idx.numel())
            for h in range(num_heads):
                if n_valid < min_valid:
                    records.append({"b": b, "chunk": c, "head": h, "n_valid": n_valid,
                                     "gram_deviation": None, "effective_rank": None, "stable_rank": None})
                    continue
                keys = k_h[b, c, valid_idx, h, :]                        # (n_valid, head_dim)
                keys_n = F.normalize(keys, dim=-1, eps=1e-8)
                gd = gram_deviation(keys_n.unsqueeze(0)).item()          # (1,) -> scalar
                er = effective_rank(keys_n.unsqueeze(0)).item()
                sr = stable_rank(keys_n.unsqueeze(0)).item()
                records.append({"b": b, "chunk": c, "head": h, "n_valid": n_valid,
                                 "gram_deviation": gd, "effective_rank": er, "stable_rank": sr})
    return records


def summarize_gram_records(records: list[dict]) -> dict:
    valid = [r for r in records if r["gram_deviation"] is not None]
    n_excluded = len(records) - len(valid)
    if not valid:
        return {"n_episodes": len(records), "n_excluded_below_min_valid": n_excluded,
                "n_scored": 0, "note": "every episode had n_valid < MIN_VALID_FOR_GRAM"}
    gd = torch.tensor([r["gram_deviation"] for r in valid])
    er = torch.tensor([r["effective_rank"] for r in valid])
    sr = torch.tensor([r["stable_rank"] for r in valid])
    return {
        "n_episodes": len(records), "n_excluded_below_min_valid": n_excluded, "n_scored": len(valid),
        "gram_deviation_mean": gd.mean().item(), "gram_deviation_std": gd.std(unbiased=False).item(),
        "effective_rank_mean": er.mean().item(), "stable_rank_mean": sr.mean().item(),
    }


# ---------------------------------------------------------------------------
# Checkpoint loading + non-invasive key capture (clones
# lm_geo3_wave_neg1_gate.py's load_checkpoint/capture_betas pattern exactly
# -- pod-safety convention, duplicated not cross-imported)
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: str) -> tuple[DeltaNetLM, dict]:
    ckpt = torch.load(path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def capture_raw_keys(model: DeltaNetLM, batches: list, device: str) -> tuple[dict, torch.Tensor]:
    """batches: list of (B,T) int64 token_id tensors. Returns
    (keys_by_layer: {layer_idx: (sum_B,T,d_state) RAW post-conv key tensor},
    token_ids_cat: (sum_B,T)). Non-invasive: hooks k_conv1d's OUTPUT
    (ShortConvolution returns (out, conv_state), mirrored here) -- no
    model-code change, works on any checkpoint."""
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


# ---------------------------------------------------------------------------
# Top-level measurement across checkpoints x corpora x layers
# ---------------------------------------------------------------------------

def run_measurement(checkpoint_paths: list, data_dir: str, chunk_size: int, n_windows: int,
                     batch_size: int, seq_len: int, device: str) -> dict:
    per_checkpoint = {}
    pooled_records: list[dict] = []          # every episode across every checkpoint/corpus/layer --
                                              # the persistence question (sec 5.7) is a PER-RUNG
                                              # comparison, so pooling across checkpoints of the SAME
                                              # rung is the caller's job (--checkpoints selects which
                                              # ones go into one call); this tool pools whatever it is
                                              # given, exactly like lm_geo3_wave_neg1_gate.py does.

    for ckpt_path in checkpoint_paths:
        model, ckpt = load_checkpoint(ckpt_path, device)
        num_heads = model.blocks[0].mixer.num_heads
        per_corpus = {}
        for corpus_name in sorted(CORPUS_DIRS):
            if not os.path.isdir(os.path.join(data_dir, CORPUS_DIRS[corpus_name])):
                continue   # sec 5.4: mix corpora may not exist yet on every box -- skip, don't crash
            _, val_tokens, meta, _, val_offs = load_corpus(data_dir, corpus_name, device)
            gen = torch.Generator(device=device).manual_seed(corpus_fixed_seed(corpus_name) + 95_000)
            n_batches = max(1, -(-n_windows // batch_size))               # ceil
            batches = [get_batch(val_tokens, batch_size, seq_len, gen)[0] for _ in range(n_batches)]
            keys_by_layer, token_ids_cat = capture_raw_keys(model, batches, device)
            content_mask = (token_ids_cat != EOT_TOKEN_ID)

            per_layer = {}
            for layer_idx, k_raw in keys_by_layer.items():
                records = chunk_key_gram_stats(k_raw, content_mask, num_heads, chunk_size)
                per_layer[layer_idx] = summarize_gram_records(records)
                for r in records:
                    pooled_records.append({**r, "checkpoint": ckpt_path, "corpus": corpus_name,
                                            "layer": layer_idx})
            per_corpus[corpus_name] = {"per_layer": per_layer,
                                        "n_windows_sampled": int(token_ids_cat.shape[0])}
        per_checkpoint[ckpt_path] = {
            "corpus_trained_on": ckpt.get("corpus"), "seed": ckpt.get("seed"), "step": ckpt.get("step"),
            "config": ckpt.get("config"),
            "n_params": sum(p.numel() for p in model.parameters()),
            "per_corpus": per_corpus,
        }
        # AUDIT-PATTERN REUSE (matches lm_geo3_wave_neg1_gate.py's own NIT-6 fix): free the
        # checkpoint's second full parameter copy + release CUDA cache between checkpoints -- this
        # tool is explicitly meant to be reused at Track C's 100M-1.3B rungs (sec 5.5), not just
        # today's smaller cells.
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled_summary = summarize_gram_records(pooled_records)
    return {
        "design_ref": "SCALE_TRANSFER_DESIGN.md sec 5.5 item 1 (attractor persistence probe)",
        "chunk_size": chunk_size, "n_windows_per_corpus_per_checkpoint": n_windows,
        "seq_len": seq_len, "checkpoints": checkpoint_paths,
        "measurement_convention_note": (
            "gram_deviation is computed on L2-NORMALIZED raw post-conv keys, PRE the kernel's own "
            "internal use_qk_l2norm_in_kernel normalization (not interceptable at this hook point) "
            "-- matches model_rd.py's own documented convention and Track B's identical "
            "pre-normalization rationale. Episodes with n_valid (non-EOT content positions in the "
            f"chunk) < MIN_VALID_FOR_GRAM={MIN_VALID_FOR_GRAM} are EXCLUDED (gram_deviation=None), "
            "counted in n_excluded_below_min_valid, never silently averaged as 0."),
        "pooled": pooled_summary,
        "per_checkpoint": per_checkpoint,
    }


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

def smoke(device: str):
    print("=" * 60 + "\n  LM_ATTRACTOR_PROBE_RD SMOKE GATE\n" + "=" * 60)

    print("\n[1] chunk_key_gram_stats POSITIVE control: a PERFECTLY ORTHONORMAL 4-key episode "
          "(head_dim=4, keys=I_4) must give gram_deviation~=0, effective_rank~=4, stable_rank~=4 "
          "(hand-computed exact values)")
    k1 = torch.eye(4).view(1, 4, 4)                                      # (B=1,T=4,D=4) -- num_heads=1 below
    content1 = torch.ones(1, 4, dtype=torch.bool)
    recs1 = chunk_key_gram_stats(k1, content1, num_heads=1, chunk_size=4)
    assert len(recs1) == 1
    r1 = recs1[0]
    assert r1["n_valid"] == 4
    assert abs(r1["gram_deviation"]) < 1e-5, r1
    assert abs(r1["effective_rank"] - 4.0) < 1e-3, r1
    assert abs(r1["stable_rank"] - 4.0) < 1e-3, r1
    print(f"  orthonormal I_4: gram_deviation={r1['gram_deviation']:.6f} (expected ~0), "
          f"effective_rank={r1['effective_rank']:.4f} (expected 4), "
          f"stable_rank={r1['stable_rank']:.4f} (expected 4)")

    print("\n[2] chunk_key_gram_stats NEGATIVE control: a RANK-COLLAPSED 4-key episode (all 4 keys "
          "IDENTICAL) must give gram_deviation=sqrt(12)~=3.4641 (hand-computed: Gram=ones(4,4), "
          "Gram-I has 12 off-diagonal 1s), effective_rank~=1, stable_rank~=1")
    k2 = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(1, 4, 4).contiguous()  # (B=1,T=4,D=4)
    content2 = torch.ones(1, 4, dtype=torch.bool)
    recs2 = chunk_key_gram_stats(k2, content2, num_heads=1, chunk_size=4)
    r2 = recs2[0]
    import math
    assert abs(r2["gram_deviation"] - math.sqrt(12)) < 1e-4, r2
    assert abs(r2["effective_rank"] - 1.0) < 1e-3, r2
    assert abs(r2["stable_rank"] - 1.0) < 1e-3, r2
    print(f"  rank-collapsed (4 identical keys): gram_deviation={r2['gram_deviation']:.6f} "
          f"(expected {math.sqrt(12):.6f}), effective_rank={r2['effective_rank']:.4f} (expected 1), "
          f"stable_rank={r2['stable_rank']:.4f} (expected 1) -- metric has teeth, correctly "
          f"distinguishes orthonormal from collapsed")

    print("\n[3] EOT-exclusion correctness: an EOT position holding an EXTREME outlier key value "
          "must be DROPPED from the episode entirely (n_valid reduced, gram_deviation IDENTICAL to "
          "the same episode built WITHOUT that position at all -- not zero-masked, not down-"
          "weighted)")
    k3 = torch.eye(4)
    k3_with_outlier = torch.cat([k3, torch.tensor([[999.0, -999.0, 0.0, 0.0]])], dim=0).view(1, 5, 4)
    content3 = torch.tensor([[True, True, True, True, False]])           # position 4 excluded (EOT)
    recs3 = chunk_key_gram_stats(k3_with_outlier, content3, num_heads=1, chunk_size=5)
    r3 = recs3[0]
    assert r3["n_valid"] == 4, r3
    assert abs(r3["gram_deviation"] - r1["gram_deviation"]) < 1e-5, (r3, r1)
    print(f"  EOT position (key=[999,-999,0,0], would dominate any non-exclusion-aware Gram) "
          f"correctly dropped: n_valid={r3['n_valid']} (of 5 raw positions), "
          f"gram_deviation={r3['gram_deviation']:.6f} == the no-EOT case ({r1['gram_deviation']:.6f})")

    print("\n[4] MIN_VALID_FOR_GRAM guard: an episode with only 1 content position must report "
          "gram_deviation=None (excluded, not silently scored as 0) -- summarize_gram_records must "
          "count it in n_excluded_below_min_valid, not n_scored")
    k4 = torch.eye(4)[:1].view(1, 1, 4)
    content4 = torch.ones(1, 1, dtype=torch.bool)
    recs4 = chunk_key_gram_stats(k4, content4, num_heads=1, chunk_size=1)
    assert recs4[0]["gram_deviation"] is None and recs4[0]["n_valid"] == 1, recs4
    summ4 = summarize_gram_records(recs4)
    assert summ4["n_scored"] == 0 and summ4["n_excluded_below_min_valid"] == 1, summ4
    print(f"  n_valid=1 episode: gram_deviation={recs4[0]['gram_deviation']!r} (None, correctly "
          f"excluded), summary n_scored={summ4['n_scored']} n_excluded={summ4['n_excluded_below_min_valid']}")

    print("\n[5] multi-head reshape correctness: num_heads=2 at d_state=8 (head_dim=4) must "
          "produce INDEPENDENT per-head episodes -- head 0 orthonormal (I_4), head 1 rank-collapsed "
          "(identical rows) -- and NOT mix the two heads' keys together")
    k5_h0 = torch.eye(4)
    k5_h1 = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).expand(4, 4)
    k5 = torch.cat([k5_h0, k5_h1], dim=1).view(1, 4, 1, 8)               # interleave as (T,1,2*4) -> reshape below
    # k_conv1d output layout is (B,T,d_state); reshape inside chunk_key_gram_stats is
    # view(B,n_chunks,chunk_size,num_heads,head_dim) -- so d_state's H*head_dim ordering must be
    # [head0_dims..., head1_dims...] PER POSITION, matching the mixer's own q/k/v reshape
    # convention (B,T,d_state)->(B,T,H,head_dim). Build k5 directly in that layout:
    k5 = torch.cat([k5_h0, k5_h1], dim=1).view(1, 4, 8)                  # (B=1,T=4,d_state=8)
    content5 = torch.ones(1, 4, dtype=torch.bool)
    recs5 = chunk_key_gram_stats(k5, content5, num_heads=2, chunk_size=4)
    r5_h0 = next(r for r in recs5 if r["head"] == 0)
    r5_h1 = next(r for r in recs5 if r["head"] == 1)
    assert abs(r5_h0["gram_deviation"]) < 1e-5, r5_h0
    assert abs(r5_h1["gram_deviation"] - math.sqrt(12)) < 1e-4, r5_h1
    print(f"  head 0 (orthonormal): gram_deviation={r5_h0['gram_deviation']:.6f} (expected ~0); "
          f"head 1 (collapsed): gram_deviation={r5_h1['gram_deviation']:.6f} (expected "
          f"{math.sqrt(12):.6f}) -- heads scored independently, no cross-head leakage")

    print("\n[6] run_measurement end-to-end on a TINY (but REAL-vocab-sized) SYNTHETIC (untrained) "
          "model + synthetic EOT-separated corpus -- shape/finiteness/JSON-serializability smoke "
          "(mirrors lm_geo3_wave_neg1_gate.py's own item [7] pattern), NOT a claim about real key "
          "geometry")
    import tempfile
    V6 = 50257                                                           # real vocab (load_corpus's
                                                                           # own literal EOT id assert)
    torch.manual_seed(0)
    model6 = DeltaNetLM(V6, d_model=32, d_state=64, n_layers=2, conv_size=4).to(device)
    tmpdir = tempfile.mkdtemp(prefix="attractor_probe_smoke_")
    ckpt_path6 = os.path.join(tmpdir, "fake_ckpt.pt")
    torch.save({"step": 0, "model_state_dict": model6.state_dict(), "config": model6.config(),
                "corpus": "openr1", "seed": 0, "run_name": "smoke"}, ckpt_path6)

    data_dir6 = os.path.join(tmpdir, "data")
    for name, dirname in CORPUS_DIRS.items():
        d = os.path.join(data_dir6, dirname)
        os.makedirs(d, exist_ok=True)
        n_tok = 20_000
        toks = torch.randint(0, V6, (n_tok,), dtype=torch.int64)
        offsets = [0]
        pos = 0
        while True:
            pos += 200
            if pos >= n_tok - 1:
                break
            toks[pos] = EOT_TOKEN_ID
            offsets.append(pos + 1)
        offsets_t = torch.tensor(offsets, dtype=torch.int64)
        torch.save(toks, os.path.join(d, "train.pt"))
        torch.save(toks, os.path.join(d, "val.pt"))
        torch.save(offsets_t, os.path.join(d, "train_doc_offsets.pt"))
        torch.save(offsets_t, os.path.join(d, "val_doc_offsets.pt"))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"vocab_size": V6, "tokenizer": "gpt2", "eot_separated": True}, f)

    result6 = run_measurement([ckpt_path6], data_dir6, chunk_size=64, n_windows=8, batch_size=4,
                               seq_len=128, device=device)
    json.dumps(result6)                                                  # must be JSON-serializable
    assert result6["pooled"]["n_scored"] > 0, "no episodes scored -- plumbing likely broken"
    print(f"  end-to-end smoke run completed: pooled gram_deviation_mean="
          f"{result6['pooled']['gram_deviation_mean']:.4f} (untrained model, NOT a real "
          f"measurement -- plumbing check only), n_scored={result6['pooled']['n_scored']}, "
          f"JSON-serializable")

    print("\n" + "=" * 60 + "\n  ALL LM_ATTRACTOR_PROBE_RD SMOKE CHECKS PASSED\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checkpoints", type=str, nargs="+", default=None,
                     help="one or more trained checkpoint .pt paths -- REQUIRED for a real "
                          "(non-smoke) run. Pool the same rung's own cells (e.g. 2 corpora x 3 "
                          "seeds) in one call for the persistence comparison across rungs, sec 5.7.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--chunk-size", type=int, default=GEO3_LM_CHUNK_SIZE_DEFAULT,
                     help="observation window for the Gram-deviation episode -- reuses Track B's "
                          "own chunk_size default (64) for cross-track consistency; this is a pure "
                          "measurement window, NOT coupled to any kernel tiling constant (no "
                          "orthogonalization happens in this tool).")
    ap.add_argument("--n-windows", type=int, default=32, help="eval windows sampled PER CORPUS PER CHECKPOINT")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512, help="must match the checkpoints' own training seq_len")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        smoke(device)
        return

    assert device == "cuda", "lm_attractor_probe_rd requires CUDA (chunk_delta_rule has no CPU path)"
    assert args.checkpoints, "--checkpoints is required for a real (non-smoke) run"
    assert args.seq_len % args.chunk_size == 0, \
        f"--seq-len={args.seq_len} must be a multiple of --chunk-size={args.chunk_size}"

    t0 = time.time()
    tf32_state = set_and_log_tf32()
    result = run_measurement(args.checkpoints, args.data_dir, args.chunk_size, args.n_windows,
                              args.batch_size, args.seq_len, device)
    result["tf32"] = tf32_state
    result["wall_s"] = time.time() - t0

    summary = {k: v for k, v in result.items() if k != "per_checkpoint"}
    print("\n" + "=" * 70)
    print("ATTRACTOR PROBE RESULT SUMMARY:", json.dumps(summary, indent=2))
    print("=" * 70)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
