"""frozen_bias_token_identity_probe.py -- FROZEN_BIAS_LM_DESIGN.md sec 12.4
Stage 1: the primary H1/H5 mechanism-wave statistic, `repeat_excess`, on
already-existing frozen-bias LM checkpoints (Arm 1 "off", Arm 2 "per_token",
Arm 2' "global"). Gated on sec 12.3.4 Stage 0.5's self-test pass
(mech_stage05_selftests.py; already RUN and PASSED, sec 12.10.1 -- this
script's own --self-test below is a SEPARATE, complementary check: sec
12.3.4 validated the FORMULA against hand-picked constructions; this file's
--self-test pushes those SAME constructions through THIS script's OWN
episode-grained plumbing (token_identity_records), closing sec 12.9 item 4's
"formula-vs-plumbing gap" -- not a re-run of the formula-level test, and not
a substitute for it.

**repeat_excess, precisely (sec 12.4):** within each (b, chunk, head)
episode -- IDENTICAL chunking to lm_attractor_probe_rd.chunk_key_gram_stats
(chunk_size=64, content_mask excludes EOT exactly as today) -- among the
n_valid content positions, group by token id via token_ids_cat (joined by
position, the ALREADY-AVAILABLE alignment capture_raw_keys already returns).
Require >=1 pair (i,j), i!=j, tok(i)=tok(j) (a genuine within-chunk repeat)
AND >=1 pair with tok(i)!=tok(j), else the episode is EXCLUDED and COUNTED
(never silently zeroed, per the standing "don't silently treat undefined as
0" [LEARN]). On L2-normalized keys (same pre-normalization chunk_key_gram_
stats already applies):

    same_tok_sim = mean over all ordered (i,j), i!=j, tok(i)=tok(j), of cos(k_i,k_j)
    diff_tok_sim = mean over all ordered (i,j), tok(i)!=tok(j), of cos(k_i,k_j)
    repeat_excess = same_tok_sim - diff_tok_sim

pooled across episodes EXACTLY the way lm_attractor_probe_rd.
summarize_gram_records pools gram_deviation -- a PLAIN (unweighted) mean
over INCLUDED episodes' own per-episode values (summarize_gram_records's own
code does `gd.mean()` over included episodes, NOT an n_valid-reweighted
mean, despite the design doc's own prose calling this "n-weighted" pooling
-- this script replicates summarize_gram_records's literal CODE SHAPE, per
this build task's own instruction to match that function's pooling "exactly",
not a re-derivation of what "n-weighted" might otherwise mean; see this
file's own BUILD REPORT for the full ambiguity note).

**kraw discipline (sec 12.4's "pre-blend k_raw -- the artifact-free
population, same capture point the existing co-primary kraw retrofit
uses"):** capture_raw_keys hooks k_conv1d's OUTPUT, which is STRICTLY BEFORE
apply_frozen_bias_blend runs inside that SAME forward pass (sec 2's own
insertion-point comment) -- true for EVERY arm's checkpoint (Arm 1 never
blends at all; Arm 2/Arm 2' blend AFTER this hook point, inside the SAME
forward call), so this script never needs to call apply_frozen_bias_blend
anywhere -- the captured population is "pre-blend" by construction of the
hook's insertion point, identically to frozen_bias_retrofit_eval_rd.py's own
`--mode kraw` path.

**H5 (frequency confound):** reuses the EXACT SAME captured (k_raw,
token_ids_cat) pairs token_identity_records already computed -- stratifies
by whether tok(i) (the "anchor"/row index of each ordered pair) falls in the
top-20 most frequent CONTENT token ids in THIS captured sample (recomputed
fresh per checkpoint pass, never hard-coded) vs. the rest, reporting both
strata's own repeat_excess separately.

**Yield floor (sec 12.4 MINOR-4 fix, registered minimum n=50):** this
script reports the pooled INCLUDED (non-excluded) episode count for this
invocation's own requested arms x steps (one corpus, one seed per
invocation, per this build task's spec) and sets
`under_powered_insufficient_yield=true` if that count is < 50 -- a
registered floor decided now, not a post-hoc call. Sec 12.4's "STOP and
report as under-powered" is implemented as a DISCLOSURE flag
(`repeat_excess_trust_status` in the output JSON), not a runtime halt --
the run still completes and reports everything, but the flag bars the
number from being read as a trusted result (audit 2026-07-07 MINOR-3
clarification). NOTE: the full sec 12.4 descriptive pass spans BOTH
corpora (2 separate invocations of this script, 2 JSON files) -- checking
the FULL descriptive-pass floor requires summing this field across both
corpora's output JSONs; this script's own flag is scoped honestly to what
a single invocation can see.

**Trajectory sub-study (sec 12.4.1):** when >=2 steps are requested, this
script computes the mechanical adjacent-checkpoint sign-flip ambiguity
check (step 2's own frozen definition: ambiguous iff the delta's sign flips
between any two ADJACENT sampled steps) for both comparisons (Arm2-vs-Arm1,
Arm2'-vs-Arm1), on the pre-blend kraw repeat_excess delta trajectory.

Checkpoint path convention -- IDENTICAL to mech_h4_paramdiff.py's own
`_resolve_ckpt_path` (verified against frozen_bias_lm_sweep.py's cell_name()
and lm_pretrain_rd.py's run_name construction):

    {ckpt_base_dir}/frozenbias_lm_{arm}_{lam_tag}_{corpus}_dm256_ds64_L2_s{seed}/
        lmC_{corpus}_dm256_ds64_L2_s{seed}_step{step}.pt

Persisted through mech_schema.wrap_exploratory (sec 12.3.1's schema
requirement, binding on every sec 12 artifact).

Usage:
    # pure-CPU gate, no checkpoints needed:
    python3 frozen_bias_token_identity_probe.py --self-test

    # descriptive pass, one corpus, one seed, all 3 arms, step 20000:
    python3 frozen_bias_token_identity_probe.py \\
        --ckpt-base-dir /data/deltanet_rd_frozenbias_ckpts \\
        --corpus openr1-mix-ext --seed 0 --steps 20000 \\
        --out results/mech_wave/mech_stage1_openr1-mix-ext_s0.json

    # trajectory sub-study, openr1-mix-ext only:
    python3 frozen_bias_token_identity_probe.py \\
        --ckpt-base-dir /data/deltanet_rd_frozenbias_ckpts \\
        --corpus openr1-mix-ext --seed 0 --steps 1000,5000,10000,15000,20000 \\
        --out results/mech_wave/mech_stage1_trajectory_openr1-mix-ext_s0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

from mech_schema import wrap_exploratory   # pure Python, no fla/torch-model dependency -- safe at import time

# NOTE ON DEFERRED IMPORTS (local-CPU-verify discipline, this build's own VERIFY requirement):
# lm_pretrain_rd.py imports `fla` (flash-linear-attention) at ITS OWN module level -- a
# box-only/GPU-environment package, not installed on the dev Mac. --self-test must be runnable
# "pure CPU, no checkpoints" (this script's own spec) with ZERO dependence on fla being
# importable at all. Every name this file needs from lm_pretrain_rd/lm_attractor_probe_rd is
# therefore imported LAZILY, inside the specific real-run function that needs it (load_checkpoint,
# measure_population, run_one_checkpoint_pass) -- never at module level -- so `--self-test`
# executes end-to-end on a torch-only environment. `from __future__ import annotations` (top of
# this file) makes every type hint referencing DeltaNetLM a lazy string, so those hints never
# force an eager import either.

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "mech_wave")
MIN_VALID_FOR_GRAM = 2   # same threshold lm_attractor_probe_rd.py/frozen_bias_retrofit_eval_rd.py use
YIELD_FLOOR = 50          # sec 12.4 MINOR-4 fix: registered minimum pooled included-episode count
TOP_K_FREQ = 20            # H5: top-20-most-frequent-tokens stratification

ARM_LAM_TAG = {"off": "lam0p00", "per_token": "lam0p58", "global": "lam0p58"}
ALL_ARMS = ("off", "per_token", "global")

# Mirrors lm_pretrain_rd.DEFAULT_DATA_DIR's own literal value (verified by direct source read at
# build time) -- hardcoded here, NOT imported, so building the argparse default in main() never
# needs lm_pretrain_rd (and therefore never needs fla) even when the invocation is --self-test.
_DEFAULT_DATA_DIR_FALLBACK = "/data/deltanet_rd_data"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYZE_PATH = os.path.normpath(os.path.join(
    _HERE, "..", "..", "experiment-runs", "2026-07-05_trackc_rung2", "analyze_probe_wave2.py"))

_SPAN_FRAC_CACHE: dict = {}


def _load_span_frac_fns():
    """IDENTICAL dynamic-load-by-path discipline frozen_bias_retrofit_eval_rd.py already uses --
    every span_frac number this tool ever produces must use the EXACT same function every
    archived span_frac figure in this codebase already used, never a reimplementation that
    could silently drift. Cached after first call (LAZY -- only invoked from measure_population,
    a real-checkpoint-run-only code path, never from --self-test)."""
    if not _SPAN_FRAC_CACHE:
        import importlib.util
        assert os.path.exists(_ANALYZE_PATH), (
            f"analyze_probe_wave2.py not found at {_ANALYZE_PATH} -- this tool requires the SAME "
            f"span_frac implementation already used for every archived span_frac number.")
        spec = importlib.util.spec_from_file_location("analyze_probe_wave2_token_identity_probe", _ANALYZE_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _SPAN_FRAC_CACHE["anchors"] = mod.anchors
        _SPAN_FRAC_CACHE["span_frac"] = mod.span_frac
        _SPAN_FRAC_CACHE["path"] = _ANALYZE_PATH
    return _SPAN_FRAC_CACHE["anchors"], _SPAN_FRAC_CACHE["span_frac"], _SPAN_FRAC_CACHE["path"]


# ---------------------------------------------------------------------------
# Checkpoint loading + non-invasive raw-key capture -- CLONED VERBATIM from
# frozen_bias_retrofit_eval_rd.py's load_checkpoint/capture_raw_keys (this
# codebase's pod-safety convention: duplicate, never cross-import, per
# lm_attractor_probe_rd.py's own precedent that file itself follows).
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: str) -> tuple[DeltaNetLM, dict]:
    from lm_pretrain_rd import DeltaNetLM   # deferred (fla dependency) -- see module docstring
    ckpt = torch.load(path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def capture_raw_keys(model: DeltaNetLM, batches: list, device: str) -> tuple[dict, torch.Tensor]:
    """IDENTICAL to frozen_bias_retrofit_eval_rd.py's own capture_raw_keys: hooks k_conv1d's
    OUTPUT -- the PRE-frozen-bias-blend key, by construction (the blend hook, when active, runs
    strictly AFTER k_conv1d inside DeltaNetLMMixer.forward). This means capturing here on ANY
    checkpoint (Arm 1/2/2') gives the co-primary's own `k_raw` (sec 4.a-i) directly -- no blend
    is present anywhere in this captured population, so no mechanical-pinning artifact is
    possible by construction."""
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
                        chunk_size: int) -> tuple[dict, list]:
    """CLONED from frozen_bias_retrofit_eval_rd.py's own measure_population -- the free
    gram_deviation/span_frac cross-check computed on the SAME k_raw population repeat_excess is
    computed on (sec 12.4.1's "free cross-check"). Returns (summary_dict, raw_records) -- the
    raw per-(b,chunk,head) records are needed here (unlike the retrofit tool) for the
    repeat_excess-vs-gram_deviation per-episode correlation byproduct."""
    from lm_attractor_probe_rd import chunk_key_gram_stats, summarize_gram_records   # deferred, see module docstring
    anchors_fn, span_frac_fn, _ = _load_span_frac_fns()
    records = chunk_key_gram_stats(k_pop, content_mask, num_heads, chunk_size)
    summary = summarize_gram_records(records)
    head_dim = k_pop.shape[-1] // num_heads
    a = anchors_fn(chunk_size, head_dim)
    sf = span_frac_fn(summary["gram_deviation_mean"], chunk_size, head_dim) if summary.get("n_scored", 0) > 0 else None
    summary["anchors"] = a
    summary["span_frac"] = sf
    return summary, records


# ---------------------------------------------------------------------------
# repeat_excess -- THE probe's own new statistic (sec 12.4). mech_stage05_
# selftests.py already pinned the FORMULA against hand-picked vectors via its
# own flat repeat_excess_stats() reference implementation; THIS function is
# the REAL episode-grained plumbing (sec 12.9 item 4's formula-vs-plumbing
# gap) -- --self-test below pushes the SAME pinned constructions through
# THIS code path, not through repeat_excess_stats().
# ---------------------------------------------------------------------------

def compute_top_k_frequent(token_ids_cat: torch.Tensor, content_mask: torch.Tensor,
                            k: int = TOP_K_FREQ) -> set:
    """H5's own frequency stratification set -- top-k most frequent CONTENT (non-EOT) token ids
    in THIS captured sample (sec 12.4: "top-20 most frequent tokens in THIS sample"), recomputed
    fresh per checkpoint pass, never hard-coded. If fewer than k unique content tokens exist,
    takes all of them (disclosed via the returned set's own size, not silently padded)."""
    valid_tokens = token_ids_cat[content_mask]
    if valid_tokens.numel() == 0:
        return set()
    ids, counts = torch.unique(valid_tokens, return_counts=True)
    order = torch.argsort(counts, descending=True)
    k_eff = min(k, ids.numel())
    return set(ids[order[:k_eff]].tolist())


def token_identity_records(k_pop: torch.Tensor, token_ids_cat: torch.Tensor,
                            content_mask: torch.Tensor, num_heads: int, chunk_size: int,
                            top20_ids: set, min_valid: int = MIN_VALID_FOR_GRAM) -> list[dict]:
    """Sec 12.4's repeat_excess, computed per (b, chunk, head) episode -- SAME chunking as
    chunk_key_gram_stats (asserted identical shapes/reshape below), token ids joined by
    position via token_ids_cat (aligned to the SAME (B,T) axes as k_pop, per capture_raw_keys's
    own contract -- the join is a reshape, not new plumbing, per sec 12.4's own text).

    k_pop: (B,T,d_state) RAW post-conv keys (pre-kernel-internal L2-norm), L2-normalized HERE
    (same pre-normalization chunk_key_gram_stats already applies). token_ids_cat: (B,T) int64.
    content_mask: (B,T) bool, True=eligible (non-EOT/pad). top20_ids: H5's frequency-stratum set
    (may be empty -- in that case both H5 strata are marked excluded/"no_top20_set", not
    silently omitted).

    Returns one record per (b, chunk, head) episode:
      {b, chunk, head, n_valid, excluded, exclusion_reason,
       same_tok_sim, diff_tok_sim, repeat_excess, n_same, n_diff,
       stratum_top20: {...same shape...}, stratum_rest: {...same shape...}}
    `excluded` is True (and same_tok_sim/diff_tok_sim/repeat_excess are None, n_same/n_diff still
    reported) whenever n_valid < min_valid, OR no same-token pair exists, OR no diff-token pair
    exists -- COUNTED, never silently zeroed, per the standing "don't silently treat undefined as
    0" [LEARN]."""
    B, T, D = k_pop.shape
    assert T % chunk_size == 0, f"T={T} not a multiple of chunk_size={chunk_size}"
    assert D % num_heads == 0, f"d_state={D} not divisible by num_heads={num_heads}"
    assert token_ids_cat.shape == (B, T), (
        f"token_ids_cat shape {tuple(token_ids_cat.shape)} != k_pop's (B,T)=({B},{T})")
    assert content_mask.shape == (B, T), (
        f"content_mask shape {tuple(content_mask.shape)} != k_pop's (B,T)=({B},{T})")
    n_chunks = T // chunk_size
    head_dim = D // num_heads
    k_h = k_pop.view(B, n_chunks, chunk_size, num_heads, head_dim)
    mask_c = content_mask.view(B, n_chunks, chunk_size)
    tok_c = token_ids_cat.view(B, n_chunks, chunk_size)

    top20_list = sorted(top20_ids)
    top20_tensor = (torch.tensor(top20_list, dtype=token_ids_cat.dtype, device=token_ids_cat.device)
                     if top20_list else None)

    def _empty_stratum(reason: str) -> dict:
        return {"excluded": True, "exclusion_reason": reason, "same_tok_sim": None,
                 "diff_tok_sim": None, "repeat_excess": None, "n_same": 0, "n_diff": 0}

    records = []
    for b in range(B):
        for c in range(n_chunks):
            valid_idx = mask_c[b, c].nonzero(as_tuple=True)[0]
            n_valid = int(valid_idx.numel())
            toks = tok_c[b, c, valid_idx]                            # (n_valid,)

            for h in range(num_heads):
                rec = {"b": b, "chunk": c, "head": h, "n_valid": n_valid}

                if n_valid < min_valid:
                    rec.update({"excluded": True, "exclusion_reason": "below_min_valid",
                                 "same_tok_sim": None, "diff_tok_sim": None, "repeat_excess": None,
                                 "n_same": 0, "n_diff": 0,
                                 "stratum_top20": _empty_stratum("below_min_valid"),
                                 "stratum_rest": _empty_stratum("below_min_valid")})
                    records.append(rec)
                    continue

                keys = k_h[b, c, valid_idx, h, :]                     # (n_valid, head_dim)
                keys_n = F.normalize(keys, dim=-1, eps=1e-8)
                sims = keys_n @ keys_n.T                              # (n_valid, n_valid) cosine
                eye = torch.eye(n_valid, dtype=torch.bool, device=keys.device)
                same_mask = (toks.unsqueeze(0) == toks.unsqueeze(1)) & ~eye   # i!=j, tok(i)=tok(j)
                diff_mask = (toks.unsqueeze(0) != toks.unsqueeze(1))          # i!=j implied (tok(i)!=tok(j))
                n_same, n_diff = int(same_mask.sum().item()), int(diff_mask.sum().item())

                if n_same == 0 or n_diff == 0:
                    reason = "no_same_pair" if n_same == 0 else "no_diff_pair"
                    rec.update({"excluded": True, "exclusion_reason": reason,
                                 "same_tok_sim": None, "diff_tok_sim": None, "repeat_excess": None,
                                 "n_same": n_same, "n_diff": n_diff,
                                 "stratum_top20": _empty_stratum(reason),
                                 "stratum_rest": _empty_stratum(reason)})
                    records.append(rec)
                    continue

                same_tok_sim = sims[same_mask].mean().item()
                diff_tok_sim = sims[diff_mask].mean().item()
                rec.update({"excluded": False, "exclusion_reason": None,
                             "same_tok_sim": same_tok_sim, "diff_tok_sim": diff_tok_sim,
                             "repeat_excess": same_tok_sim - diff_tok_sim,
                             "n_same": n_same, "n_diff": n_diff})

                # H5: same pair population, split by whether tok(i) (the ordered pair's row/
                # "anchor" index) is in the top-20-most-frequent set vs. the rest.
                if top20_tensor is not None and top20_tensor.numel() > 0:
                    row_top20 = torch.isin(toks, top20_tensor)        # (n_valid,)
                    for stratum_name, row_mask in (("top20", row_top20), ("rest", ~row_top20)):
                        row_mask_2d = row_mask.unsqueeze(1).expand(n_valid, n_valid)
                        s_same = same_mask & row_mask_2d
                        s_diff = diff_mask & row_mask_2d
                        ns, nd = int(s_same.sum().item()), int(s_diff.sum().item())
                        if ns == 0 or nd == 0:
                            s_reason = "no_same_pair" if ns == 0 else "no_diff_pair"
                            rec[f"stratum_{stratum_name}"] = _empty_stratum(s_reason)
                            rec[f"stratum_{stratum_name}"]["n_same"], rec[f"stratum_{stratum_name}"]["n_diff"] = ns, nd
                        else:
                            sts, dts = sims[s_same].mean().item(), sims[s_diff].mean().item()
                            rec[f"stratum_{stratum_name}"] = {
                                "excluded": False, "exclusion_reason": None,
                                "same_tok_sim": sts, "diff_tok_sim": dts, "repeat_excess": sts - dts,
                                "n_same": ns, "n_diff": nd,
                            }
                else:
                    rec["stratum_top20"] = _empty_stratum("no_top20_set")
                    rec["stratum_rest"] = _empty_stratum("no_top20_set")

                records.append(rec)
    return records


def _pool_flat(flat: list[dict]) -> dict:
    """Pools a flat list of {excluded, same_tok_sim, diff_tok_sim, repeat_excess} dicts EXACTLY
    the way lm_attractor_probe_rd.summarize_gram_records pools gram_deviation: a plain
    (unweighted) `.mean()` over INCLUDED entries' own values -- see this file's module docstring
    for the "n-weighted" prose-vs-code discrepancy this replicates deliberately."""
    valid = [r for r in flat if not r["excluded"]]
    n_excluded = len(flat) - len(valid)
    if not valid:
        return {"n_episodes": len(flat), "n_excluded": n_excluded, "n_included": 0,
                 "repeat_excess_mean": None, "repeat_excess_std": None,
                 "same_tok_sim_mean": None, "diff_tok_sim_mean": None,
                 "note": "every episode excluded -- no qualifying same+diff pair anywhere in this population"}
    re_ = torch.tensor([r["repeat_excess"] for r in valid], dtype=torch.float64)
    ss = torch.tensor([r["same_tok_sim"] for r in valid], dtype=torch.float64)
    ds = torch.tensor([r["diff_tok_sim"] for r in valid], dtype=torch.float64)
    return {
        "n_episodes": len(flat), "n_excluded": n_excluded, "n_included": len(valid),
        "repeat_excess_mean": re_.mean().item(),
        "repeat_excess_std": re_.std(unbiased=False).item() if len(valid) > 1 else 0.0,
        "same_tok_sim_mean": ss.mean().item(), "diff_tok_sim_mean": ds.mean().item(),
    }


def summarize_repeat_excess(records: list[dict]) -> dict:
    """Overall pool + the two H5 strata pools, from the SAME records list token_identity_records
    produced (H5 "reuses the exact same captured pairs", sec 12.4)."""
    return {
        "overall": _pool_flat(records),
        "h5_top20_stratum": _pool_flat([r["stratum_top20"] for r in records]),
        "h5_rest_stratum": _pool_flat([r["stratum_rest"] for r in records]),
    }


def correlate_repeat_excess_vs_gram(re_records: list[dict], gram_records: list[dict]) -> dict:
    """Free byproduct (sec 12.4.1's own "free cross-check"): the capture population for
    repeat_excess (token_identity_records, this file) and gram_deviation
    (lm_attractor_probe_rd.chunk_key_gram_stats) is IDENTICAL by construction -- same k_raw
    tensor, same content_mask, same chunk_size, same (b,chunk,head) grid -- so a per-episode join
    by (b,chunk,head) is exact, not approximate. Restricted to episodes where BOTH statistics are
    defined (neither excluded)."""
    gram_by_key = {(r["b"], r["chunk"], r["head"]): r["gram_deviation"] for r in gram_records
                    if r["gram_deviation"] is not None}
    paired = [(r["repeat_excess"], gram_by_key[(r["b"], r["chunk"], r["head"])])
               for r in re_records if not r["excluded"] and (r["b"], r["chunk"], r["head"]) in gram_by_key]
    if len(paired) < 2:
        return {"n_paired": len(paired), "pearson_r": None,
                 "note": "fewer than 2 jointly-included episodes -- correlation undefined"}
    xs = torch.tensor([p[0] for p in paired], dtype=torch.float64)
    ys = torch.tensor([p[1] for p in paired], dtype=torch.float64)
    xs_c, ys_c = xs - xs.mean(), ys - ys.mean()
    denom = xs_c.norm().item() * ys_c.norm().item()
    r = (xs_c @ ys_c).item() / denom if denom > 0 else None
    return {"n_paired": len(paired), "pearson_r": r}


# ---------------------------------------------------------------------------
# Real-data driver.
# ---------------------------------------------------------------------------

def _resolve_ckpt_path(ckpt_base_dir: str, corpus: str, seed: int, arm: str, step: int) -> str:
    """IDENTICAL convention to mech_h4_paramdiff.py's own _resolve_ckpt_path (verified against
    frozen_bias_lm_sweep.py's cell_name()/lm_pretrain_rd.py's run_name construction)."""
    lam_tag = ARM_LAM_TAG[arm]
    cell = f"frozenbias_lm_{arm}_{lam_tag}_{corpus}_dm256_ds64_L2_s{seed}"
    run_name = f"lmC_{corpus}_dm256_ds64_L2_s{seed}"
    return os.path.join(ckpt_base_dir, cell, f"{run_name}_step{step}.pt")


def run_one_checkpoint_pass(ckpt_path: str, corpus: str, data_dir: str, chunk_size: int,
                             n_windows: int, batch_size: int, seq_len: int, device: str) -> dict:
    """One checkpoint's full capture + repeat_excess + gram/span_frac + H5 + correlation, on one
    corpus's held-out windows. Sampling discipline IDENTICAL to frozen_bias_retrofit_eval_rd.py
    (n_windows=32, batch_size=16, seq_len=512, generator seed corpus_fixed_seed(corpus)+95_000)
    so the captured population is the SAME one the existing kraw co-primary already measured --
    repeat_excess and gram_deviation are directly joint-interpretable per episode (sec 12.4's own
    "free byproduct" framing).

    kraw discipline: capture_raw_keys hooks k_conv1d's OUTPUT, strictly BEFORE
    apply_frozen_bias_blend runs inside that SAME forward pass, for ANY arm's checkpoint, by
    construction of the insertion point -- so no explicit blend call is made anywhere here."""
    from lm_pretrain_rd import EOT_TOKEN_ID, corpus_fixed_seed, get_batch, load_corpus   # deferred, see module docstring
    model, ckpt = load_checkpoint(ckpt_path, device)
    num_heads = model.blocks[0].mixer.num_heads

    _, val_tokens, meta, _, val_offs = load_corpus(data_dir, corpus, device)
    gen = torch.Generator(device=device).manual_seed(corpus_fixed_seed(corpus) + 95_000)
    n_batches = max(1, -(-n_windows // batch_size))
    batches = [get_batch(val_tokens, batch_size, seq_len, gen)[0] for _ in range(n_batches)]
    keys_by_layer, token_ids_cat = capture_raw_keys(model, batches, device)
    content_mask = (token_ids_cat != EOT_TOKEN_ID)
    top20_ids = compute_top_k_frequent(token_ids_cat, content_mask, TOP_K_FREQ)

    per_layer = {}
    re_records_all_layers = []
    for layer_idx, k_raw in keys_by_layer.items():
        re_records = token_identity_records(k_raw, token_ids_cat, content_mask, num_heads,
                                              chunk_size, top20_ids)
        gram_summary, gram_records = measure_population(k_raw, content_mask, num_heads, chunk_size)
        re_summary = summarize_repeat_excess(re_records)
        corr = correlate_repeat_excess_vs_gram(re_records, gram_records)
        per_layer[layer_idx] = {
            "repeat_excess": re_summary,
            "gram_deviation_span_frac": gram_summary,
            "repeat_excess_vs_gram_deviation_correlation": corr,
        }
        re_records_all_layers.extend({**r, "layer": layer_idx} for r in re_records)

    del model, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": ckpt_path, "corpus": corpus,
        "n_windows_sampled": int(token_ids_cat.shape[0]),
        "top20_token_ids": sorted(top20_ids),
        "per_layer": per_layer,
        "pooled_across_layers": summarize_repeat_excess(re_records_all_layers),
    }


def adjacent_sign_flip_check(deltas_by_step: dict) -> dict:
    """Sec 12.4.1 step 2's own MECHANICAL definition: a trajectory is "ambiguous" iff the sign of
    the delta flips between ANY two ADJACENT sampled steps (sorted ascending) -- not an eyeball
    call. `deltas_by_step`: {step: delta_or_None} (None when a checkpoint at that step is
    missing)."""
    steps_sorted = sorted(deltas_by_step.keys())
    if len(steps_sorted) < 2:
        return {"applicable": False,
                 "note": "fewer than 2 steps present in this invocation -- ambiguity check requires >=2"}
    adjacent_checks = []
    for i in range(len(steps_sorted) - 1):
        s0, s1 = steps_sorted[i], steps_sorted[i + 1]
        d0, d1 = deltas_by_step[s0], deltas_by_step[s1]
        if d0 is None or d1 is None:
            adjacent_checks.append({"steps": [s0, s1], "flip": None,
                                      "note": "missing delta (checkpoint absent) -- flip undefined here"})
            continue
        sign0 = 0 if d0 == 0 else math.copysign(1, d0)
        sign1 = 0 if d1 == 0 else math.copysign(1, d1)
        flip = bool(sign0 != 0 and sign1 != 0 and sign0 != sign1)
        adjacent_checks.append({"steps": [s0, s1], "flip": flip, "delta_early": d0, "delta_late": d1})
    ambiguous = any(c["flip"] for c in adjacent_checks if c["flip"] is not None)
    return {"applicable": True, "steps": steps_sorted, "adjacent_checks": adjacent_checks,
             "ambiguous": ambiguous}


def run_probe(args) -> dict:
    device = args.device
    steps = sorted(int(s) for s in args.steps.split(","))
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(ALL_ARMS)
    for a in arms:
        assert a in ALL_ARMS, f"unknown arm {a!r}, expected one of {ALL_ARMS}"

    per_arm_step = {arm: {} for arm in arms}
    missing = []

    for arm in arms:
        for step in steps:
            ckpt_path = _resolve_ckpt_path(args.ckpt_base_dir, args.corpus, args.seed, arm, step)
            if not os.path.exists(ckpt_path):
                missing.append(ckpt_path)
                per_arm_step[arm][step] = {"missing_checkpoint": ckpt_path}
                continue
            pass_result = run_one_checkpoint_pass(
                ckpt_path, args.corpus, args.data_dir, args.chunk_size, args.n_windows,
                args.batch_size, args.seq_len, device)
            per_arm_step[arm][step] = pass_result

    if missing:
        print(f"MISSING {len(missing)} checkpoint(s) -- their (arm, step) cells are reported as "
              f"'missing_checkpoint' in the output, not silently skipped:")
        for p in missing:
            print(f"  MISSING: {p}")

    # Pre-registered deltas: Arm2-vs-Arm1 (per_token - off) and Arm2'-vs-Arm1 (global - off), per
    # step, read off pooled_across_layers.overall.repeat_excess_mean (kraw pre-blend).
    def _re_mean(arm, step):
        cell = per_arm_step.get(arm, {}).get(step)
        if cell is None or "missing_checkpoint" in cell:
            return None
        return cell["pooled_across_layers"]["overall"]["repeat_excess_mean"]

    deltas_arm2 = {}
    deltas_arm2p = {}
    for step in steps:
        m1 = _re_mean("off", step)
        m2 = _re_mean("per_token", step) if "per_token" in arms else None
        m2p = _re_mean("global", step) if "global" in arms else None
        deltas_arm2[step] = (m2 - m1) if (m1 is not None and m2 is not None) else None
        deltas_arm2p[step] = (m2p - m1) if (m1 is not None and m2p is not None) else None

    ambiguity_arm2 = adjacent_sign_flip_check(deltas_arm2) if "per_token" in arms and "off" in arms else \
        {"applicable": False, "note": "arm2 (per_token) or arm1 (off) not requested this invocation"}
    ambiguity_arm2p = adjacent_sign_flip_check(deltas_arm2p) if "global" in arms and "off" in arms else \
        {"applicable": False, "note": "arm2' (global) or arm1 (off) not requested this invocation"}

    # Yield floor: pooled INCLUDED episode count across every (arm,step) pass actually captured
    # in THIS invocation (this corpus, this seed) -- summed directly from each cell's own
    # n_included/n_excluded/n_episodes (pooled_across_layers already pooled within a cell; this
    # is a further sum ACROSS cells) -- see module docstring for the multi-corpus scoping caveat.
    total_included = sum(
        per_arm_step[arm][step]["pooled_across_layers"]["overall"]["n_included"]
        for arm in arms for step in steps if "missing_checkpoint" not in per_arm_step[arm][step]
    )
    total_excluded = sum(
        per_arm_step[arm][step]["pooled_across_layers"]["overall"]["n_excluded"]
        for arm in arms for step in steps if "missing_checkpoint" not in per_arm_step[arm][step]
    )
    total_episodes = sum(
        per_arm_step[arm][step]["pooled_across_layers"]["overall"]["n_episodes"]
        for arm in arms for step in steps if "missing_checkpoint" not in per_arm_step[arm][step]
    )
    under_powered = total_included < YIELD_FLOOR
    _, _, analyze_path_used = _load_span_frac_fns()   # cached by now (every real pass already loaded it)

    output = {
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 12.4/12.4.1 (Stage 1 -- repeat_excess H1/H5)",
        "corpus": args.corpus, "seed": args.seed, "arms": arms, "steps": steps,
        "chunk_size": args.chunk_size, "n_windows": args.n_windows, "batch_size": args.batch_size,
        "seq_len": args.seq_len, "analyze_path_used": analyze_path_used,
        "kraw_discipline_note": (
            "every repeat_excess/gram_deviation/span_frac number in this file is computed on "
            "capture_raw_keys' own pre-blend k_raw -- no apply_frozen_bias_blend call is made "
            "anywhere in this script; see module docstring."),
        "per_arm_per_step": per_arm_step,
        "deltas": {
            "arm2_minus_arm1_per_step": deltas_arm2,
            "arm2prime_minus_arm1_per_step": deltas_arm2p,
        },
        "trajectory_ambiguity_check_sec_12_4_1_step2": {
            "arm2_vs_arm1": ambiguity_arm2,
            "arm2prime_vs_arm1": ambiguity_arm2p,
        },
        "yield_floor": {
            "registered_minimum": YIELD_FLOOR,
            "scope": "pooled across every (arm, step) checkpoint pass THIS invocation captured "
                      "(this corpus, this seed only) -- sum this field across both corpora's own "
                      "output JSONs to check the FULL sec 12.4 descriptive-pass floor.",
            "pooled_included_episodes_this_invocation": total_included,
            "pooled_excluded_episodes_this_invocation": total_excluded,
            "pooled_total_episodes_this_invocation": total_episodes,
            "under_powered_insufficient_yield": under_powered,
        },
        "missing_checkpoints": missing,
    }
    if under_powered:
        output["repeat_excess_trust_status"] = (
            "UNDER-POWERED / INSUFFICIENT YIELD -- pooled included-episode count "
            f"({total_included}) is below the registered floor ({YIELD_FLOOR}); repeat_excess "
            "numbers in this file must NOT be presented as a trusted reading, per sec 12.4's "
            "MINOR-4 fix.")
    else:
        output["repeat_excess_trust_status"] = (
            f"yield floor cleared ({total_included} >= {YIELD_FLOOR} included episodes) -- "
            "repeat_excess numbers here may be read as intended, subject to sec 12's own "
            "exploratory-tier/correlational-ceiling caveats (sec 12.9 items 1-2).")
    return output


# ---------------------------------------------------------------------------
# --self-test -- pure CPU, no checkpoints. Three parts:
#   1. Reproduce sec 12.3.4 self-test 1's TWO pinned constructions through
#      THIS SCRIPT's OWN token_identity_records() code path (sec 12.9 item 4's
#      formula-vs-plumbing gap) -- NOT mech_stage05_selftests.py's own flat
#      repeat_excess_stats() reference implementation, deliberately not
#      imported/reused here.
#   2. The exclusion rule: an all-identical-token episode and an
#      all-distinct-token episode must BOTH be excluded and COUNTED.
#   3. The yield floor: n=49 included episodes sets the flag; n=50 does not.
# ---------------------------------------------------------------------------

_ST1_RAW = [
    (1, 2, -1), (0.5, -1.5, 2), (-2, 0.5, 1), (1.5, 1.5, 1.5),
    (-1, -2, 0.5), (2, -0.5, -1.5), (0, 1, -2), (-1.5, 0.5, -1),
    (1, -1, 1), (-0.5, 2, 0.5), (2, 2, -0.5), (-1, 0, 2),
]


def self_test() -> bool:
    print("=" * 70)
    print("FROZEN_BIAS_TOKEN_IDENTITY_PROBE --self-test")
    print("(pushes sec 12.3.4's pinned constructions through THIS FILE's own")
    print(" token_identity_records() code path -- closes sec 12.9 item 4's")
    print(" formula-vs-plumbing gap for the repeat_excess statistic)")
    print("=" * 70)
    ok = True

    k_pop = torch.tensor([_ST1_RAW], dtype=torch.float64)          # (B=1, T=12, D=3)
    content_mask = torch.ones(1, 12, dtype=torch.bool)
    chunk_size, num_heads = 12, 1

    # --- Part 1a: uncorrelated-assignment construction (round-robin A,B,C,D x3) ---
    tok_a = torch.tensor([[i % 4 for i in range(12)]], dtype=torch.long)
    recs_a = token_identity_records(k_pop, tok_a, content_mask, num_heads, chunk_size, top20_ids=set())
    assert len(recs_a) == 1, f"expected exactly 1 (b,chunk,head) episode, got {len(recs_a)}"
    r_a = recs_a[0]
    print(f"\n(a) uncorrelated, THROUGH PROBE CODE PATH: repeat_excess={r_a['repeat_excess']:.6f} "
          f"same_tok_sim={r_a['same_tok_sim']:.6f} diff_tok_sim={r_a['diff_tok_sim']:.6f} "
          f"n_same={r_a['n_same']} n_diff={r_a['n_diff']}")
    a_ok = (
        not r_a["excluded"]
        and abs(r_a["repeat_excess"] - (-0.238404)) < 1e-4
        and abs(r_a["same_tok_sim"] - (-0.252186)) < 1e-4
        and abs(r_a["diff_tok_sim"] - (-0.013781)) < 1e-4
        and r_a["n_same"] == 24 and r_a["n_diff"] == 108
    )
    print(f"    expected repeat_excess=-0.238404 (+/-1e-4), n_same=24, n_diff=108  PASS={a_ok}")
    ok = ok and a_ok

    # --- Part 1b: planted-clustering construction ((0,10),(1,8),(2,11) merged into one id) ---
    tok_b = list(range(100, 112))     # unique ids per position (offset, avoids clashing with 0..3)
    shared_id = 999
    for p in (0, 10, 1, 8, 2, 11):
        tok_b[p] = shared_id
    tok_b_t = torch.tensor([tok_b], dtype=torch.long)
    recs_b = token_identity_records(k_pop, tok_b_t, content_mask, num_heads, chunk_size, top20_ids=set())
    r_b = recs_b[0]
    print(f"\n(b) planted, THROUGH PROBE CODE PATH: repeat_excess={r_b['repeat_excess']:.6f} "
          f"same_tok_sim={r_b['same_tok_sim']:.6f} diff_tok_sim={r_b['diff_tok_sim']:.6f} "
          f"n_same={r_b['n_same']} n_diff={r_b['n_diff']}")
    b_ok = (
        not r_b["excluded"]
        and abs(r_b["repeat_excess"] - 0.034437) < 1e-4
        and abs(r_b["same_tok_sim"] - (-0.030517)) < 1e-4
        and abs(r_b["diff_tok_sim"] - (-0.064954)) < 1e-4
        and r_b["n_same"] == 30 and r_b["n_diff"] == 102
    )
    print(f"    expected repeat_excess=+0.034437 (+/-1e-4), n_same=30, n_diff=102  PASS={b_ok}")
    ok = ok and b_ok

    delta = r_b["repeat_excess"] - r_a["repeat_excess"]
    delta_ok = delta >= 0.2
    print(f"\n    planted delta = {delta:.6f} (registered 0.272841), >= 0.2 required  PASS={delta_ok}")
    ok = ok and delta_ok

    # --- Part 2: exclusion rule -- all-identical and all-distinct episodes must BOTH be
    # excluded AND counted (never silently dropped/zeroed). ---
    print("\n" + "-" * 70)
    print("EXCLUSION-RULE self-test")
    print("-" * 70)
    tok_allsame = torch.zeros(1, 12, dtype=torch.long)               # every position same id
    recs_allsame = token_identity_records(k_pop, tok_allsame, content_mask, num_heads, chunk_size, top20_ids=set())
    e_same = recs_allsame[0]
    t_allsame = (e_same["excluded"] is True and e_same["exclusion_reason"] == "no_diff_pair"
                 and e_same["repeat_excess"] is None and e_same["n_same"] == 132 and e_same["n_diff"] == 0)
    print(f"all-identical-token episode: excluded={e_same['excluded']} reason={e_same['exclusion_reason']} "
          f"n_same={e_same['n_same']} n_diff={e_same['n_diff']}  PASS={t_allsame}")
    ok = ok and t_allsame

    tok_alldistinct = torch.arange(12, dtype=torch.long).unsqueeze(0)  # every position unique
    recs_alldistinct = token_identity_records(k_pop, tok_alldistinct, content_mask, num_heads, chunk_size, top20_ids=set())
    e_dist = recs_alldistinct[0]
    t_alldistinct = (e_dist["excluded"] is True and e_dist["exclusion_reason"] == "no_same_pair"
                      and e_dist["repeat_excess"] is None and e_dist["n_same"] == 0 and e_dist["n_diff"] == 132)
    print(f"all-distinct-token episode:  excluded={e_dist['excluded']} reason={e_dist['exclusion_reason']} "
          f"n_same={e_dist['n_same']} n_diff={e_dist['n_diff']}  PASS={t_alldistinct}")
    ok = ok and t_alldistinct

    combined = recs_allsame + recs_alldistinct
    summary = summarize_repeat_excess(combined)
    ov = summary["overall"]
    t_counted = ov["n_episodes"] == 2 and ov["n_excluded"] == 2 and ov["n_included"] == 0
    print(f"both exclusions COUNTED (not silently dropped): n_episodes={ov['n_episodes']} "
          f"n_excluded={ov['n_excluded']} n_included={ov['n_included']}  PASS={t_counted}")
    ok = ok and t_counted

    # --- Part 3: yield floor -- n=49 included episodes sets the flag, n=50 does not. ---
    print("\n" + "-" * 70)
    print("YIELD-FLOOR self-test")
    print("-" * 70)

    def _fake_included_episode() -> dict:
        empty = {"excluded": True, "exclusion_reason": "no_top20_set", "same_tok_sim": None,
                  "diff_tok_sim": None, "repeat_excess": None, "n_same": 0, "n_diff": 0}
        return {"excluded": False, "exclusion_reason": None, "same_tok_sim": 0.10,
                 "diff_tok_sim": 0.05, "repeat_excess": 0.05, "n_same": 2, "n_diff": 2,
                 "stratum_top20": dict(empty), "stratum_rest": dict(empty)}

    recs_49 = [_fake_included_episode() for _ in range(49)]
    recs_50 = [_fake_included_episode() for _ in range(50)]
    n_incl_49 = summarize_repeat_excess(recs_49)["overall"]["n_included"]
    n_incl_50 = summarize_repeat_excess(recs_50)["overall"]["n_included"]
    flag_49 = n_incl_49 < YIELD_FLOOR
    flag_50 = n_incl_50 < YIELD_FLOOR
    t_yield = (n_incl_49 == 49 and flag_49 is True and n_incl_50 == 50 and flag_50 is False)
    print(f"n=49 included -> under_powered={flag_49} (expect True); "
          f"n=50 included -> under_powered={flag_50} (expect False)  PASS={t_yield}")
    ok = ok and t_yield

    print("\n" + "=" * 70)
    print("FROZEN_BIAS_TOKEN_IDENTITY_PROBE --self-test: " + ("ALL PASSED" if ok else "FAILURES PRESENT"))
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = wrap_exploratory({
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 12.4 (Stage 1 repeat_excess) -- "
                       "PLUMBING self-test (sec 12.9 item 4), NOT sec 12.3.4's formula-level gate",
        "self_test_passed": ok,
        "construction_a_uncorrelated": r_a,
        "construction_b_planted": r_b,
        "planted_delta": delta,
        "exclusion_rule": {"all_same": e_same, "all_distinct": e_dist, "counted_check": ov},
        "yield_floor_check": {"n_included_49": n_incl_49, "under_powered_49": flag_49,
                                "n_included_50": n_incl_50, "under_powered_50": flag_50},
    })
    out_path = os.path.join(RESULTS_DIR, "frozen_bias_token_identity_probe_selftest_results.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_path}")

    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ckpt-base-dir", type=str, default=None,
                     help="e.g. /data/deltanet_rd_frozenbias_ckpts (box path root)")
    ap.add_argument("--corpus", type=str, default="openr1-mix-ext",
                     help="ONE corpus per invocation -- run twice (once per corpus) for the "
                          "sec 12.4 descriptive pass's own 'both corpora' requirement.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=str, default="20000",
                     help="comma list, e.g. '20000' (descriptive) or "
                          "'1000,5000,10000,15000,20000' (sec 12.4.1 trajectory sub-study)")
    ap.add_argument("--arms", type=str, default=",".join(ALL_ARMS),
                     help="comma list from {off, per_token, global}")
    ap.add_argument("--data-dir", default=_DEFAULT_DATA_DIR_FALLBACK)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default=None, help="default: cuda if available else cpu")
    args = ap.parse_args()

    if args.self_test:
        ok = self_test()
        return 0 if ok else 1

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert args.ckpt_base_dir, "--ckpt-base-dir is required for a real (non-self-test) run"

    result = run_probe(args)
    payload = wrap_exploratory(result)
    out_path = args.out or os.path.join(
        RESULTS_DIR, f"frozen_bias_token_identity_probe_{args.corpus}_s{args.seed}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_arm_per_step"}, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
