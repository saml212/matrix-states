"""keyanchor_drift_diagnostic.py -- KEY_ANCHORING_DESIGN.md sec 4's
near-verbatim clone of geo3_drift_diagnostic.py, for candidate (d) (the
anchor-first blend). Differences from the bare-geo3 diagnostic, ALL
additive:

  1. Builds the model with anchor_active=True (learned lambda by default,
     --anchor-lambda-fixed to probe a fixed-grid value instead).
  2. measure_drift's pre_ns_attr="anchor_last_k_blend_raw" (candidate (d)'s
     own pre-NS side channel, sec 2.2) instead of bare geo3's
     "geo3_last_k_eff_raw" -- sec 3.1 item 5's evidentiary object for this
     candidate is the BLENDED pre-NS key, not the raw one.
  3. Gate 1's launch-read still reads the POST-NS measured drift (sec 4's
     scope carve-out, unchanged from bare geo3) -- but now goes through the
     PER-K-THREADED launch_read (geo3_simulator.py's fixed API), inheriting
     the fix by construction (this module never had the bug -- it did not
     exist before this wave -- but is built against the corrected API from
     day one, per the design's own "inherits the fixed API by construction"
     phrasing).
  4. ALSO runs the full-pool (all 107 train entities) per-entity
     anchor-INPUT-alignment sweep (sec 3.7) + the h=1 behavioral companion
     (bookkeeping on the SAME probe-train's own eval scores -- no new
     forward passes beyond what the probe train already runs) at the
     final probe checkpoint.

Requires CUDA (chunk_delta_rule has no CPU path, same constraint as
geo3_drift_diagnostic.py) -- a GPU driver script, NOT runnable under this
wave's CPU-only build constraint. Built here so it is ready to run the
moment a GPU frees up; smoke_key_anchoring.py covers everything CPU-testable
about the underlying mechanism in the meantime.

2026-07-06 keyanchor-confirm build -- TWO fixes to this script, both
diagnosed from its own documented crash (KEY_ANCHORING_DESIGN.md sec 9.3;
the exact traceback is archived at
experiment-runs/2026-07-05_keyanchor_wave/logs/04_drift_diag.log):

1. THE CRASH ITSELF (log_every, this file's own bug): `run_one_k` used to
   call `train_geo3(..., log_every=max(1, probe_steps // 5), ...)` --
   1000 at the default probe_steps=5000 -- which fails
   run_deltanet_rd.py::train()'s own `ka.assert_lambda_log_cadence(log_every)`
   startup assertion (key_anchoring.py L279-292, registered
   LAMBDA_LOG_CADENCE_STEPS=200) the FIRST time this script ever ran.
   Fixed: log_every is now PROBE_LOG_EVERY (== ka.LAMBDA_LOG_CADENCE_STEPS
   unconditionally), never a probe_steps-derived fraction. See
   smoke_keyanchor_confirm.py for the regression check (proves the fixed
   value passes the assertion the retired formula reproducibly fails).
2. THE SWALLOWED-FAILURE BUG (NOT in this file -- diagnosed and recorded
   here for the reader's benefit): the crash above should have stopped
   `keyanchor_chain.sh`'s `&&`-gated chain ("any failure stops the chain
   and surfaces", per that script's own header) -- it did not, because
   EVERY stage in that script pipes through `tee`
   (`cmd 2>&1 | tee logs/X.log && next`) WITHOUT `set -o pipefail`. Bash's
   default pipeline exit status is the LAST command's (`tee`, which
   succeeds as long as it can write the log file) -- so this script's own
   real, non-zero exit code was silently discarded before `&&` ever saw
   it, and `waveref`/`wavekeyanchor` both ran anyway (by hand, per sec
   9.3's account). The historical archived chain
   (experiment-runs/2026-07-05_keyanchor_wave/scripts/keyanchor_chain.sh)
   is left untouched as a faithful record of what actually ran; this
   wave's own forward-looking chain,
   keyanchor_confirm_chain.sh, is written WITH `set -euo pipefail` from
   the start. Defensively (belt-and-suspenders, since a bare uncaught
   Python exception already exits 1 today), `main()` below is now a thin
   wrapper around `_run()` that guarantees ANY exception --> `sys.exit(1)`,
   never a silent/implicit success path, with its own regression test in
   smoke_key_anchoring.py (fla-required, runs on box) and a static-source
   companion in smoke_keyanchor_confirm.py (fla-free, runs anywhere).

Usage (single GPU per process, matching this harness's own convention):
  python keyanchor_drift_diagnostic.py --k 16 32 \\
      --out results/deltanet_rd_exactness/keyanchor_drift/wave_neg1_drift.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
import geo3_simulator as g3sim
import grammar_rd as grd
from model_rd import DeltaNetRDBlock
from run_deltanet_rd import train as train_geo3
from key_anchoring import (LAMBDA_LOG_CADENCE_STEPS, measure_drift, measure_full_pool_alignment,
                             measure_h1_behavioral_companion, sample_batch_fixed_entity)  # noqa: F401 (re-export parity)

# sec 9.3's fix (item 1 above): the probe-train's log_every MUST equal the
# registered cadence whenever anchor_lambda_mode=='learned' -- never a
# probe_steps-derived fraction (the retired `max(1, probe_steps // 5)`
# formula, == 1000 at the default below, crashed on first invocation).
DEFAULT_PROBE_STEPS = 5000
PROBE_LOG_EVERY = LAMBDA_LOG_CADENCE_STEPS


def run_one_k(K, tokenizer_pools, device, seed, n_entities, n_resamples, probe_steps, probe_batch_size,
              anchor_lambda_mode="learned", anchor_lambda_fixed=None):
    pools, pool_report = tokenizer_pools
    cfg = grd.DeltaNetRDTaskConfig(K=K, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    assert pool_report["n_train_names"] >= max(n_entities, K), \
        f"K={K}: train pool ({pool_report['n_train_names']}) too small for n_entities={n_entities} or K={K}"

    torch.manual_seed(seed)
    model = DeltaNetRDBlock(pools.vocab_size_total, d_model=256, d_state=64, conv_size=cfg.conv_size,
                             buffer_id=pools.buffer_id, geo3_active=True, geo3_n_iter=12,
                             geo3_resid_tol=1e-2, anchor_active=True,
                             anchor_lambda_mode=anchor_lambda_mode, anchor_lambda_fixed=anchor_lambda_fixed,
                             anchor_train_ids=pools.train_name_ids).to(device)

    gen_init = torch.Generator(device=device).manual_seed(seed + 1)
    t0 = time.time()
    at_init = measure_drift(model, cfg, pools, n_entities, n_resamples, gen_init, device,
                              pre_ns_attr="anchor_last_k_blend_raw")
    print(f"  K={K} AT INIT: post_ns mean={at_init['post_ns']['mean']:.4f} "
          f"p10={at_init['post_ns']['p10']:.4f} ({at_init['n_pooled_pairs']} pooled pairs, "
          f"{time.time() - t0:.1f}s)", flush=True)

    print(f"  K={K}: probe-training {probe_steps} steps (batch_size={probe_batch_size}, "
          f"anchor_lambda_mode={anchor_lambda_mode})...", flush=True)
    t1 = time.time()
    probe_result = train_geo3(model, cfg, pools, pool_report, device, d_model=256, d_state=64,
                               steps=probe_steps, batch_size=probe_batch_size, lr=3e-4, seed=seed,
                               log_every=PROBE_LOG_EVERY, ckpt_every=probe_steps + 1,
                               exactness_config={"geo3_active": True, "geo3_n_iter": 12,
                                                  "geo3_resid_tol": 1e-2, "anchor_active": True,
                                                  "anchor_lambda_mode": anchor_lambda_mode,
                                                  "anchor_lambda_fixed": anchor_lambda_fixed})
    print(f"  K={K}: probe train done in {time.time() - t1:.1f}s, "
          f"final loss={probe_result['trajectory'][-1]['loss']:.4f}, "
          f"skip_rate={probe_result['skip_rate']:.4%}", flush=True)

    gen_trained = torch.Generator(device=device).manual_seed(seed + 2)
    t2 = time.time()
    after_probe = measure_drift(model, cfg, pools, n_entities, n_resamples, gen_trained, device,
                                  pre_ns_attr="anchor_last_k_blend_raw")
    print(f"  K={K} AFTER {probe_steps}-STEP PROBE: post_ns mean={after_probe['post_ns']['mean']:.4f} "
          f"p10={after_probe['post_ns']['p10']:.4f} pre_ns mean={after_probe.get('pre_ns', {}).get('mean')} "
          f"({after_probe['n_pooled_pairs']} pooled pairs, {time.time() - t2:.1f}s)", flush=True)

    gen_align = torch.Generator(device=device).manual_seed(seed + 3)
    t3 = time.time()
    alignment = measure_full_pool_alignment(model, cfg, pools, n_resamples, gen_align, device)
    print(f"  K={K} PER-ENTITY ALIGNMENT: engaged_frac={alignment['engaged_frac']:.4f} "
          f"over {len(alignment['a_e'])} train entities ({time.time() - t3:.1f}s)", flush=True)

    gen_h1 = torch.Generator(device=device).manual_seed(seed + 4)
    t4 = time.time()
    h1_companion = measure_h1_behavioral_companion(model, cfg, pools, gen_h1, device)
    print(f"  K={K} H=1 BEHAVIORAL COMPANION: {len(h1_companion)} entities covered "
          f"({time.time() - t4:.1f}s)", flush=True)

    lam_final = float(model.anchor_lambda().item())

    return {
        "K": K, "at_init": at_init, "after_probe": after_probe,
        "probe_steps": probe_steps, "probe_batch_size": probe_batch_size,
        "probe_final_loss": probe_result["trajectory"][-1]["loss"],
        "probe_skip_rate": probe_result["skip_rate"],
        "anchor_lambda_final": lam_final,
        "anchor_lambda_mode": anchor_lambda_mode,
        "per_entity_alignment": alignment,
        "per_entity_h1_companion": h1_companion,
    }


def _run():
    """The diagnostic's real body (moved out of `main()`, 2026-07-06): every
    failure path here is either an `assert` (AssertionError, uncaught -->
    non-zero exit already) or propagates an underlying exception -- `main()`
    below wraps THIS function in a catch-all so NO exception, of any kind,
    from any line in this script, can ever result in a zero exit code. See
    this module's docstring, item 2, for why that guarantee matters here
    even though a bare uncaught exception already exits non-zero today."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--n-entities", type=int, default=8)
    ap.add_argument("--n-resamples", type=int, default=32)
    ap.add_argument("--probe-steps", type=int, default=DEFAULT_PROBE_STEPS)
    ap.add_argument("--probe-batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--heldout-frac", type=float, default=0.5)
    ap.add_argument("--anchor-lambda-mode", choices=["learned", "fixed"], default="learned")
    ap.add_argument("--anchor-lambda-fixed", type=float, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "F-geo-3/anchor's bind() requires CUDA (chunk_delta_rule has no CPU path)"
    assert args.n_entities >= 8 and args.n_resamples >= 32, \
        "sec 14.6 pins n_entities>=8 and n_resamples>=32 -- only increase them, never below the minimum"

    tokenizer = grd.load_gpt2_tokenizer()
    pools, pool_report = grd.build_entity_pools(tokenizer, heldout_frac=args.heldout_frac, seed=0)
    pools = pools.to(device)

    per_k = {}
    for K in args.k:
        print(f"=== K={K} (candidate d, anchor_lambda_mode={args.anchor_lambda_mode}) ===", flush=True)
        per_k[K] = run_one_k(K, (pools, pool_report), device, args.seed, args.n_entities,
                              args.n_resamples, args.probe_steps, args.probe_batch_size,
                              anchor_lambda_mode=args.anchor_lambda_mode,
                              anchor_lambda_fixed=args.anchor_lambda_fixed)

    # Gate 1's scope carve-out (sec 4, unchanged from bare geo3): the
    # launch-read's input STAYS the post-NS measured drift. Per-K threaded
    # by construction (this module never had the shared-c bug).
    assert 16 in per_k, "the sec 14.6 GATE requires K=16 to have been measured -- pass --k with 16 included"
    drift_by_k = {K: {"mean": per_k[K]["after_probe"]["post_ns"]["mean"],
                       "p10": per_k[K]["after_probe"]["post_ns"]["p10"]} for K in per_k}
    launch = g3sim.launch_read(drift_by_k, gram_resid=1e-2, seed=args.seed, device=device)

    out = {
        "design_ref": "KEY_ANCHORING_DESIGN.md sec 4 -- candidate (d) Wave -1 gating diagnostic",
        "sampling_spec": {"n_entities": args.n_entities, "n_resamples": args.n_resamples,
                            "probe_steps": args.probe_steps, "seed": args.seed,
                            "anchor_lambda_mode": args.anchor_lambda_mode,
                            "anchor_lambda_fixed": args.anchor_lambda_fixed},
        "per_k": per_k,
        "launch_read": launch,
        "launch": launch["launch"],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 70)
    print(f"LAUNCH READ (gate: K=16 h=4 rec@0.9 >= 0.8 under MEAN drift mapping):")
    print(f"  predicted K=16 h=4 rec@0.9 (mean mapping) = {launch['predicted_gate_value']:.4f}")
    print(f"  LAUNCH = {launch['launch']}")
    for K in per_k:
        if K != 16:
            print(f"  [non-gating] K={K}: predicted h=4 rec@0.9 (mean mapping) = "
                  f"{launch['mean'][K]['rec'][4]:.4f}; engaged_frac="
                  f"{per_k[K]['per_entity_alignment']['engaged_frac']:.4f}; "
                  f"lambda_final={per_k[K]['anchor_lambda_final']:.4f}")
    print(f"wrote {args.out}")
    print("=" * 70)


def main():
    """Thin, defensive wrapper (sec 9.3's swallowed-failure-bug fix, item 2
    of this module's docstring): ANY exception out of `_run()` --
    AssertionError, a CUDA OOM, a malformed-JSON write, anything -- is
    caught, its traceback printed, and turned into an explicit
    `sys.exit(1)`. `SystemExit` itself (e.g. argparse's own --help exit, or
    a future explicit sys.exit call inside `_run()`) is re-raised
    unchanged, never reinterpreted. See smoke_key_anchoring.py's own
    regression test (monkeypatches `_run` to raise, asserts SystemExit(1))
    and smoke_keyanchor_confirm.py's fla-free static-source companion."""
    try:
        _run()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
