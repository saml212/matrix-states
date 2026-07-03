"""lm_rd_rung_configs.py -- SCALE_TRANSFER_DESIGN.md sec 5.3's scaling-ladder
rung configs, as a single documented source of truth (Track C build brief
item (1): "thin config layer"). lm_pretrain_rd.py itself stays
architecture-generic (sec 5.2's n_layers widening is the ONLY rung-specific
change made to that file) -- this module only PARAMETRIZES it.

Sec 5.3's own caveat, repeated here verbatim because it governs how this
file must be used: "This table is a planning approximation, not a build
spec ... Every rung's config must be verified against the harness's own
printed/asserted parameter count ... before any GPU time is spent on it --
this is a Wave -1 (sec 5.5) blocking item, not assumed correct from this
table." verify_param_count() below IS that verification (also duplicated,
independently, as lm_pretrain_rd.py's own smoke() item [11] for rung 1 --
the harness's smoke gate must not depend on this file importing cleanly).

Verified on-box 2026-07-04 (rung 1 only -- rungs 2/3 are registered here for
completeness, sec 5.2's harness change benefits the whole ladder, but are
NOT part of this session's build/launch scope, which is rung 1 only per the
task brief and Track C Wave 4's gate-out):
  rung 1 (d_model=768, n_layers=12, d_state=64): measured 97,618,176 params
  vs target 98,000,000 -- 0.4% off, well inside the 15% Wave -1 gate.
"""
from __future__ import annotations

import argparse
import json

# Sec 5.3's table verbatim. d_state=64 for rung 1 (this session's only
# launch-authorized rung); rungs 2/3 use d_state=128 per sec 5.3 and carry
# the sec 4.2/5.5-item-3 "window stays 64, K_sel<=min(window,d_state)=64"
# rule if a future geo3-at-scale wave is ever un-gated -- moot for rung 1
# (d_state already 64) and entirely moot this session (Track C Wave 4 is
# GATED OUT: Track B's Wave -1 gate returned HARD NO-LAUNCH, sec 4/11).
RUNGS = {
    1: {"d_model": 768, "n_layers": 12, "d_state": 64, "approx_params": 98_000_000, "target": "~100M"},
    2: {"d_model": 1536, "n_layers": 16, "d_state": 128, "approx_params": 392_000_000, "target": "~400M"},
    3: {"d_model": 2560, "n_layers": 22, "d_state": 128, "approx_params": 1_310_000_000, "target": "~1.3B"},
}

# sec 5.6 Wave -1 gate criterion: "measured param counts within 15% of
# target (else adjust config)".
PARAM_COUNT_TOLERANCE = 0.15

VOCAB_SIZE = 50257  # GPT-2, shared tokenizer across every corpus (load_corpus's own assert)

# This session's build/launch scope (task brief): rung 1 ONLY. Rungs 2/3
# stay registered above (so a future session doesn't have to re-derive the
# table) but any launcher/calibration tool in this build refuses them
# explicitly rather than silently proceeding -- see run_lm_rd_trackc_sweep.py.
BUILD_SCOPE_RUNGS = (1,)


def formula_param_estimate(d_model: int, n_layers: int, d_state: int, vocab_size: int = VOCAB_SIZE) -> int:
    """Sec 5.3's own per-layer cost decomposition: FFN (8*d_model^2, the 4x-
    expansion 2-matrix MLP) + q/k/v/o_proj (4*d_model*d_state -- q/k/v are
    d_model->d_state, o_proj is d_state->d_model; MINOR-3's corrected
    labeling, b_proj's d_model*num_heads term and RMSNorm/conv1d params are
    each <0.1% of a rung-1-scale model and OMITTED here deliberately -- this
    is the PLANNING formula, not the verification. verify_param_count()
    below counts the REAL model exactly, including every omitted term."""
    per_layer = 8 * d_model ** 2 + 4 * d_model * d_state
    return vocab_size * d_model + n_layers * per_layer


def verify_param_count(rung: int, tolerance: float = PARAM_COUNT_TOLERANCE) -> dict:
    """Instantiates the REAL DeltaNetLM at rung `rung`'s config (CPU-only --
    no fla kernel call, just module construction and a numel() sum, so this
    needs no forward pass and no actual GPU compute; it DOES need fla/torch
    importable, which on this codebase only happens on-box) and counts
    sum(p.numel() for p in model.parameters()) -- sec 5.3/5.6's Wave -1
    blocking check, done for real, not assumed from RUNGS' approx_params."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lm_pretrain_rd import DeltaNetLM  # noqa: local import, pod-safe (see module docstring)

    cfg = RUNGS[rung]
    m = DeltaNetLM(VOCAB_SIZE, d_model=cfg["d_model"], d_state=cfg["d_state"],
                    n_layers=cfg["n_layers"], conv_size=4, num_heads=1, ffn_mult=4)
    n_real = sum(p.numel() for p in m.parameters())
    target = cfg["approx_params"]
    rel_err = abs(n_real - target) / target
    ok = rel_err <= tolerance
    return {
        "rung": rung, "config": cfg, "n_params_measured": n_real,
        "n_params_target": target, "relative_error": rel_err,
        "within_tolerance": ok, "tolerance": tolerance,
        "formula_estimate": formula_param_estimate(cfg["d_model"], cfg["n_layers"], cfg["d_state"]),
    }


def smoke():
    """Config-layer self-check: verify_param_count() for every BUILD_SCOPE_RUNGS entry
    (rung 1 only, this session) -- genuinely lightweight (no forward pass, no GPU
    compute, just module construction + numel() sums), not a training launch."""
    print("=" * 60 + "\n  LM_RD_RUNG_CONFIGS SMOKE GATE\n" + "=" * 60)
    for rung in BUILD_SCOPE_RUNGS:
        result = verify_param_count(rung)
        assert result["within_tolerance"], (
            f"rung {rung}: measured {result['n_params_measured']:,} params is "
            f"{result['relative_error'] * 100:.1f}% off target {result['n_params_target']:,} "
            f"(tolerance {result['tolerance'] * 100:.0f}%) -- sec 5.3's config table needs "
            f"adjustment before any GPU time is spent on this rung.")
        print(f"  rung {rung} PASSED: {result['n_params_measured']:,} params "
              f"(target {result['n_params_target']:,}, {result['relative_error'] * 100:.1f}% off, "
              f"within {result['tolerance'] * 100:.0f}% tolerance)")
    print("\n" + "=" * 60 + "\n  ALL LM_RD_RUNG_CONFIGS SMOKE CHECKS PASSED\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", type=int, choices=sorted(RUNGS), default=1)
    ap.add_argument("--smoke", action="store_true",
                     help="self-check every BUILD_SCOPE_RUNGS entry (rung 1) instead of a single "
                          "--rung's verification; this codebase's standard smoke-gate convention.")
    args = ap.parse_args()
    if args.smoke:
        smoke()
        return
    result = verify_param_count(args.rung)
    print(json.dumps(result, indent=2))
    assert result["within_tolerance"], (
        f"rung {args.rung}: measured {result['n_params_measured']:,} params is "
        f"{result['relative_error'] * 100:.1f}% off target {result['n_params_target']:,} "
        f"(tolerance {result['tolerance'] * 100:.0f}%) -- sec 5.3's config table needs adjustment "
        f"before any GPU time is spent on this rung.")
    print(f"\nrung {args.rung} PASSED: {result['n_params_measured']:,} params "
          f"(target {result['n_params_target']:,}, {result['relative_error'] * 100:.1f}% off, "
          f"within {result['tolerance'] * 100:.0f}% tolerance)")


if __name__ == "__main__":
    main()
