"""mech_h4_paramdiff.py -- FROZEN_BIAS_LM_DESIGN.md sec 12.3.3's H4 tool
(compensatory-constant test): checkpoint parameter-diff norm/cosine between a
matched seed/corpus triple's Arm 1 (off, control), Arm 2 (per_token), and
Arm 2' (global) checkpoints.

*** BOX-RUNNABLE, NOT YET EXECUTED AGAINST REAL CHECKPOINTS ***
This script needs the REAL checkpoint `.pt` files, which live ONLY on the Brev
box at `/data/deltanet_rd_frozenbias_ckpts/` (sec 12.0's own pre-check,
verified present there: 400 files = 20 training cells x 20 checkpoints each,
every 1,000 steps). This BUILD pass is Stage 0, explicitly ZERO-GPU/CPU-only,
and has no SSH execution in its own scope -- so per this task's own
instruction, this script is REGISTERED as the run step, not executed against
real data, in this pass. Its `--self-test` mode (bottom of this file) DOES run
to completion, locally, against synthetic fake checkpoints sharing the exact
real key-naming convention -- this verifies the diff/norm/cosine arithmetic
end-to-end (including real `torch.save`/`torch.load` I/O) before a future
session points `--ckpt-base-dir`/`--ckpt-dir` at real data.

--------------------------------------------------------------------------
ORCHESTRATION NOTE (the "small local orchestrating note" this build task asked
for, kept inline rather than as a separate doc file):
--------------------------------------------------------------------------
Checkpoint layout (verified against frozen_bias_lm_sweep.py's own
`cell_name()`/`ckpt_dir` convention and lm_pretrain_rd.py's own default
`run_name` construction, both read directly this session):

    <ckpt_base_dir>/<cell_name>/<run_name>_step<n>.pt
    cell_name = f"frozenbias_lm_{arm}_lam{lam_tag}_{corpus}_dm256_ds64_L2_s{seed}"
    run_name  = f"lmC_{corpus}_dm256_ds64_L2_s{seed}"          (NOTE: run_name
                does NOT itself encode arm/lambda -- cell_name's own directory
                is what disambiguates arms; this is a real, already-existing
                repo convention, not a guess made up for this script)

sec 12.3.3 recommends seed 0, openr1-mix-ext, steps {1000, 20000}. The 6 files
this implies (arm in {off, per_token, global} x step in {1000, 20000}):

    frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0/
        lmC_openr1-mix-ext_dm256_ds64_L2_s0_step1000.pt
        lmC_openr1-mix-ext_dm256_ds64_L2_s0_step20000.pt
    frozenbias_lm_per_token_lam0p58_openr1-mix-ext_dm256_ds64_L2_s0/
        lmC_openr1-mix-ext_dm256_ds64_L2_s0_step1000.pt
        lmC_openr1-mix-ext_dm256_ds64_L2_s0_step20000.pt
    frozenbias_lm_global_lam0p58_openr1-mix-ext_dm256_ds64_L2_s0/
        lmC_openr1-mix-ext_dm256_ds64_L2_s0_step1000.pt
        lmC_openr1-mix-ext_dm256_ds64_L2_s0_step20000.pt

Option A -- run directly ON the box (no data transfer needed):
    ssh youthful-indigo-turkey
    cd /home/nvidia/chapter2/deltanet_rd   # copy this file there first
    DRY_RUN_BYPASS=1 python3 mech_h4_paramdiff.py \\
        --ckpt-base-dir /data/deltanet_rd_frozenbias_ckpts \\
        --corpus openr1-mix-ext --seed 0 --step-early 1000 --step-late 20000 \\
        --out results/frozen_bias_lm/mech_h4_paramdiff_results.json

Option B -- scp the 6 files down and run locally (this Mac or elsewhere):
    BASE=/data/deltanet_rd_frozenbias_ckpts
    RUN=lmC_openr1-mix-ext_dm256_ds64_L2_s0
    mkdir -p ./h4_ckpts
    for cell in frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0 \\
                frozenbias_lm_per_token_lam0p58_openr1-mix-ext_dm256_ds64_L2_s0 \\
                frozenbias_lm_global_lam0p58_openr1-mix-ext_dm256_ds64_L2_s0; do
      for step in 1000 20000; do
        scp youthful-indigo-turkey:$BASE/$cell/${RUN}_step${step}.pt ./h4_ckpts/${cell}__step${step}.pt
      done
    done
    DRY_RUN_BYPASS=1 python3 mech_h4_paramdiff.py --flat-dir ./h4_ckpts \\
        --corpus openr1-mix-ext --seed 0 --step-early 1000 --step-late 20000

This script deliberately does NOT import `fla`, `lm_pretrain_rd`, or
`model_rd` -- it operates purely on the flat `state_dict` tensors inside each
checkpoint's own `["model_state_dict"]`, keyed by the exact dotted names
`nn.Module.state_dict()` produces (`blocks.<i>.mixer.k_conv1d.weight`,
`blocks.<i>.mixer.frozen_bias_global_vec` for the global arm only -- both
verified directly against lm_pretrain_rd.py's DeltaNetLMBlock/DeltaNetLMMixer/
DeltaNetLM source this session, sec 2/sec 5). `torch.load(...,
map_location="cpu")` only -- no CUDA context ever created, matching sec
12.3.3's own "no GPU needed" framing. `n_layers=2` (RUNG1_CFG) is the only
architecture fact this script hard-codes; everything else is read from each
checkpoint's own saved `config` dict.

Persisted through mech_schema.wrap_exploratory when writing real results
(sec 12.3.1's schema requirement, binding on every sec 12 artifact).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mech_schema import wrap_exploratory  # noqa: E402  (headline_verdict NOT imported anywhere in sec 12)

N_LAYERS = 2  # RUNG1_CFG["n_layers"] -- the only architecture fact hard-coded here
FROZEN_BUFFER_SUFFIXES = ("frozen_bias_table", "frozen_bias_global_vec")


# ---------------------------------------------------------------------------
# Checkpoint I/O -- pure torch.load, no model instantiation, no `fla` import.
# ---------------------------------------------------------------------------

def load_ckpt(path: str):
    """CPU-only load. Returns (state_dict, config_dict_or_None, step_or_None)."""
    import torch
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model_state_dict"], ckpt.get("config"), ckpt.get("step")


def select_keys(state_dict: dict, mode: str, n_layers: int = N_LAYERS) -> list:
    """mode: 'k_conv1d' (sec 12.3.3 PRIMARY -- blocks[i].mixer.k_conv1d's own
    weight/bias, every layer) or 'block0_full' (sec 12.3.3 SECONDARY, per
    attack Q2 -- the full block-0 parameter set, EXCLUDING the two known-
    frozen buffers, which never change by construction and would just dilute
    the trained-parameter-drift signal with a guaranteed-zero block)."""
    if mode == "k_conv1d":
        keys = []
        for i in range(n_layers):
            for suffix in ("weight", "bias"):
                k = f"blocks.{i}.mixer.k_conv1d.{suffix}"
                if k in state_dict:
                    keys.append(k)
        assert keys, "no k_conv1d keys found -- state_dict key convention may have changed"
        return keys
    elif mode == "block0_full":
        keys = sorted(
            k for k in state_dict
            if k.startswith("blocks.0.") and not any(k.endswith(s) for s in FROZEN_BUFFER_SUFFIXES)
        )
        assert keys, "no blocks.0.* keys found"
        return keys
    raise ValueError(f"unknown mode {mode!r}")


def extract_b_global(state_dict: dict, n_layers: int = N_LAYERS):
    """Reads the Arm-2' (global) frozen bias vector directly from its own
    checkpoint's buffer -- `frozen_bias_global_vector(table)` (lm_pretrain_rd.py
    sec 5) is IDENTICAL across every layer (same seed, sec 2/sec 5's own
    per-layer-independent-but-identically-seeded construction, verified in
    source), so block 0's copy is sufficient. Returns None if the checkpoint
    has no such buffer (i.e. it is not an Arm-2' 'global' checkpoint)."""
    for i in range(n_layers):
        k = f"blocks.{i}.mixer.frozen_bias_global_vec"
        if k in state_dict:
            return state_dict[k]
    return None


# ---------------------------------------------------------------------------
# Diff arithmetic.
# ---------------------------------------------------------------------------

def flat_delta(sd_late: dict, sd_early: dict, keys: list):
    """Returns (concatenated 1D double delta vector, {key: delta_tensor})."""
    import torch
    per_key = {}
    parts = []
    for k in keys:
        assert sd_late[k].shape == sd_early[k].shape, (
            f"{k}: shape mismatch {tuple(sd_late[k].shape)} vs {tuple(sd_early[k].shape)}")
        d = (sd_late[k].double() - sd_early[k].double())
        per_key[k] = d
        parts.append(d.flatten())
    return torch.cat(parts), per_key


def frobenius_norm(vec) -> float:
    return vec.norm().item()


def cosine(a, b) -> float | None:
    af, bf = a.flatten().double(), b.flatten().double()
    na, nb = af.norm().item(), bf.norm().item()
    if na == 0.0 or nb == 0.0:
        return None
    return (af @ bf).item() / (na * nb)


def structural_cosine_report(per_key_deltas: dict, b_global) -> dict:
    """sec 12.3.3: 'cos(ΔW(Arm2′), b_global) where structurally meaningful
    (e.g. if a bias/weight column can be reshaped to (d_state,))'. For each
    per-key delta tensor, if some axis has length == b_global.numel(), slice
    along that axis into (d_state,)-shaped 'columns' and report cosine per
    column plus a mean-|cosine| summary; otherwise disclose the tensor as not
    structurally comparable rather than silently skipping it."""
    if b_global is None:
        return {"note": "no b_global available (not a global-arm checkpoint)"}
    d_state = b_global.numel()
    out = {}
    for k, delta in per_key_deltas.items():
        shape = list(delta.shape)
        if d_state in shape:
            axis = shape.index(d_state)
            moved = delta.movedim(axis, -1).reshape(-1, d_state)
            cos_per_col = [cosine(moved[i], b_global) for i in range(moved.shape[0])]
            valid = [c for c in cos_per_col if c is not None]
            out[k] = {
                "shape": shape, "structurally_comparable": True,
                "cosine_per_column": cos_per_col,
                "mean_abs_cosine": (sum(abs(c) for c in valid) / len(valid)) if valid else None,
                "max_abs_cosine": max((abs(c) for c in valid), default=None),
            }
        else:
            out[k] = {"shape": shape, "structurally_comparable": False,
                       "note": f"no axis of length d_state={d_state}"}
    return out


def paramdiff_report(sd_arm1_early: dict, sd_arm1_late: dict, sd_arm2_late: dict,
                      sd_arm2p_late: dict, mode: str) -> dict:
    """Sec 12.3.3's core comparison for ONE parameter-subset mode ('k_conv1d'
    primary or 'block0_full' secondary). ΔW(arm) := W(arm, step_late) -
    W(Arm1, step_early) for every arm INCLUDING Arm1 itself (its own natural
    drift is the reference/control the other two are read against)."""
    keys = select_keys(sd_arm1_late, mode)
    b_global = extract_b_global(sd_arm2p_late)

    d_arm1, per_key_arm1 = flat_delta(sd_arm1_late, sd_arm1_early, keys)
    d_arm2, per_key_arm2 = flat_delta(sd_arm2_late, sd_arm1_early, keys)
    d_arm2p, per_key_arm2p = flat_delta(sd_arm2p_late, sd_arm1_early, keys)

    return {
        "mode": mode, "keys": keys,
        "arm1_natural_drift": {
            "frobenius_norm": frobenius_norm(d_arm1),
            "cosine_to_b_global": structural_cosine_report(per_key_arm1, b_global),
        },
        "arm2_drift_vs_arm1_step_early": {
            "frobenius_norm": frobenius_norm(d_arm2),
            "cosine_to_b_global": structural_cosine_report(per_key_arm2, b_global),
        },
        "arm2prime_drift_vs_arm1_step_early": {
            "frobenius_norm": frobenius_norm(d_arm2p),
            "cosine_to_b_global": structural_cosine_report(per_key_arm2p, b_global),
        },
        "norm_bigger_reading_ONLY": {
            "note": "sec 12.3.3: norm comparison ALONE is NOT diagnostic of compensation -- "
                     "directionality (the cosine_to_b_global block above) is. Reported for "
                     "completeness, not as the pre-registered reading.",
            "arm2_norm_gt_arm2prime_norm": frobenius_norm(d_arm2) > frobenius_norm(d_arm2p),
        },
    }


def early_step_crossarm_check(sd_arm1_early: dict, sd_arm2_early: dict, sd_arm2p_early: dict,
                               mode: str) -> dict:
    """Sanity check on the shared-baseline assumption (ΔW uses ONLY Arm1's
    step_early checkpoint): if Arm2/Arm2' step_early state is already far from
    Arm1's step_early state, using Arm1's alone as the common baseline for all
    three arms would be a bad assumption. Reports how close the three arms
    already are at step_early."""
    keys = select_keys(sd_arm1_early, mode)
    d_arm2_early, _ = flat_delta(sd_arm2_early, sd_arm1_early, keys)
    d_arm2p_early, _ = flat_delta(sd_arm2p_early, sd_arm1_early, keys)
    return {
        "mode": mode,
        "arm2_vs_arm1_at_step_early_frobenius_norm": frobenius_norm(d_arm2_early),
        "arm2prime_vs_arm1_at_step_early_frobenius_norm": frobenius_norm(d_arm2p_early),
        "note": "small values here support using Arm1's step_early checkpoint as a shared "
                 "baseline for all three arms' ΔW; large values would undermine that assumption.",
    }


# ---------------------------------------------------------------------------
# Real-data driver (registered, NOT executed by this build pass -- see the
# orchestration note at the top of this file).
# ---------------------------------------------------------------------------

def _resolve_ckpt_path(args, arm: str, step: int) -> str:
    lam_tag = {"off": "lam0p00", "per_token": "lam0p58", "global": "lam0p58"}[arm]
    cell = f"frozenbias_lm_{arm}_{lam_tag}_{args.corpus}_dm256_ds64_L2_s{args.seed}"
    run_name = f"lmC_{args.corpus}_dm256_ds64_L2_s{args.seed}"
    if args.flat_dir:
        return os.path.join(args.flat_dir, f"{cell}__step{step}.pt")
    assert args.ckpt_base_dir, "need --ckpt-base-dir or --flat-dir"
    return os.path.join(args.ckpt_base_dir, cell, f"{run_name}_step{step}.pt")


def run_real(args) -> int:
    paths = {
        ("off", args.step_early): _resolve_ckpt_path(args, "off", args.step_early),
        ("off", args.step_late): _resolve_ckpt_path(args, "off", args.step_late),
        ("per_token", args.step_early): _resolve_ckpt_path(args, "per_token", args.step_early),
        ("per_token", args.step_late): _resolve_ckpt_path(args, "per_token", args.step_late),
        ("global", args.step_early): _resolve_ckpt_path(args, "global", args.step_early),
        ("global", args.step_late): _resolve_ckpt_path(args, "global", args.step_late),
    }
    print("resolved checkpoint paths (6 files):")
    for k, p in paths.items():
        print(f"  {k}: {p}")
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print(f"\nMISSING {len(missing)}/6 checkpoint file(s) -- this script is registered as a "
              f"run step, not executed against real data, until these exist (box or scp'd copy):")
        for p in missing:
            print(f"  MISSING: {p}")
        return 2

    sd_arm1_early, _, _ = load_ckpt(paths[("off", args.step_early)])
    sd_arm1_late, _, _ = load_ckpt(paths[("off", args.step_late)])
    sd_arm2_early, _, _ = load_ckpt(paths[("per_token", args.step_early)])
    sd_arm2_late, _, _ = load_ckpt(paths[("per_token", args.step_late)])
    sd_arm2p_early, _, _ = load_ckpt(paths[("global", args.step_early)])
    sd_arm2p_late, _, _ = load_ckpt(paths[("global", args.step_late)])

    report = {
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 12.3.3 (H4 checkpoint parameter-diff)",
        "corpus": args.corpus, "seed": args.seed,
        "step_early": args.step_early, "step_late": args.step_late,
        "primary_k_conv1d": paramdiff_report(sd_arm1_early, sd_arm1_late, sd_arm2_late, sd_arm2p_late, "k_conv1d"),
        "secondary_block0_full": paramdiff_report(sd_arm1_early, sd_arm1_late, sd_arm2_late, sd_arm2p_late, "block0_full"),
        "shared_baseline_sanity_check": {
            "k_conv1d": early_step_crossarm_check(sd_arm1_early, sd_arm2_early, sd_arm2p_early, "k_conv1d"),
            "block0_full": early_step_crossarm_check(sd_arm1_early, sd_arm2_early, sd_arm2p_early, "block0_full"),
        },
    }
    payload = wrap_exploratory(report)
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "mech_wave", "mech_h4_paramdiff_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Self-test -- synthetic fake checkpoints, exact key-naming convention,
# EXERCISES real torch.save/torch.load + the full diff/norm/cosine pipeline.
# Run to completion (this project's own standing "run the negative unit test
# to completion" rule) -- this is what stands in for real execution in this
# CPU-only, no-real-checkpoints Stage-0 build pass.
# ---------------------------------------------------------------------------

def _make_fake_state_dict(arm: str, d_state: int, kernel_size: int, n_layers: int,
                           k_conv1d_delta_source=None, seed: int = 0):
    """Builds a tiny fake state_dict with the EXACT real key-naming convention.
    `k_conv1d_delta_source`: if given, block-0's k_conv1d.weight is set to
    `base + k_conv1d_delta_source` (base shared across arms via `seed`, so a
    caller can plant an EXACT, known ΔW for block 0 while blocks 1..n-1 get
    independent small random drift, mirroring a real multi-layer model where
    only SOME layers might show a planted effect)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    sd = {}
    base_block0 = torch.randn(d_state, 1, kernel_size, generator=g, dtype=torch.float64)
    for i in range(n_layers):
        if i == 0:
            w = base_block0.clone()
            if k_conv1d_delta_source is not None:
                w = w + k_conv1d_delta_source
        else:
            w = torch.randn(d_state, 1, kernel_size, generator=g, dtype=torch.float64) * 0.01
        sd[f"blocks.{i}.mixer.k_conv1d.weight"] = w
        sd[f"blocks.{i}.mixer.q_conv1d.weight"] = torch.randn(d_state, 1, kernel_size, generator=g, dtype=torch.float64)
        sd[f"blocks.{i}.mixer.v_conv1d.weight"] = torch.randn(d_state, 1, kernel_size, generator=g, dtype=torch.float64)
        sd[f"blocks.{i}.mixer.norm.weight"] = torch.ones(d_state, dtype=torch.float64)
    if arm == "per_token":
        vocab = 37  # tiny toy vocab, not the real 50,257 -- self-test only cares about key shape/naming
        table = torch.randn(vocab, d_state, generator=g, dtype=torch.float64)
        for i in range(n_layers):
            sd[f"blocks.{i}.mixer.frozen_bias_table"] = table.clone()
    elif arm == "global":
        import torch.nn.functional as F
        vocab = 37
        table = torch.randn(vocab, d_state, generator=g, dtype=torch.float64)
        gvec = F.normalize(table.mean(dim=0), dim=-1)
        for i in range(n_layers):
            sd[f"blocks.{i}.mixer.frozen_bias_global_vec"] = gvec.clone()
    return sd


def _selftest() -> bool:
    import torch
    ok = True
    d_state, kernel_size, n_layers = 4, 3, 2

    print("[setup] building fake Arm1(off)/Arm2(per_token)/Arm2'(global) state_dicts, "
          "step_early and step_late, with a PLANTED coherent drift for Arm2' block-0 "
          "k_conv1d.weight and an unrelated (incoherent) drift for Arm2's block-0")

    sd_arm1_early = _make_fake_state_dict("off", d_state, kernel_size, n_layers, seed=1)
    sd_arm1_late = _make_fake_state_dict("off", d_state, kernel_size, n_layers, seed=1)  # tiny natural drift only (independent random draw)

    # Get Arm2'/Arm2' step_early's b_global to plant a KNOWN-coherent delta against.
    sd_arm2p_early = _make_fake_state_dict("global", d_state, kernel_size, n_layers, seed=1)
    b_global = sd_arm2p_early[f"blocks.0.mixer.frozen_bias_global_vec"]

    # Arm2' step_late: block-0 k_conv1d.weight's diff-from-arm1-early is planted as an EXACT
    # scalar multiple of b_global broadcast across the kernel axis -- a perfectly coherent drift.
    planted_source = 1.7 * b_global.reshape(d_state, 1, 1).expand(d_state, 1, kernel_size).clone()
    sd_arm2p_late = _make_fake_state_dict("global", d_state, kernel_size, n_layers, seed=1)
    sd_arm2p_late["blocks.0.mixer.k_conv1d.weight"] = (
        sd_arm1_early["blocks.0.mixer.k_conv1d.weight"] + planted_source)

    # Arm2 (per_token) step_early/late: ordinary independent random state -- its block-0
    # k_conv1d.weight diff-from-arm1-early has NO relationship to b_global (incoherent by
    # construction of this synthetic test).
    sd_arm2_early = _make_fake_state_dict("per_token", d_state, kernel_size, n_layers, seed=2)
    sd_arm2_late = _make_fake_state_dict("per_token", d_state, kernel_size, n_layers, seed=3)

    with tempfile.TemporaryDirectory() as td:
        paths = {}
        for name, sd in [("arm1_early", sd_arm1_early), ("arm1_late", sd_arm1_late),
                          ("arm2_early", sd_arm2_early), ("arm2_late", sd_arm2_late),
                          ("arm2p_early", sd_arm2p_early), ("arm2p_late", sd_arm2p_late)]:
            p = os.path.join(td, f"{name}.pt")
            torch.save({"step": 1000 if "early" in name else 20000, "model_state_dict": sd,
                        "config": {"d_state": d_state, "n_layers": n_layers}, "corpus": "fake",
                        "seed": 0, "run_name": "selftest"}, p)
            paths[name] = p

        print("[test 1] real torch.save/torch.load round-trip preserves tensors exactly")
        loaded_sd, loaded_cfg, loaded_step = load_ckpt(paths["arm2p_late"])
        t1 = torch.equal(loaded_sd["blocks.0.mixer.k_conv1d.weight"],
                          sd_arm2p_late["blocks.0.mixer.k_conv1d.weight"]) and loaded_step == 20000
        print(f"    round-trip exact + step preserved  PASS={t1}")
        ok = ok and t1

        sd1e, _, _ = load_ckpt(paths["arm1_early"])
        sd1l, _, _ = load_ckpt(paths["arm1_late"])
        sd2l, _, _ = load_ckpt(paths["arm2_late"])
        sd2pl, _, _ = load_ckpt(paths["arm2p_late"])

        report = paramdiff_report(sd1e, sd1l, sd2l, sd2pl, "k_conv1d")

        print("[test 2] extract_b_global reads the SAME vector planted above")
        recovered_b_global = extract_b_global(sd2pl)
        t2 = torch.allclose(recovered_b_global, b_global)
        print(f"    recovered b_global matches planted b_global  PASS={t2}")
        ok = ok and t2

        print("[test 3] Arm2' block-0 k_conv1d.weight cosine-to-b_global ~= 1.0 (planted coherent)")
        arm2p_cos = report["arm2prime_drift_vs_arm1_step_early"]["cosine_to_b_global"][
            "blocks.0.mixer.k_conv1d.weight"]
        print(f"    cosine_per_column={arm2p_cos['cosine_per_column']}")
        t3 = arm2p_cos["structurally_comparable"] and all(abs(c - 1.0) < 1e-6 for c in arm2p_cos["cosine_per_column"])
        print(f"    all columns cosine ~= 1.0  PASS={t3}")
        ok = ok and t3

        print("[test 4] Arm2 block-0 k_conv1d.weight cosine-to-b_global is NOT near 1.0 (incoherent, unplanted)")
        arm2_cos = report["arm2_drift_vs_arm1_step_early"]["cosine_to_b_global"]["blocks.0.mixer.k_conv1d.weight"]
        print(f"    mean_abs_cosine={arm2_cos['mean_abs_cosine']:.6f}")
        t4 = arm2_cos["mean_abs_cosine"] < 0.9  # not artificially aligned -- generous margin below the planted case's ~1.0
        print(f"    mean_abs_cosine < 0.9 (i.e. clearly less coherent than Arm2's ~1.0)  PASS={t4}")
        ok = ok and t4

        print("[test 5] H4-consistent comparative reading: Arm2' cosine > Arm2 cosine on the planted block")
        t5 = min(abs(c) for c in arm2p_cos["cosine_per_column"]) > arm2_cos["mean_abs_cosine"]
        print(f"    Arm2' coherence > Arm2 coherence on block-0 k_conv1d.weight  PASS={t5}")
        ok = ok and t5

        print("[test 6] block0_full mode includes q_conv1d/v_conv1d/norm but EXCLUDES frozen buffers")
        keys_full = select_keys(sd2pl, "block0_full")
        t6 = (
            "blocks.0.mixer.k_conv1d.weight" in keys_full
            and "blocks.0.mixer.q_conv1d.weight" in keys_full
            and "blocks.0.mixer.frozen_bias_global_vec" not in keys_full
        )
        print(f"    keys_full sample: {keys_full}")
        print(f"    includes q/v/k_conv1d, excludes frozen_bias_global_vec  PASS={t6}")
        ok = ok and t6

        print("[test 7] Arm1 (off) checkpoint has no b_global -- structural_cosine_report degrades honestly")
        sd1l_report = structural_cosine_report(
            flat_delta(sd1l, sd1e, select_keys(sd1e, "k_conv1d"))[1], extract_b_global(sd1l))
        t7 = sd1l_report.get("note") == "no b_global available (not a global-arm checkpoint)"
        print(f"    Arm1 has no frozen_bias_global_vec, reported honestly  PASS={t7}")
        ok = ok and t7

        print("[test 8] shared-baseline sanity check runs without error and returns finite norms")
        sanity = early_step_crossarm_check(sd1e, load_ckpt(paths["arm2_early"])[0],
                                            load_ckpt(paths["arm2p_early"])[0], "k_conv1d")
        t8 = (sanity["arm2_vs_arm1_at_step_early_frobenius_norm"] >= 0
              and sanity["arm2prime_vs_arm1_at_step_early_frobenius_norm"] >= 0)
        print(f"    sanity check norms: {sanity}")
        print(f"    both finite/nonnegative  PASS={t8}")
        ok = ok and t8

    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ckpt-base-dir", type=str, default=None,
                     help="box path root, e.g. /data/deltanet_rd_frozenbias_ckpts")
    ap.add_argument("--flat-dir", type=str, default=None,
                     help="local dir of scp'd files named <cell>__step<n>.pt (Option B)")
    ap.add_argument("--corpus", type=str, default="openr1-mix-ext")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step-early", type=int, default=1000)
    ap.add_argument("--step-late", type=int, default=20000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.self_test:
        ok = _selftest()
        print("=" * 60)
        print("mech_h4_paramdiff SELF-TEST: " + ("ALL PASSED" if ok else "FAILURES PRESENT"))
        return 0 if ok else 1

    return run_real(args)


if __name__ == "__main__":
    sys.exit(main())
