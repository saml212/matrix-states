"""reasoning_link_validation.py -- REASONING_LINK_DESIGN.md sec 17.6
(validation pre-registration, 2026-07-11). Implements items 1 (label-shuffle
null) and 2 (seed-bimodality resample) of the sec 17.6 charter by running
TWO forward-A/forward-B passes per committed grid cell (the SAME 80-cell
enumeration `reasoning_link_remetric_chain.sh` uses) on the SAME already-
archived checkpoints: (a) the ORIGINAL registered eval seed, carrying its
own label-shuffle null grading (fused into the same pass, via
`measure_cell_all_h`'s existing `null_seed` mechanism -- zero extra forward
cost), and (b) a freshly-RESAMPLED seed (RESAMPLE_OFFSET, sec 17.6), also
carrying its own null grading, for the episode-resampling stability check.

Item 3 (per_token h>=2 concentration) is pure analysis over the ALREADY-
ARCHIVED sec 17.4 `04_remetric/` JSONs -- no new pass, not implemented in
this file (see `reasoning_link_validation_analyze.py`, which also digests
this script's own output for items 1/2).

Deliberately does NOT modify `run_cell`/the CLI's `--mode cell` (sec 17.6's
own "closes the gap with a NEW script, not a rebuild" framing) -- this file
duplicates run_cell's checkpoint-resolution/episode-seed plumbing (the SAME
duplication convention every sibling script in this directory already uses,
e.g. `load_checkpoint`'s own "pod-safety: duplicated, not cross-imported"
docstring) rather than adding a second parameter surface to the shared,
just-audited entry point.

Run standalone:
    python reasoning_link_validation.py --selftest        # no GPU, no checkpoints
    python reasoning_link_validation.py --mode cell --family leg_a --arm per_token \
        --corpus wikitext-mix-ext --ckpt-seed 1 --k 32 --surgery native \
        --device cuda --out results/reasoning_link_validation/leg_a_per_token_wikitext-mix-ext_s1_k32_native.json
    python reasoning_link_validation.py --mode all --device cuda \
        --out-dir results/reasoning_link_validation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports, matches every sibling script in this dir

import reasoning_link_probe as rlp  # noqa: E402

DESIGN_REF = "REASONING_LINK_DESIGN.md sec 17.6 (2026-07-11 validation pre-registration)"

# sec 17.6 item 2's construction: a LOCAL convention scoped to THIS script only -- never edits
# reasoning_link_probe.PURPOSE_BASE/episode_seed itself. Verified >35x margin above the current
# schema's own max representable value (see `max_registered_seed_value` below, exercised by
# --selftest).
RESAMPLE_OFFSET = 10 ** 9

# sec 17.6's registered absolute floor for "coherent signal" (reused verbatim from
# rlp.H1_ABSOLUTE_FLOOR, never re-derived) -- applied at every h, not just h=1, for item 2's
# cell classification and item 1's decision rule's condition (b).
ABSOLUTE_FLOOR = rlp.H1_ABSOLUTE_FLOOR
assert ABSOLUTE_FLOOR == 0.10, f"rlp.H1_ABSOLUTE_FLOOR drifted from the sec 17.6-cited 0.10: {ABSOLUTE_FLOOR}"


def max_registered_seed_value() -> int:
    """sec 17.6's own bound computation, reproduced here as a checkable function (not just prose) --
    exercised by --selftest so a future PURPOSE_BASE/STRIDE_* edit in reasoning_link_probe.py cannot
    silently shrink RESAMPLE_OFFSET's safety margin without this script's own selftest catching it."""
    return (max(rlp.PURPOSE_BASE.values()) + max(rlp.LEG_BASE.values())
            + 3 * rlp.STRIDE_CONDITION + 1 * rlp.STRIDE_CORPUS
            + 11 * rlp.STRIDE_SEED + 5 * rlp.STRIDE_K)


# ---------------------------------------------------------------------------
# sec 17.6's 80-cell enumeration -- IDENTICAL loop structure/params to
# `reasoning_link_remetric_chain.sh` (verified by direct comparison against that file), reproduced
# here as data (a list of dicts) rather than shell loops, so one Python process can iterate all 80
# cells without 80 separate process launches (each of which would reload the tokenizer/model from
# scratch via bash -- this keeps that same per-cell reload behavior for now, matching the shell
# chain's own cost profile exactly, but affords a single `--mode all` invocation for tmux
# convenience).
# ---------------------------------------------------------------------------

LEG_A_ARMS = ("off", "per_token", "global")
LEG_A_CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")
LEG_A_SEEDS = (0, 1, 2)
LEG_A_KS = (20, 32)

LEG_B_CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")


def enumerate_cells() -> list[dict]:
    cells = []
    for arm in LEG_A_ARMS:
        for corpus in LEG_A_CORPORA:
            for seed in LEG_A_SEEDS:
                for k in LEG_A_KS:
                    for surgery in ("native", "off"):
                        if arm == "off" and surgery == "off":
                            continue
                        cells.append({
                            "leg": "leg_a", "arm": arm, "corpus": corpus, "ckpt_seed": seed,
                            "K": k, "surgery": surgery, "batch_size": 16,
                            "cell_name": f"leg_a_{arm}_{corpus}_s{seed}_k{k}_{surgery}",
                        })
    for rung in (0, 1, 2, 3):
        k = 32 if rung < 2 else 64
        seeds = (0,) if rung == 3 else (0, 1, 2)
        cell_batch = 4 if rung == 3 else (8 if rung == 2 else 16)
        for corpus in LEG_B_CORPORA:
            for seed in seeds:
                cells.append({
                    "leg": "leg_b", "rung": rung, "corpus": corpus, "ckpt_seed": seed,
                    "K": k, "surgery": "native", "batch_size": cell_batch,
                    "cell_name": f"leg_b_rung{rung}_{corpus}_s{seed}_k{k}",
                })
    assert len(cells) == 80, f"expected 80 cells (60 leg_a + 20 leg_b), got {len(cells)}"
    return cells


def resolve_ckpt_path(cell: dict, frozen_bias_root: str, trackc_root: str) -> str:
    if cell["leg"] == "leg_a":
        lam = 0.0 if cell["arm"] == "off" else 0.58
        return rlp.leg_a_ckpt_path(frozen_bias_root, cell["arm"], lam, cell["corpus"], cell["ckpt_seed"])
    return rlp.leg_b_ckpt_path_final(trackc_root, cell["rung"], cell["corpus"], cell["ckpt_seed"])


def seeds_for_cell(cell: dict) -> dict:
    """sec 17.6 item 1/2's exact seed construction, reproduced as one checkable function."""
    leg = cell["leg"]
    if leg == "leg_a":
        condition_idx = rlp.LEG_A_ARM_INDEX[cell["arm"]]
    else:
        condition_idx = cell["rung"]
    corpus_idx = rlp.CORPUS_INDEX[cell["corpus"]]
    k_idx = rlp.K_INDEX[cell["K"]]
    ckpt_seed_idx = cell["ckpt_seed"]
    seed_eval = rlp.episode_seed("eval", leg, condition_idx, corpus_idx, ckpt_seed_idx, k_idx)
    null_seed = rlp.episode_seed("null_shuffle", leg, condition_idx, corpus_idx, ckpt_seed_idx, k_idx)
    return {
        "seed_eval": seed_eval, "null_seed": null_seed,
        "seed_resample": seed_eval + RESAMPLE_OFFSET, "null_seed_resample": null_seed + RESAMPLE_OFFSET,
    }


# ---------------------------------------------------------------------------
# Per-(cell,h) decision rule -- sec 17.6 item 1's pre-registered "NULL-CLEARS" test, a pure function
# so it can be unit-tested (mirrors this codebase's own premise_action_rule/killer_prediction_verdict
# convention: decision rules are pure functions over already-computed numbers, never inline
# if/else scattered through a harvest script).
# ---------------------------------------------------------------------------

def null_clears(real_recovered_frac: float, null_recovered_frac: float, null_ci_upper: float,
                 null_width: float, absolute_floor: float = ABSOLUTE_FLOOR) -> dict:
    """sec 17.6 item 1: BOTH conditions required.
    (a) charter's explicit trivial-artifact bar: null < 0.5 * real.
    (b) null-relative statistical test (h1_sanity_floor's formula, generalized to every h) AND the
        registered absolute floor."""
    cond_a = null_recovered_frac < 0.5 * real_recovered_frac if real_recovered_frac > 0 else False
    null_relative_pass = real_recovered_frac > (null_ci_upper + null_width)
    absolute_pass = real_recovered_frac >= absolute_floor
    cond_b = null_relative_pass and absolute_pass
    return {
        "cond_a_half_reproduction": cond_a, "cond_b_null_relative_and_floor": cond_b,
        "null_clears": bool(cond_a and cond_b),
    }


def coherently_on(recovered_frac: float, floor: float = ABSOLUTE_FLOOR) -> bool:
    """sec 17.6 item 2's cell classification floor -- reused verbatim, not the raw >0 criterion."""
    return recovered_frac >= floor


# ---------------------------------------------------------------------------
# Cell measurement -- two variants (original-seed, resampled-seed), each carrying its own null
# grading, sharing ONE loaded checkpoint (loaded once per cell, matching run_cell's own per-call
# load -- this script still reloads per cell, matching the shell chain's cost profile, see the
# module docstring's own note on this).
# ---------------------------------------------------------------------------

def measure_cell_validation(ckpt_path: str, cell: dict, device: str) -> dict:
    model, ckpt = rlp.load_checkpoint(ckpt_path, device)
    conv_size = ckpt["config"]["conv_size"]
    episode_cfg = rlp.episode_config_for_checkpoint(conv_size, cell["K"])
    assert ckpt["config"]["conv_size"] == episode_cfg.conv_size, "conv_size mismatch (Stage -1 item 7 analog)"
    readout_layer = rlp.readout_layer_index(model)
    pools, pool_report = rlp.build_reasoning_link_pools(seed=0)  # matches run_cell's own seed=0 pool convention

    seeds = seeds_for_cell(cell)

    per_h_orig, fcounts_orig = rlp.measure_cell_all_h(
        model, episode_cfg, pools, readout_layer, cell["K"], rlp.H_TEST, cell["batch_size"],
        seeds["seed_eval"], cell["surgery"], device, compute_option2=False, compute_premises=True,
        null_seed=seeds["null_seed"])
    rlp.assert_forward_call_counts(fcounts_orig, context=f"validation-original {cell['cell_name']}")

    per_h_resamp, fcounts_resamp = rlp.measure_cell_all_h(
        model, episode_cfg, pools, readout_layer, cell["K"], rlp.H_TEST, cell["batch_size"],
        seeds["seed_resample"], cell["surgery"], device, compute_option2=False, compute_premises=True,
        null_seed=seeds["null_seed_resample"])
    rlp.assert_forward_call_counts(fcounts_resamp, context=f"validation-resample {cell['cell_name']}")

    return {
        "design_ref": DESIGN_REF, "program": "REASONING-LINK sec 17.6 validation",
        "stage": "validation-cell", "timestamp": time.time(),
        "cell": cell, "ckpt_path": ckpt_path, "ckpt_config": ckpt["config"],
        "readout_layer": readout_layer, "n_layers": model.n_layers,
        "seeds": seeds, "resample_offset": RESAMPLE_OFFSET,
        "pool_report_marker": pool_report["marker_word"],
        "per_h_original": per_h_orig, "per_h_resample": per_h_resamp,
        "forward_counts_original": fcounts_orig, "forward_counts_resample": fcounts_resamp,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _cell_is_done(out_path: str) -> bool:
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        return False
    try:
        with open(out_path) as f:
            d = json.load(f)
        return bool(d.get("forward_counts_original")) and bool(d.get("forward_counts_resample"))
    except Exception:
        return False


def run_all(device: str, out_dir: str, frozen_bias_root: str, trackc_root: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cells = enumerate_cells()
    t0 = time.time()
    n_done = 0
    for cell in cells:
        out_path = os.path.join(out_dir, f"{cell['cell_name']}.json")
        if _cell_is_done(out_path):
            print(f"SKIP (already complete): {out_path}")
            n_done += 1
            continue
        ckpt_path = resolve_ckpt_path(cell, frozen_bias_root, trackc_root)
        t_cell = time.time()
        result = measure_cell_validation(ckpt_path, cell, device)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        n_done += 1
        print(f"[{n_done}/80] {cell['cell_name']}: wrote {out_path} ({time.time() - t_cell:.1f}s)")
    wall = time.time() - t0
    with open(os.path.join(out_dir, "REASONING_LINK_VALIDATION_DONE"), "w") as f:
        f.write(f"{n_done}/80 cells complete, wall={wall:.1f}s\n")
    print(f"TOTAL WALL: {wall:.1f}s -- {n_done}/80 cells complete.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["cell", "all"], default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--family", choices=["leg_a", "leg_b"], default=None)
    ap.add_argument("--arm", choices=list(rlp.FROZEN_BIAS_ARM_MODES), default="off")
    ap.add_argument("--rung", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--corpus", type=str, default="openr1-mix-ext")
    ap.add_argument("--ckpt-seed", type=int, default=0)
    ap.add_argument("--ckpt-base-dir", type=str, default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--surgery", choices=["native", "off"], default="native")
    ap.add_argument("--batch-size", type=int, default=rlp.BATCH_SIZE_DEFAULT)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="results/reasoning_link_validation")
    ap.add_argument("--frozen-bias-root", type=str, default=rlp.FROZEN_BIAS_CKPT_ROOT_DEFAULT)
    ap.add_argument("--trackc-root", type=str, default=rlp.TRACKC_CKPT_ROOT_DEFAULT)
    ap.add_argument("--device", type=str,
                     default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.mode == "all":
        run_all(args.device, args.out_dir, args.frozen_bias_root, args.trackc_root)
        return

    if args.mode == "cell":
        if args.family == "leg_a":
            cell = {"leg": "leg_a", "arm": args.arm, "corpus": args.corpus, "ckpt_seed": args.ckpt_seed,
                    "K": args.k, "surgery": args.surgery, "batch_size": args.batch_size,
                    "cell_name": f"leg_a_{args.arm}_{args.corpus}_s{args.ckpt_seed}_k{args.k}_{args.surgery}"}
        elif args.family == "leg_b":
            cell = {"leg": "leg_b", "rung": args.rung, "corpus": args.corpus, "ckpt_seed": args.ckpt_seed,
                    "K": args.k, "surgery": "native", "batch_size": args.batch_size,
                    "cell_name": f"leg_b_rung{args.rung}_{args.corpus}_s{args.ckpt_seed}_k{args.k}"}
        else:
            raise SystemExit("--mode cell requires --family leg_a|leg_b")
        ckpt_path = args.ckpt or resolve_ckpt_path(cell, args.frozen_bias_root, args.trackc_root)
        result = measure_cell_validation(ckpt_path, cell, args.device)
        print(json.dumps({k: v for k, v in result.items() if not k.startswith("per_h")}, indent=2, default=str))
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"wrote {args.out}")
        return

    raise SystemExit("nothing to do: pass --selftest or --mode cell|all")


# ---------------------------------------------------------------------------
# Selftest -- no GPU, no checkpoints, no real/stub fla needed at all (this module never imports
# anything kernel-touching beyond what reasoning_link_probe.py's own top-level import already
# pulls in, which installs the CPU stub automatically on a machine with no real fla -- see that
# file's _ensure_fla_stub. Every check below is pure Python/torch-CPU arithmetic).
# ---------------------------------------------------------------------------

def run_selftest() -> bool:
    import torch
    ok = True

    # 1. RESAMPLE_OFFSET collision-safety margin (sec 17.6's own bound, checkable not just prose).
    bound = max_registered_seed_value()
    expected_bound = 28_193_000
    if bound != expected_bound:
        print(f"SELFTEST FAIL: max_registered_seed_value()={bound}, sec 17.6 pre-registered {expected_bound} "
              f"-- reasoning_link_probe.py's PURPOSE_BASE/LEG_BASE/STRIDE_* constants drifted; re-derive "
              f"the sec 17.6 margin claim before trusting RESAMPLE_OFFSET's safety.")
        ok = False
    margin = RESAMPLE_OFFSET / bound
    if margin < 10:
        print(f"SELFTEST FAIL: RESAMPLE_OFFSET margin only {margin:.1f}x over max_registered_seed_value "
              f"({bound}) -- sec 17.6 claims >35x, expected a wide margin, not a near-collision.")
        ok = False

    # 2. build_null_labeling draws a genuinely different Hamiltonian K-cycle from the real one (pure
    #    CPU torch, no model/kernel needed -- grammar_rd._permutation_graph has no fla dependency).
    import grammar_rd
    B, K = 8, 32
    gen_true = torch.Generator().manual_seed(seeds_for_cell(
        {"leg": "leg_a", "arm": "off", "corpus": "openr1-mix-ext", "ckpt_seed": 0, "K": 32})["seed_eval"])
    gen_null = torch.Generator().manual_seed(seeds_for_cell(
        {"leg": "leg_a", "arm": "off", "corpus": "openr1-mix-ext", "ckpt_seed": 0, "K": 32})["null_seed"])
    succ_true = grammar_rd._permutation_graph(B, K, gen_true, torch.device("cpu"), dtype=torch.float32)
    succ_null = rlp.build_null_labeling(succ_true, gen_null)
    frac_agree = (succ_true == succ_null).float().mean().item()
    if frac_agree > 0.5:
        print(f"SELFTEST FAIL: succ_true and succ_null agree on {frac_agree:.2%} of entries -- the "
              f"null-shuffle draw is not independent of the real permutation (expected near-chance "
              f"agreement, ~1/K={1/K:.3f}, for two independently-drawn K-cycles).")
        ok = False
    if seeds_for_cell({"leg": "leg_a", "arm": "off", "corpus": "openr1-mix-ext", "ckpt_seed": 0, "K": 32})["seed_eval"] == \
       seeds_for_cell({"leg": "leg_a", "arm": "off", "corpus": "openr1-mix-ext", "ckpt_seed": 0, "K": 32})["null_seed"]:
        print("SELFTEST FAIL: seed_eval == null_seed for a real cell (would make the null draw a "
              "no-op duplicate of the real draw).")
        ok = False

    # 3. null_clears decision rule -- positive (signal survives) and negative (trivial artifact)
    #    fixtures, mirroring premise_action_rule/killer_prediction_verdict's own pure-function
    #    testing convention.
    strong_signal = null_clears(real_recovered_frac=0.80, null_recovered_frac=0.02,
                                 null_ci_upper=0.05, null_width=0.03)
    if not strong_signal["null_clears"]:
        print(f"SELFTEST FAIL: strong-signal fixture (real=0.80, null=0.02) did not NULL-CLEAR: {strong_signal}")
        ok = False

    trivial_artifact = null_clears(real_recovered_frac=0.30, null_recovered_frac=0.28,
                                    null_ci_upper=0.30, null_width=0.05)
    if trivial_artifact["null_clears"]:
        print(f"SELFTEST FAIL: trivial-artifact fixture (real=0.30, null=0.28, reproduces >=50%) "
              f"incorrectly NULL-CLEARED: {trivial_artifact}")
        ok = False

    below_floor = null_clears(real_recovered_frac=0.02, null_recovered_frac=0.00,
                               null_ci_upper=0.001, null_width=0.001)
    if below_floor["null_clears"]:
        print(f"SELFTEST FAIL: below-absolute-floor fixture (real=0.02 < 0.10 floor, null~0) "
              f"incorrectly NULL-CLEARED despite failing the absolute-floor condition: {below_floor}")
        ok = False
    if not below_floor["cond_a_half_reproduction"]:
        print(f"SELFTEST FAIL: below-absolute-floor fixture's condition (a) should still pass "
              f"vacuously (0.00 < 0.5*0.02) even though the overall verdict is False: {below_floor}")
        ok = False

    zero_real = null_clears(real_recovered_frac=0.0, null_recovered_frac=0.0,
                             null_ci_upper=0.0, null_width=0.0)
    if zero_real["null_clears"] or zero_real["cond_a_half_reproduction"]:
        print(f"SELFTEST FAIL: zero-real fixture (real=0.0) must not vacuously NULL-CLEAR "
              f"(0 < 0.5*0 is False, not True): {zero_real}")
        ok = False

    # 4. coherently_on floor.
    if coherently_on(0.099) or not coherently_on(0.10) or not coherently_on(0.50):
        print(f"SELFTEST FAIL: coherently_on floor boundary wrong -- 0.099={coherently_on(0.099)} "
              f"(expect False), 0.10={coherently_on(0.10)} (expect True), 0.50={coherently_on(0.50)} (expect True)")
        ok = False

    # 5. enumerate_cells produces exactly 80 cells with no duplicate cell_names (the exact grid
    #    reasoning_link_remetric_chain.sh runs, re-derived here as a checkable assertion rather than
    #    a re-read of that shell script's own loop bounds).
    cells = enumerate_cells()
    names = [c["cell_name"] for c in cells]
    if len(names) != 80 or len(set(names)) != 80:
        print(f"SELFTEST FAIL: enumerate_cells produced {len(names)} cells, {len(set(names))} unique "
              f"names (expected 80/80).")
        ok = False
    n_leg_a = sum(1 for c in cells if c["leg"] == "leg_a")
    n_leg_b = sum(1 for c in cells if c["leg"] == "leg_b")
    if n_leg_a != 60 or n_leg_b != 20:
        print(f"SELFTEST FAIL: leg split wrong -- leg_a={n_leg_a} (expect 60), leg_b={n_leg_b} (expect 20)")
        ok = False

    if ok:
        print("reasoning_link_validation --selftest: ALL CHECKS PASSED (RESAMPLE_OFFSET margin, "
              "null-draw independence, null_clears decision-rule fixtures [positive + 3 negative + "
              "boundary], coherently_on floor, 80-cell enumeration).")
    return ok


if __name__ == "__main__":
    main()
