"""reasoning_link_derange_control.py -- REASONING_LINK_DESIGN.md sec 17.6a
(addendum, 2026-07-11): the positional/derangement-slot null control.

The sec 17.6 item-1 succ-shuffle null is VACUOUS AT h=1 by construction
(prev_slot = _iterate_permutation(succ, a_slot, h-1) = a_slot at h=1 -- the
graded target never consults the permutation). This control breaks the
state<->target correspondence at EVERY h including h=1: grade pred(a,h)
against v_eff[delta(prev_slot)] where delta is a fresh per-row random
DERANGEMENT of the K slots (no fixed points -- reuses
reasoning_link_probe._random_derangement verbatim), so the graded slot is
never the correct one, while the value population / episode surface /
forward pass are held fixed.

One forward-A + one forward-B (main queries only -- premises/option2 not
needed, both cell-level facts already recorded by the sec 17.4 re-metric and
the sec 17.6 item-1 pass) per cell, same 80-cell grid, same eval seeds
(bit-identical episodes).

Run standalone:
    python reasoning_link_derange_control.py --selftest
    python reasoning_link_derange_control.py --mode all --device cuda \
        --out-dir results/reasoning_link_derange
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reasoning_link_probe as rlp  # noqa: E402
from reasoning_link_validation import (  # noqa: E402
    enumerate_cells, resolve_ckpt_path, seeds_for_cell, RESAMPLE_OFFSET, max_registered_seed_value)

DESIGN_REF = "REASONING_LINK_DESIGN.md sec 17.6a (2026-07-11 derangement-control addendum)"

# sec 17.6a's registered offset: disjoint from RESAMPLE_OFFSET's 1e9 block, >70x above the seed
# schema's own max representable value (28,193,000) -- same collision-safety argument, exercised
# by --selftest below.
DERANGE_OFFSET = 2_000_000_000
assert DERANGE_OFFSET != RESAMPLE_OFFSET


def measure_cell_derange(ckpt_path: str, cell: dict, device: str) -> dict:
    """Mirrors measure_cell_all_h's forward protocol (ONE forward-A + ONE forward-B for all h --
    the FATAL-1 discipline) but scores each h against BOTH the real target and the deranged-slot
    target from the SAME pred tensor."""
    import torch
    import grammar_rd

    model, ckpt = rlp.load_checkpoint(ckpt_path, device)
    conv_size = ckpt["config"]["conv_size"]
    episode_cfg = rlp.episode_config_for_checkpoint(conv_size, cell["K"])
    readout_layer = rlp.readout_layer_index(model)
    pools, pool_report = rlp.build_reasoning_link_pools(seed=0)
    seeds = seeds_for_cell(cell)
    derange_seed = seeds["null_seed"] + DERANGE_OFFSET
    surgery_off = (cell["surgery"] == "off")

    gen = torch.Generator(device=device).manual_seed(seeds["seed_eval"])
    batch = grammar_rd.sample_batch_rd(episode_cfg, cell["batch_size"], gen, hop_set=tuple(rlp.H_TEST),
                                        pools=pools, device=device)

    forward_counts = {"forward_a": 0, "forward_b": 0}
    final_states, k_raw, v_raw = rlp.run_forward_a(model, batch["token_ids"], surgery_off=surgery_off)
    forward_counts["forward_a"] += 1
    v_eff_items = rlp.gather_at_positions(v_raw[readout_layer], batch["item_pos"])
    S_T = rlp.squeeze_state_head(final_states[readout_layer])

    B, Q = batch["hops"].shape
    K = cell["K"]
    T_bind = batch["token_ids"].shape[1]
    query_len = batch["query_tokens"].shape[-1]
    concat = torch.cat([batch["token_ids"].unsqueeze(1).expand(B, Q, T_bind).reshape(B * Q, T_bind),
                        batch["query_tokens"].reshape(B * Q, query_len)], dim=1)
    _, _, q_raw = rlp.run_forward_b(model, concat, need_option2_hidden=False, surgery_off=surgery_off)
    forward_counts["forward_b"] += 1
    q_eff = q_raw[readout_layer][:, -1, :].view(B, Q, -1)
    rlp.assert_forward_call_counts(forward_counts, context=f"derange-control {cell['cell_name']}")

    gen_der = torch.Generator(device=device).manual_seed(derange_seed)
    derangement = rlp._random_derangement(B, K, gen_der, device)  # (B,K), no fixed points

    per_h = {}
    for h in rlp.H_TEST:
        hops_h = torch.full_like(batch["a_slot"], h)
        prev_slot, v_eff_target = rlp.compute_prev_slot_and_target(batch["succ"], batch["a_slot"],
                                                                     hops_h, v_eff_items)
        # sec 17.6a: deranged slot = delta(prev_slot) -- guaranteed != prev_slot per row by the
        # derangement's no-fixed-point property, so the graded target is NEVER the correct slot.
        deranged_slot = torch.gather(derangement, 1, prev_slot)
        v_eff_deranged = rlp.gather_at_positions(v_eff_items, deranged_slot)
        assert bool((deranged_slot == prev_slot).any().item()) is False, (
            "derangement produced a fixed point at some (row,query) -- _random_derangement's "
            "no-fixed-point guarantee is broken")

        pred = rlp.apply_state_power(S_T, q_eff, h)
        cos_real, rec_real = rlp.cosine_and_recovered(pred, v_eff_target)
        cos_der, rec_der = rlp.cosine_and_recovered(pred, v_eff_deranged)

        floor = rlp.h1_sanity_floor(rec_real.float().mean().item(), rec_der)
        per_h[h] = {
            "h": h, "K": K, "surgery": cell["surgery"],
            "recovered_frac": rec_real.float().mean().item(),
            "cos_mean": cos_real.mean().item(),
            "deranged_recovered_frac": rec_der.float().mean().item(),
            "deranged_cos_mean": cos_der.mean().item(),
            "n_scored": int(rec_real.numel()),
            # h1_sanity_floor's fields, with the DERANGED null in the null role (sec 17.6a's
            # registered decision rule reuses the identical formula):
            "deranged_ci_lower": floor["null_ci_lower"], "deranged_ci_upper": floor["null_ci_upper"],
            "deranged_width": floor["null_width"],
            "null_relative_pass": floor["null_relative_pass"], "absolute_pass": floor["absolute_pass"],
        }

    return {
        "design_ref": DESIGN_REF, "program": "REASONING-LINK sec 17.6a derangement control",
        "stage": "derange-control-cell", "timestamp": time.time(),
        "cell": cell, "ckpt_path": ckpt_path, "seeds": seeds, "derange_seed": derange_seed,
        "derange_offset": DERANGE_OFFSET, "readout_layer": readout_layer,
        "pool_report_marker": pool_report["marker_word"],
        "per_h": per_h, "forward_counts": forward_counts,
    }


def _cell_is_done(out_path: str) -> bool:
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        return False
    try:
        with open(out_path) as f:
            return bool(json.load(f).get("forward_counts"))
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
        result = measure_cell_derange(ckpt_path, cell, device)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        n_done += 1
        print(f"[{n_done}/80] {cell['cell_name']}: wrote {out_path} ({time.time() - t_cell:.1f}s)")
    wall = time.time() - t0
    with open(os.path.join(out_dir, "REASONING_LINK_DERANGE_DONE"), "w") as f:
        f.write(f"{n_done}/80 cells complete, wall={wall:.1f}s\n")
    print(f"TOTAL WALL: {wall:.1f}s -- {n_done}/80 cells complete.")


def run_selftest() -> bool:
    import torch
    ok = True

    # 1. DERANGE_OFFSET collision safety + block-disjointness from RESAMPLE_OFFSET.
    bound = max_registered_seed_value()
    if DERANGE_OFFSET / bound < 70:
        print(f"SELFTEST FAIL: DERANGE_OFFSET margin {DERANGE_OFFSET / bound:.1f}x < the sec 17.6a-"
              f"claimed >70x over max_registered_seed_value ({bound}).")
        ok = False
    # every derange seed lives in [2e9, 2e9+bound]; every resample seed in [1e9, 1e9+bound]; every
    # registered seed in [0, bound] -- disjoint iff bound < 1e9, which check 1 already guarantees
    # with huge margin. Assert anyway (cheap, explicit).
    if not (bound < RESAMPLE_OFFSET < DERANGE_OFFSET):
        print(f"SELFTEST FAIL: seed blocks overlap (bound={bound}, RESAMPLE={RESAMPLE_OFFSET}, "
              f"DERANGE={DERANGE_OFFSET}).")
        ok = False

    # 2. deranged_slot != prev_slot for every entry, by construction, on a synthetic example.
    gen = torch.Generator().manual_seed(0)
    B, K = 16, 20
    derangement = rlp._random_derangement(B, K, gen, torch.device("cpu"))
    if bool((derangement == torch.arange(K).unsqueeze(0)).any().item()):
        print("SELFTEST FAIL: _random_derangement returned a fixed point.")
        ok = False
    prev_slot = torch.randint(0, K, (B, 7))
    deranged_slot = torch.gather(derangement, 1, prev_slot)
    if bool((deranged_slot == prev_slot).any().item()):
        print("SELFTEST FAIL: delta(prev_slot) == prev_slot at some entry -- gather through a "
              "derangement must never map a slot to itself.")
        ok = False

    # 3. The control has TEETH on synthetic data (kill-proof run to completion, both directions):
    #    a perfect-correspondence S_T (identity-like map from orthonormal keys to distinct values)
    #    must score real >> deranged; a collapsed/rank-1 S_T with collinear values must score
    #    real ~= deranged (the trivial-geometry signature the control exists to catch).
    d = 32
    keys = torch.linalg.qr(torch.randn(d, d, generator=gen))[0][:K, :]      # K orthonormal keys
    vals_distinct = torch.linalg.qr(torch.randn(d, d, generator=gen))[0][:K, :]  # K orthonormal values
    S = (vals_distinct.T @ keys)                                             # S k_i = v_i exactly
    succ = torch.arange(K).roll(-1).unsqueeze(0)                             # one K-cycle
    a_slot = torch.arange(K).unsqueeze(0)                                    # query every slot
    hops_h = torch.ones_like(a_slot)
    prev = rlp.compute_prev_slot_and_target(succ, a_slot, hops_h, vals_distinct.unsqueeze(0))[0]
    tgt = rlp.gather_at_positions(vals_distinct.unsqueeze(0), prev)
    der = rlp._random_derangement(1, K, gen, torch.device("cpu"))
    tgt_der = rlp.gather_at_positions(vals_distinct.unsqueeze(0), torch.gather(der, 1, prev))
    pred = rlp.apply_state_power(S.unsqueeze(0), keys.unsqueeze(0), 1)
    _, rec_real = rlp.cosine_and_recovered(pred, tgt)
    _, rec_der = rlp.cosine_and_recovered(pred, tgt_der)
    if not (rec_real.float().mean().item() == 1.0 and rec_der.float().mean().item() == 0.0):
        print(f"SELFTEST FAIL: perfect-correspondence fixture should read real=1.0/deranged=0.0, got "
              f"real={rec_real.float().mean().item()}, deranged={rec_der.float().mean().item()} -- "
              f"the control would miss a GENUINE signal (false-negative teeth broken).")
        ok = False

    v_dir = torch.randn(d, generator=gen)
    v_dir = v_dir / v_dir.norm()
    vals_collinear = v_dir.unsqueeze(0).expand(K, d) + 0.01 * torch.randn(K, d, generator=gen)
    S_collapsed = torch.outer(v_dir, keys.mean(dim=0))                       # rank-1: every pred ~ v_dir
    tgt_c = rlp.gather_at_positions(vals_collinear.unsqueeze(0), prev)
    tgt_c_der = rlp.gather_at_positions(vals_collinear.unsqueeze(0), torch.gather(der, 1, prev))
    pred_c = rlp.apply_state_power(S_collapsed.unsqueeze(0), keys.unsqueeze(0), 1)
    _, rec_c_real = rlp.cosine_and_recovered(pred_c, tgt_c)
    _, rec_c_der = rlp.cosine_and_recovered(pred_c, tgt_c_der)
    r, dr = rec_c_real.float().mean().item(), rec_c_der.float().mean().item()
    if abs(r - dr) > 0.15 or r < 0.5:
        print(f"SELFTEST FAIL: collapsed-geometry fixture should read real ~= deranged, both high "
              f"(trivial artifact signature), got real={r}, deranged={dr} -- the control would "
              f"manufacture a false GENUINE verdict on degenerate geometry.")
        ok = False

    if ok:
        print("reasoning_link_derange_control --selftest: ALL CHECKS PASSED (offset margins/"
              "disjointness, no-fixed-point gather, kill-proof teeth in BOTH directions: "
              "perfect-correspondence real=1.0/deranged=0.0, collapsed-geometry real~=deranged).")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["all"], default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out-dir", type=str, default="results/reasoning_link_derange")
    ap.add_argument("--frozen-bias-root", type=str, default=rlp.FROZEN_BIAS_CKPT_ROOT_DEFAULT)
    ap.add_argument("--trackc-root", type=str, default=rlp.TRACKC_CKPT_ROOT_DEFAULT)
    ap.add_argument("--device", type=str,
                     default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if run_selftest() else 1)
    if args.mode == "all":
        run_all(args.device, args.out_dir, args.frozen_bias_root, args.trackc_root)
        return
    raise SystemExit("nothing to do: pass --selftest or --mode all")


if __name__ == "__main__":
    main()
