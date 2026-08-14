#!/usr/bin/env python3
"""Stage 0' (write-conditioning, Rev-5) -- amended per attack R4's A1-A8
(NCR_WRITECOND_ATTACK_R4.md sec STAGE-0' RULING), adopted verbatim by the
coordinator adjudication (NCR_WRITE_CONDITIONING_DESIGN.md
sec A4-ADJUDICATION). Eval-only + cheap parametrized training on a FRESH,
small (173,209-param) head -- no backbone training, no box-config change.
Deploy alongside the pinned `9a93198b` runner (SAME directory), same
discipline as the premise battery's own launch card -- imports the pinned
runner's real functions, reuses `nm.binexp_read`/`R.discriminability_
metrics`/`R.ortho_regularization_loss` verbatim, and imports THIS build's
own tested `stage0prime_helpers` module for every item's math (A1-A8) --
reinvents nothing.

Box-only (this file, like the pinned runner, cannot be imported off the
box -- `import ncr_lm_wave1_runner as R` pulls in `fla` via
`ncr_lm_wave1_smoke -> lm_pretrain_rd -> fla`). `stage0prime_helpers.py`
(THIS repo's own writecond_build/, imported below) is where every item's
CORE MATH lives and is unit-tested off-box -- see
test_stage0prime_helpers.py, run to completion as part of this revision.

DEPLOY: scp this file AND stage0prime_helpers.py AND write_supervision_
loss.py (its own dependency) into the SAME directory as
ncr_lm_wave1_runner.py on the box before running.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncr_lm_wave1_runner as R          # noqa: E402 -- the pinned runner (md5 9a93198b642242f512ff8489e32b0a53)
import ncr_models as nm                  # noqa: E402 -- binexp_read, real read path
from stage0prime_helpers import (        # noqa: E402 -- THIS build's tested A1-A8 item logic
    build_fresh_encoder,
    item_1_2_keygeom,
    item_3_reachability,
    item_4_transverse,
    item_5_ortho_conflict,
    item_6_achievability_probe,
)

BASE_SEED = 90210
N_EPISODES = 256
HOPS = (1, 13, 37, 61)         # the premise battery's own eval-ladder convention (pbe_repl.py), reused

# A1: checkpoint paths corrected to the battery's OWN recorded convention
# (verified directly: writecond_premise_REPL_{compA,primary}.json's own
# "ckpt" field, and pbe_supplement.py's hardcoded CKPT constant --
# `~/ncr_g3b31_contrastive/results/mob_g3b31_{tag}_s0_ckpts/
# mob_g3b31_{tag}_s0.ckpt.pt`, NOT `~/ncr_g3b31/results/
# mob_g3b31_{tag}_s0.ckpt.pt` -- wrong directory AND missing the
# `_ckpts/` level, F3.1's own finding).
CKPTS = {
    "primary": os.path.expanduser("~/ncr_g3b31_contrastive/results/mob_g3b31_primary_s0_ckpts/mob_g3b31_primary_s0.ckpt.pt"),
    "compA":   os.path.expanduser("~/ncr_g3b31_contrastive/results/mob_g3b31_compA_s0_ckpts/mob_g3b31_compA_s0.ckpt.pt"),
    "compB":   os.path.expanduser("~/ncr_g3b31_contrastive/results/mob_g3b31_compB_s0_ckpts/mob_g3b31_compB_s0.ckpt.pt"),
}
OUT_DIR = os.path.expanduser("~/ncr_writecond/results")
CEILING_S = 40 * 60          # m3: a real kill-switch -- wrap the tmux command in `timeout 2400`, A8


def extract_real_kv(arms, pools, cfg, device, n=N_EPISODES, seed=BASE_SEED, hop=61):
    """Items 1-5's shared TRAINING input, and item 6's TRAINING episodes:
    real (keys_v, values_v) from n real episodes, via the pinned forward
    path. A8: seed corrected to `+ EVAL_SEED_OFFSET + 61` (m2's own finding
    -- every OTHER eval in this program seeds `base_seed + EVAL_SEED_
    OFFSET + h`, runner:940; the un-amended card used a bare
    `+ EVAL_SEED_OFFSET`, drawing a DIFFERENT episode distribution than the
    one that produced P0=0.0664, making item 4's transverse-gain readings
    non-comparable to the harvest's own numbers)."""
    gen = torch.Generator(device=device).manual_seed(seed + R.EVAL_SEED_OFFSET + hop)
    probe = R.graft.build_task1_document(cfg, pools, gen, n, hop, device)
    with torch.no_grad():
        _, _, _, _, Z_sgd, keys_v, values_v = R.ncr_lm_forward_ablatable(
            arms["full_graft"]["backbone"], arms["full_graft"]["ncr"],
            arms["full_graft"]["integ"], probe, read_ablate=False, teacher_force=False)
    # A6's own probe-batch-retention requirement: entity_ids/tgt_slot/
    # query_key_col must survive (F3.4's finding: the un-amended card
    # discarded `probe` entirely, leaving item 6 with no way to score
    # retrieval at all).
    return Z_sgd, keys_v, values_v, probe


def extract_held_out(arms, pools, cfg, device, n, seed, hop):
    """A6/F3.5: item 6's held-out set, drawn at a DIFFERENT seed offset
    than the training draw AND at EVERY hop item 6 scores (F3.5's fix: one
    fresh document PER HOP, never an h=61-fit Z scored against a document
    built for a different h -- matches eval_arm_at_hops's own per-h
    document convention, runner:940-941)."""
    gen = torch.Generator(device=device).manual_seed(seed + R.EVAL_SEED_OFFSET + hop + 500_000)  # disjoint from training's own seed+hop
    batch = R.graft.build_task1_document(cfg, pools, gen, n, hop, device)
    with torch.no_grad():
        _, _, _, _, _, keys_v, values_v = R.ncr_lm_forward_ablatable(
            arms["full_graft"]["backbone"], arms["full_graft"]["ncr"],
            arms["full_graft"]["integ"], batch, read_ablate=False, teacher_force=False)
    q_key = arms["full_graft"]["integ"].query_key(batch["doc"][:, :-1], batch["query_key_col"],
                                                    arms["full_graft"]["backbone"].embed)
    return keys_v, values_v, q_key, batch["entity_ids"], batch["tgt_slot"]


def run_one_checkpoint(name, ckpt_path, device, n,
                       lambda_t_grid=(0.0, 0.1, 1.0, 3.0), lr_grid=(3e-4, 1e-3), n_steps=8000):
    pools, cfg, pool_report = R.build_grammar_pools_and_cfg(seed=0)
    pools = pools.to(device)
    ckpt = R.load_checkpoint(ckpt_path, device)
    assert ckpt is not None, f"checkpoint not found/invalid at {ckpt_path} -- run the A1 pre-flight `ls` step first"
    # R5 F3 repair: freeze_entity_adapter is PER-CHECKPOINT (True for
    # primary/compA, False for compB -- recorded in each config and in the
    # checkpoint itself); read it off the checkpoint, never hardcode. The
    # battery's own pbe_repl took `freeze` as argv[3] for exactly this
    # reason. Printed per A1's extended pre-flight.
    fe = bool(ckpt.get("freeze_entity_adapter", False))
    print(f"[pre-flight A1+] {name}: freeze_entity_adapter={fe} (read from checkpoint)")
    arms, _, _ = R.restore_arms_and_opts(ckpt, pool_report["vocab_size_total"], lr=3e-4,
                                          device=device, freeze_entity_adapter=fe)
    integ = arms["full_graft"]["integ"]
    backbone = arms["full_graft"]["backbone"]
    ncr_head = arms["full_graft"]["ncr"]

    Z_sgd, keys_v, values_v, probe = extract_real_kv(arms, pools, cfg, device, n=n)
    tf_fn = integ.teacher_force_operator

    keygeom = item_1_2_keygeom(tf_fn, keys_v, values_v)
    # A2: ncr_head.encoder.row_out, NOT ncr_head.row_out (NCREarlyLNModel(nm.NCRModel)'s
    # own __init__ sets self.encoder = BindingEncoder(...); row_out lives on the encoder --
    # verified against ncr_models.py:165 and chapter2/model_v4.py:52).
    item3 = item_3_reachability(ncr_head.encoder, keygeom)
    item4 = item_4_transverse(Z_sgd, keys_v)
    item5 = item_5_ortho_conflict(tf_fn, keys_v, values_v)

    result = dict(item12=keygeom, item3=item3, item4=item4, item5=item5, item6=None)

    if name == "compB":   # A6: item 6 stays compB-only (unchanged scope, only the METHOD changed)
        def score_fn(o, entity_ids, tgt_slot):
            return R.discriminability_metrics(integ, backbone.embed, o, entity_ids, tgt_slot)

        # F3.5, closed at 1x training cost (see item_6_achievability_probe's
        # own docstring for the disclosed scoping decision): train ONCE per
        # (lr, lambda_t) grid cell on the h=61-extracted training episodes
        # (keys_v/values_v content is not itself hop-conditioned -- only the
        # query/target construction is), then score EACH trained cell
        # against hop-MATCHED held-out documents at every h in {1, 61} --
        # never an h=61-fit Z scored against a DIFFERENT hop's own document
        # (F3.5's own diagnosed defect).
        held_out_by_hop = {h: extract_held_out(arms, pools, cfg, device, n=n, seed=BASE_SEED, hop=h) for h in (1, 61)}
        result["item6"] = item_6_achievability_probe(
            keys_v, values_v, held_out_by_hop, score_fn,
            lambda_t_grid=lambda_t_grid, lr_grid=lr_grid, n_steps=n_steps)

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-only", action="store_true")
    # R5 M1 repair: the timing probe (and any grid restriction) must be
    # runnable WITHOUT editing the deployed copy.
    ap.add_argument("--n-steps", type=int, default=8000)
    ap.add_argument("--lr-grid", type=str, default="3e-4,1e-3")
    ap.add_argument("--lambda-t-grid", type=str, default="0.0,0.1,1.0,3.0")
    ap.add_argument("--timing-probe", action="store_true",
                    help="1 cell (first lr x first lambda_t) x 500 steps, write JSON, exit")
    args = ap.parse_args()
    n = 8 if args.smoke_only else N_EPISODES
    lr_grid = tuple(float(x) for x in args.lr_grid.split(","))
    lam_grid = tuple(float(x) for x in args.lambda_t_grid.split(","))
    n_steps = args.n_steps
    if args.timing_probe:
        lr_grid, lam_grid, n_steps = lr_grid[:1], lam_grid[:1], 500
    os.makedirs(OUT_DIR, exist_ok=True)   # A8: was never created in the un-amended card
    t0 = time.time()
    out = {}
    for name, ckpt_path in CKPTS.items():
        out[name] = run_one_checkpoint(name, ckpt_path, args.device, n,
                                       lambda_t_grid=lam_grid, lr_grid=lr_grid, n_steps=n_steps)
    suffix = "_timing" if args.timing_probe else ("_smoke" if args.smoke_only else "")
    with open(os.path.join(OUT_DIR, f"stage0prime{suffix}.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    elapsed = time.time() - t0
    if elapsed > CEILING_S:
        print(f"WARNING: exceeded {CEILING_S}s wall-clock ceiling ({elapsed:.0f}s) -- "
              f"this is a WARNING only, printed AFTER the job finished; the real ceiling "
              f"is the `timeout 2400` wrapper on the launch command (m3, A8)", file=sys.stderr)
    print(f"stage0prime DONE in {elapsed:.0f}s -> {OUT_DIR}/stage0prime.json")
