"""phase2_smoke_gpu.py -- REAL-KERNEL (fla/Triton/CUDA) smoke gate for the
Phase-2 familiarization PRODUCTION path (deploy decision, 2026-07-07).

**`--arm` extension (sec 16.16.9, MAJOR-5 -- Phase-2b build round).** Rev 0
of this gate hardcoded `SMOKE_ARM = "off"` (no CLI flag) -- the real
fla/Triton kernel path had NEVER exercised `apply_frozen_bias_blend`
(`lm_pretrain_rd.py` L854-857, fires UNCONDITIONALLY whenever
`frozen_bias_arm != "off"`), the exact code path all 12 of Phase-2b's NEW
cells run. `per_token` and `global` exercise DIFFERENT tensor operations
inside that blend (a per-token gather-lookup vs. a broadcast add, not two
branches of one op) -- this program has direct precedent for a kernel bug
that was branch-and-device-specific (sec 16.15.7 item 2's own Triton
`Pointer argument cannot be accessed from Triton` failure, invisible to
the CPU-stub suite). `--arm {off,per_token,global}` threads `SMOKE_ARM`
through checkpoint selection (init-checkpoint path resolution now mirrors
`phase2_chain.sh`'s own `run_cells_2way` lam_tag templating for non-off
arms) and model config; `phase2b_chain.sh` runs this FULL positive suite
THREE TIMES, once per arm, ALL THREE required to pass before the 12-cell
launch (never merely one non-off arm as a stand-in for both). Items 4/5
(the end-to-end `run_familiarization_cell` + resume check) assert the
Stage-0.5 gate fields ONLY for `arm="off"` (the gate is OFF-arm-only by
registered design, sec 16.2.1 `compute_stage05_gate`'s own
`assert arm == "off"`) -- for `per_token`/`global` those items instead
assert the gate is correctly `None`, mirroring
`phase2_stage_minus1.test_item_8`'s own per_token/global sub-check.

Why this exists: the Stage -1 self-test suites (phase2_stage_minus1.py,
reasoning_link_stage_minus1.py) are CPU-stub-only BY DESIGN (`device="cpu"`
hardcoded throughout; their own docstrings disclose it), and fla 0.5.1's
RMSNorm has NO CPU fallback -- forcing those suites onto real kernels crashes
in Triton (`Pointer argument cannot be accessed from Triton`, verified on-box
2026-07-07 pre-deploy, logs/predeploy_*_real.log). Net effect: before this
gate, NO Stage -1 item had ever exercised the real fla/Triton kernel path the
18 registered cells actually train on. Per the deploy decision this gate does
NOT patch those suites (they test LOGIC, correctly, under the stub) -- it
closes the KERNEL coverage gap on the production path itself:

  1. `load_init_checkpoint_strict` on a REAL archived Leg-A frozen-bias
     step-20,000 checkpoint (off arm) -- the exact loader + checkpoint every
     cell launch uses.
  2. ONE direct `measure_cell_all_h` call (K=20, Q=K gate episode config,
     `use_heldout_entities=True`, premises + null computed) on real kernels --
     asserting finite recovered_frac / cos / premise stats and well-formed
     gate booleans.
  3. A fine-grained ~20-step training loop built from the SAME production
     building blocks `run_familiarization_cell` calls per step (`get_batch` ->
     `model(...)` -> `F.cross_entropy` -> `query_loss_forward` -> `backward`
     -> `clip_grad_norm_` -> `opt.step()`), asserting BOTH losses and the
     grad-norm finite at EVERY step. (Disclosed: this loop mirrors
     `run_familiarization_cell`'s per-step body rather than calling it,
     because that function only logs trajectory entries at step 1 and every
     50th step -- it cannot provide per-step finiteness assertions at smoke
     scale. The end-to-end call in item 4 covers the function itself.)
  4. `run_familiarization_cell` END-TO-END (steps=12, ckpt_steps=[10], off
     arm): asserts exactly ONE trajectory checkpoint written, its .pt
     round-trip-loads with `optimizer_state_dict` present at step 10, both
     corpora's val_loss finite, and the off-arm Stage-0.5 gate JSON written
     with gate_q == gate_k (Q=K, MAJOR-2 fix) and finite premise stats.
  5. RESUME: a second `run_familiarization_cell` call on the same ckpt_dir
     (steps=11, ckpt_steps=[11]) must resume FROM the step-10 checkpoint
     (never restart), run one more step, and checkpoint at step 11 with
     finite val losses.

Negative test (proves this gate has teeth -- house convention: run it, don't
just write it): `PHASE2_SMOKE_FORCE_FAIL=1` poisons one model parameter with
NaN immediately after the strict load; the first finiteness assertion below
must then fail, exiting nonzero. The chain's `set -euo pipefail` turns that
nonzero exit into a mechanical abort before any OFF cell launches.

Run (positive):  CUDA_VISIBLE_DEVICES=0 python phase2_smoke_gpu.py
Run (negative):  PHASE2_SMOKE_FORCE_FAIL=1 CUDA_VISIBLE_DEVICES=0 python phase2_smoke_gpu.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reasoning_link_probe as rlp  # noqa: E402  (would install the CPU stub if fla were missing --
                                     # this gate REFUSES to run in that state, see main())
import grammar_rd  # noqa: E402
import phase2_familiarization_train as pft  # noqa: E402
from lm_pretrain_rd import (  # noqa: E402
    DeltaNetLM, load_corpus, get_batch, get_lr, load_init_checkpoint_strict, set_and_log_tf32,
)

SMOKE_K = 20          # the SMALLER of Leg A's committed K pair -- fastest real-kernel gate episode
FROZEN_BIAS_LAMBDA = 0.58   # sec 5's registered fixed blend weight -- matches phase2_chain.sh's own
                            # FROZEN_BIAS_LAMBDA / lam_tag templating for non-off init checkpoints.


def default_init_checkpoint_for_arm(arm: str, corpus: str, seed: int = 0) -> str:
    """sec 16.16.9 MAJOR-5's own `--arm` extension: mirrors `phase2_chain.sh`'s own `run_cells_2way`
    init-checkpoint path templating EXACTLY (off vs. non-off arms resolve to structurally different
    directory names -- the lam_tag component only exists for arm != 'off')."""
    root = os.environ.get("FROZEN_BIAS_CKPT_ROOT", "/data/deltanet_rd_frozenbias_ckpts")
    if arm == "off":
        cell = f"frozenbias_lm_off_lam0p00_{corpus}_dm256_ds64_L2_s{seed}"
    else:
        lam_tag = f"lam{FROZEN_BIAS_LAMBDA:.2f}".replace(".", "p")
        cell = f"frozenbias_lm_{arm}_{lam_tag}_{corpus}_dm256_ds64_L2_s{seed}"
    return os.path.join(root, cell, f"lmC_{corpus}_dm256_ds64_L2_s{seed}_step20000.pt")


def _assert_finite(name: str, value: float) -> None:
    assert isinstance(value, (int, float)) and math.isfinite(value), (
        f"REAL-KERNEL SMOKE FAILURE: {name}={value!r} is not finite -- the fla/Triton kernel path "
        f"produced a non-finite value on the production math (this is exactly the failure class "
        f"this gate exists to catch before any 5,000-step cell launches)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["off", "per_token", "global"], default="off",
                     help="sec 16.16.9 MAJOR-5: which frozen-bias arm's checkpoint/blend path to "
                          "exercise on real kernels -- phase2b_chain.sh runs this gate three times, "
                          "once per arm, all three required to pass")
    ap.add_argument("--init-checkpoint", default=None,
                     help="defaults to default_init_checkpoint_for_arm(--arm, --corpus) if omitted")
    ap.add_argument("--corpus", default="openr1-mix-ext")
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--finegrained-steps", type=int, default=20)
    args = ap.parse_args()
    if args.init_checkpoint is None:
        args.init_checkpoint = default_init_checkpoint_for_arm(args.arm, args.corpus)
    SMOKE_ARM = args.arm
    t0 = time.time()

    # -- Reality checks: this gate is meaningless under the CPU stub or off-GPU. Refuse loudly. --
    assert not rlp.FLA_STUB_INSTALLED, (
        "the CPU fla stub is installed -- this REAL-KERNEL gate must run with the real fla package "
        "(do not set REASONING_LINK_FORCE_CPU_STUB); a stubbed pass here would be vacuous")
    assert torch.cuda.is_available(), "CUDA is not available -- this gate must run on a GPU"
    assert args.device.startswith("cuda")
    device = args.device
    print(f"[smoke] real fla kernels confirmed (stub_installed=False), device={device}, "
          f"tf32={set_and_log_tf32()}", flush=True)

    # -- 1. Strict load of the REAL archived init checkpoint (the exact production first step). --
    ck_config = torch.load(args.init_checkpoint, map_location=device)["config"]
    model = DeltaNetLM(**ck_config).to(device)
    loaded = load_init_checkpoint_strict(model, args.init_checkpoint, device)
    assert loaded["step"] == 20000, f"expected the archived step-20,000 checkpoint, got step={loaded['step']}"
    print(f"[smoke] load_init_checkpoint_strict OK: {args.init_checkpoint} (archived step 20000, "
          f"config={ck_config})", flush=True)

    if os.environ.get("PHASE2_SMOKE_FORCE_FAIL") == "1":
        # Negative-test branch: poison ONE parameter with NaN. The first finiteness assertion
        # below MUST now fail (nonzero exit) -- proving the abort path is real, not narrated.
        with torch.no_grad():
            next(model.parameters()).mul_(float("nan"))
        print("[smoke] PHASE2_SMOKE_FORCE_FAIL=1: injected NaN into a model parameter -- this run "
              "MUST now exit nonzero at the first finiteness assertion", flush=True)

    tokenizer = grammar_rd.load_gpt2_tokenizer()
    pools, _ = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
    assert pools.vocab_size_total == model.vocab_size
    conv_size = model.conv_size

    # -- 2. Direct measure_cell_all_h on real kernels: Q=K gate config, held-out entities, premises. --
    gate_cfg = pft.familiarization_gate_episode_config(conv_size, SMOKE_K)
    readout_layer = rlp.readout_layer_index(model)
    per_h, forward_counts = rlp.measure_cell_all_h(
        model, gate_cfg, pools, readout_layer, SMOKE_K, hops=(1,), batch_size=4,
        seed=pft.phase2_seed("eval_gate_self", SMOKE_ARM, args.corpus, 0, SMOKE_K, 0),
        surgery="native", device=device, compute_option2=False, compute_premises=True,
        null_seed=pft.phase2_seed("eval_gate_null", SMOKE_ARM, args.corpus, 0, SMOKE_K, 0),
        use_heldout_entities=True)
    h1 = per_h[1]
    for field in ("recovered_frac", "cos_mean", "cos_std", "premise_iii_median",
                  "premise_iii_null_p95", "premise_iv_median", "premise_iv_null_p95",
                  "null_recovered_frac_mean"):
        _assert_finite(f"measure_cell_all_h per_h[1].{field}", h1[field])
    for field in ("premise_iii_pass", "premise_iv_pass", "probe_valid"):
        assert isinstance(h1[field], bool), f"per_h[1].{field} must be a bool, got {type(h1[field])}"
    print(f"[smoke] measure_cell_all_h (K={SMOKE_K}, Q=K, heldout entities) OK on real kernels: "
          f"recovered_frac={h1['recovered_frac']:.4f} cos_mean={h1['cos_mean']:.4f} "
          f"forward_counts={forward_counts}", flush=True)

    # -- 3. Fine-grained per-step finiteness loop (production building blocks, see docstring). --
    train_tokens, _, _, _, _ = load_corpus(args.data_dir, args.corpus, device)
    episode_cfg = pft.familiarization_episode_config(conv_size, SMOKE_K)
    gen_corpus = torch.Generator(device=device).manual_seed(
        pft.phase2_seed("train_corpus", SMOKE_ARM, args.corpus, 0, SMOKE_K, 0))
    gen_episode = torch.Generator(device=device).manual_seed(
        pft.phase2_seed("train_episode", SMOKE_ARM, args.corpus, 0, SMOKE_K, 0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    n = args.finegrained_steps
    for step in range(1, n + 1):
        cur_lr = get_lr(step, 3e-4, 100, n)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = get_batch(train_tokens, 8, 512, gen_corpus)
        logits = model(x, step=step)
        L_corpus = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        L_query, _, _ = pft.query_loss_forward(model, episode_cfg, pools, 4, gen_episode, device,
                                                use_heldout_entities=False, step=step)
        _assert_finite(f"step {step} L_corpus", L_corpus.item())
        _assert_finite(f"step {step} L_query", L_query.item())
        opt.zero_grad()
        (L_corpus + L_query).backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        _assert_finite(f"step {step} grad_norm", total_norm.item())
        opt.step()
    print(f"[smoke] {n}-step real-kernel loop OK: every step's L_corpus, L_query, and grad-norm "
          f"finite (last: L_corpus={L_corpus.item():.4f} L_query={L_query.item():.4f} "
          f"grad_norm={total_norm.item():.4f})", flush=True)
    # Free everything the loop pinned (the last logits tensor alone is ~800MB at vocab 50257)
    # before the end-to-end cell below loads BOTH corpora onto the same GPU.
    del model, opt, train_tokens, logits, x, y, L_corpus, L_query, total_norm
    torch.cuda.empty_cache()

    # -- 4+5. run_familiarization_cell END-TO-END + RESUME, in an isolated throwaway ckpt_dir
    # (NEVER results/phase2/ckpts -- the smoke's run_name would collide with the real cell's own
    # resume scan). Cleaned up in `finally` so a failed run can't poison a rerun's resume scan. --
    td = tempfile.mkdtemp(prefix="phase2_smoke_gpu_")
    try:
        res1 = pft.run_familiarization_cell(
            args.init_checkpoint, SMOKE_ARM, args.corpus, 0, K=SMOKE_K, steps=12, ckpt_steps=[10],
            corpus_batch_size=8, episode_batch_size=4, gate_batch_size=4, seq_len=512,
            eval_batches=1, eval_batch_size=2, data_dir=args.data_dir, device=device,
            ckpt_dir=td, out_path=os.path.join(td, "cell.json"), resume=True)
        assert res1["resumed_from"] is None, "fresh run must not have resumed from anything"
        assert len(res1["checkpoints"]) == 1, (
            f"expected exactly ONE trajectory checkpoint (ckpt_steps=[10]), got "
            f"{[c['step'] for c in res1['checkpoints']]}")
        ck1 = res1["checkpoints"][0]
        assert ck1["step"] == 10
        for corpus_name, v in ck1["val_loss"].items():
            _assert_finite(f"e2e cell val_loss[{corpus_name}]", v)
        payload = torch.load(ck1["checkpoint_path"], map_location="cpu")
        assert payload["step"] == 10 and "optimizer_state_dict" in payload and "model_state_dict" in payload, (
            f"trajectory checkpoint {ck1['checkpoint_path']} is missing required fields "
            f"(keys: {sorted(payload.keys())})")
        gate = ck1["stage05_gate"]
        # sec 16.16.9 MAJOR-5 fix: the Stage-0.5 gate is registered OFF-arm-only
        # (compute_stage05_gate's own `assert arm == "off"`, sec 16.2.1) -- for per_token/global,
        # `gate` must be correctly None (mirrors phase2_stage_minus1.test_item_8's own
        # per_token/global sub-check), never a fabricated/omitted assertion either way.
        if SMOKE_ARM == "off":
            assert gate is not None and isinstance(gate["gate_pass"], bool)
            with open(gate["gate_json_path"]) as f:
                gate_doc = json.load(f)
            assert gate_doc["gate_k"] == gate_doc["gate_q"] == SMOKE_K, (
                f"Stage-0.5 gate must run Q=K (MAJOR-2 fix): gate_k={gate_doc['gate_k']} "
                f"gate_q={gate_doc['gate_q']}")
            for field in ("recovered_frac", "cos_mean", "premise_iii_median", "premise_iv_median",
                          "null_recovered_frac_mean"):
                _assert_finite(f"stage05 gate per_h.1.{field}", gate_doc["per_h"]["1"][field])
            print(f"[smoke] run_familiarization_cell end-to-end OK: 1 checkpoint at step 10 "
                  f"(optimizer_state_dict present), val_loss finite, stage05 gate Q=K={SMOKE_K} with "
                  f"finite premise stats (gate_pass={gate['gate_pass']} -- its VALUE is box-truth, "
                  f"only well-formedness is asserted here)", flush=True)
        else:
            assert gate is None, (
                f"arm={SMOKE_ARM!r}: the OFF-arm-only Stage-0.5 gate must be None, got {gate!r}")
            print(f"[smoke] run_familiarization_cell end-to-end OK: 1 checkpoint at step 10 "
                  f"(optimizer_state_dict present), val_loss finite, stage05_gate correctly None "
                  f"(arm={SMOKE_ARM!r} is not OFF-arm-only)", flush=True)

        res2 = pft.run_familiarization_cell(
            args.init_checkpoint, SMOKE_ARM, args.corpus, 0, K=SMOKE_K, steps=11, ckpt_steps=[11],
            corpus_batch_size=8, episode_batch_size=4, gate_batch_size=4, seq_len=512,
            eval_batches=1, eval_batch_size=2, data_dir=args.data_dir, device=device,
            ckpt_dir=td, out_path=os.path.join(td, "cell_resume.json"), resume=True)
        assert res2["resumed_from"] == ck1["checkpoint_path"], (
            f"resume must pick up the step-10 checkpoint, got resumed_from={res2['resumed_from']}")
        assert len(res2["checkpoints"]) == 1 and res2["checkpoints"][0]["step"] == 11
        for corpus_name, v in res2["checkpoints"][0]["val_loss"].items():
            _assert_finite(f"resumed cell val_loss[{corpus_name}]", v)
        print(f"[smoke] resume OK: picked up {os.path.basename(ck1['checkpoint_path'])}, ran 1 more "
              f"step, checkpointed at step 11 with finite val losses", flush=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)

    print(f"PHASE2_SMOKE_GPU: ALL REAL-KERNEL CHECKS PASSED in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
