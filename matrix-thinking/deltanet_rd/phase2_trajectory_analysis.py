"""phase2_trajectory_analysis.py -- REASONING_LINK_DESIGN.md sec 16.2.1's
"Trajectory readout schedule" + the hexachotomy classification, wired end
to end: for a given corpus, re-scores EVERY trajectory checkpoint of EVERY
familiarized cell (3 arms x 3 ckpt_seeds x 2 K's = 18 killer-prediction
readouts PER checkpoint, 90 per corpus, 180 total across both corpora --
sec 16.2.3's own priced "5 checkpoints x 18 cells x 2 K's" trajectory-
readout line) via `reasoning_link_probe.run_cell(..., surgery="off")`
(sec 16.2.1's own explicit reuse instruction: "reasoning_link_probe.py's
frozen_bias_surgery is the reference pattern"), pools the 3 ckpt_seeds into
a `delta_ci_n3` 3-seed CI per (arm, checkpoint) -- the SAME formula sec
5.3's own killer-prediction condition already uses, reused verbatim, never
reimplemented -- computes `det`/`holds`/`agree` via `phase2_hexachotomy`'s
own primitives, reads the OFF-arm's own per-checkpoint Stage-0.5 gate
verdict via `phase2_gate_enforce.gate_verdict` (never recomputed), and
classifies the resulting trajectory via `phase2_hexachotomy.classify_
trajectory`.

**Scoping decision, disclosed explicitly (this BUILD's own resolution of a
genuinely underspecified point, mirroring reasoning_link_probe.py's own
"resolved during BUILD" convention -- flagged for the independent audit's
attention, not silently assumed):** sec 16.2.1's own prose alternates
between "classify each (K, corpus, seed) trajectory" and, two paragraphs
later, "a cell whose 3 SEEDS classify {PERSISTENT, NON-MONOTONE,
TRANSIENT} reports all three labels individually" -- the second sentence
only parses if "3 seeds" are three INPUTS to ONE classification, not three
SEPARATE per-seed classifications (a lone seed has no natural multi-sample
CI to compute `det`/`holds` from at all). The worked totality table (sec
16.2.1) also operates on ONE `holds(c)` pattern per row, matching a SINGLE
classification per (corpus) -- `holds(c)` itself already folds BOTH K's
into one boolean (`det(32,c) AND NOT det(20,c) AND ...`), so there is no
separate per-K classification either. This module therefore classifies
ONE trajectory PER CORPUS (2 total: openr1-mix-ext, wikitext-mix-ext),
each built from a 3-seed-pooled `delta_ci_n3` CI at every checkpoint, for
BOTH arms (global, per_token) -- consistent with every mechanical
primitive's own literal definition in sec 16.2.1, and with the totality
table's own single-holds()-pattern-per-row structure.

Run (per corpus, after both OFF and the two intervention arms' 6 cells
each have completed and their .pt checkpoints + stage05_gate JSONs exist
in `--ckpt-dir`):
    python phase2_trajectory_analysis.py --corpus openr1-mix-ext \\
        --ckpt-dir results/phase2/ckpts --out results/phase2/traj_openr1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reasoning_link_probe as rlp  # noqa: E402  (installs the CPU fla stub as an import side effect)
import phase2_familiarization_train as pft  # noqa: E402
import phase2_gate_enforce as pge  # noqa: E402
import phase2_hexachotomy as phx  # noqa: E402

CKPT_SEEDS = (0, 1, 2)
ARMS_NON_OFF = ("global", "per_token")


def ckpt_path_for(ckpt_dir: str, arm: str, corpus: str, ckpt_seed: int, checkpoint_step: int) -> str:
    """NO K in this path (build-time correction, sec 16.2.3's own cost arithmetic: 18 training
    cells, not 36 -- see phase2_familiarization_train.K_TRAIN_DEFAULT's own comment). The SAME
    checkpoint file is re-scored at BOTH K=20 and K=32 by `killer_prediction_readout` below --
    `reasoning_link_probe.run_cell`'s own `K` parameter controls EVAL episode construction only,
    independent of what K the checkpoint trained under."""
    return os.path.join(ckpt_dir, f"phase2fam_{arm}_{corpus}_s{ckpt_seed}_step{checkpoint_step}.pt")


def gate_json_path_for(ckpt_dir: str, corpus: str, ckpt_seed: int, checkpoint_step: int) -> str:
    run_name = f"phase2fam_off_{corpus}_s{ckpt_seed}"
    return os.path.join(ckpt_dir, f"stage05_gate_{run_name}_step{checkpoint_step}.json")


def killer_prediction_readout(ckpt_dir: str, arm: str, corpus: str, ckpt_seed: int, K: int,
                               checkpoint_step: int, batch_size: int = 16, device: str = "cpu") -> dict:
    """ONE (arm,corpus,ckpt_seed,K,checkpoint_step) readout: `run_cell` with
    blend-OFF surgery on the FAMILIARIZED checkpoint at this trajectory
    step (sec 16.2.1's own registered reuse of the audited machinery,
    verbatim -- no new scoring code). `K` here governs ONLY how `run_cell`
    constructs its own EVAL episode (the checkpoint itself carries no K).

    MAJOR-1 fix (Phase-2 build-audit round): `use_heldout_entities=True` --
    this is the wave's central arm-contrast measurement, so its EVAL
    episodes must draw from `pools.heldout_name_ids`, disjoint from the
    `pools.train_name_ids` familiarization training itself drew from
    (`phase2_familiarization_train.query_loss_forward`'s own
    `use_heldout_entities=False`) -- never the same entities re-tested."""
    ckpt_path = ckpt_path_for(ckpt_dir, arm, corpus, ckpt_seed, checkpoint_step)
    seed = pft.phase2_seed("eval_killer", arm, corpus, ckpt_seed, K, checkpoint_step)
    return rlp.run_cell(ckpt_path, K, hops=(1, 2, 3, 4), surgery="off", batch_size=batch_size,
                         device=device, compute_option2=False, seed_override=seed,
                         use_heldout_entities=True)


def build_holds_and_gate_by_checkpoint(ckpt_dir: str, arm: str, corpus: str, K_pair=(32, 20),
                                        h: int = 1, batch_size: int = 16, device: str = "cpu") -> dict:
    """For ONE (arm, corpus): at each of the 5 trajectory checkpoints, pools the 3 ckpt_seeds'
    own recovered_frac(h) readings (via killer_prediction_readout, arm AND off) into a 3-seed
    delta_ci_n3 CI at BOTH K's, then computes det(32,c)/det(20,c)/holds(c) via phase2_hexachotomy's
    own primitives. Also reads the OFF arm's own per-checkpoint Stage-0.5 gate verdict (NEVER
    recomputed -- phase2_gate_enforce.gate_verdict on the JSON phase2_familiarization_train.py
    already wrote). Returns {"holds_by_c": {...}, "stage05_pass_by_c": {...}, "det_arm_by_c": {...},
    "raw": {...}} (raw = every intermediate CI, for disclosure/debugging)."""
    K32, K20 = K_pair
    holds_by_c, stage05_pass_by_c, det_arm_by_c, raw = {}, {}, {}, {}
    for c in phx.CHECKPOINTS:
        per_k_delta = {}
        for K in (K32, K20):
            off_vals, arm_vals = [], []
            for s in CKPT_SEEDS:
                off_r = killer_prediction_readout(ckpt_dir, "off", corpus, s, K, c, batch_size, device)
                arm_r = killer_prediction_readout(ckpt_dir, arm, corpus, s, K, c, batch_size, device)
                off_vals.append(off_r["per_h"][h]["recovered_frac"])
                arm_vals.append(arm_r["per_h"][h]["recovered_frac"])
            per_k_delta[K] = rlp.delta_ci_n3(arm_vals, off_vals)   # arm - off, sec 5.2a's own convention
        det32 = phx.det(per_k_delta[K32]["ci_low"], per_k_delta[K32]["ci_high"])
        det20 = phx.det(per_k_delta[K20]["ci_low"], per_k_delta[K20]["ci_high"])
        holds_by_c[c] = phx.holds(det32, det20, abs(per_k_delta[K32]["mean"]), abs(per_k_delta[K20]["mean"]))
        det_arm_by_c[c] = det32   # det_arm(arm,c) reuses the SAME K=32 delta (sec 16.2.1's own citation)
        raw[c] = {"delta_k32": per_k_delta[K32], "delta_k20": per_k_delta[K20], "det32": det32, "det20": det20}

        gate_path = gate_json_path_for(ckpt_dir, corpus, 0, c)   # the OFF arm's own seed=0 gate at this
                                                                   # checkpoint gates this corpus's
                                                                   # reading (seed=0 is this module's
                                                                   # own build-time pin, disclosed here)
        if os.path.exists(gate_path):
            with open(gate_path) as f:
                gate_pass, _ = pge.gate_verdict(json.load(f))
        else:
            gate_pass = False  # fail-closed: no gate JSON means uninterpretable, never silently assumed OK
        stage05_pass_by_c[c] = gate_pass
    return {"holds_by_c": holds_by_c, "stage05_pass_by_c": stage05_pass_by_c,
            "det_arm_by_c": det_arm_by_c, "raw": raw}


def analyze_corpus(ckpt_dir: str, corpus: str, batch_size: int = 16, device: str = "cpu") -> dict:
    per_arm = {arm: build_holds_and_gate_by_checkpoint(ckpt_dir, arm, corpus, batch_size=batch_size,
                                                          device=device)
               for arm in ARMS_NON_OFF}
    # agree(c): global's vs per_token's own Delta(K=32,c) CIs overlap.
    agree_by_c = {}
    for c in phx.CHECKPOINTS:
        g = per_arm["global"]["raw"][c]["delta_k32"]
        p = per_arm["per_token"]["raw"][c]["delta_k32"]
        agree_by_c[c] = phx.agree(g["ci_low"], g["ci_high"], p["ci_low"], p["ci_high"])

    # holds(c) at the CORPUS level is the SAME condition regardless of which non-off arm is being
    # read (sec 16.2.1's own "holds(c) -- the full killer-prediction pass condition ... exactly as
    # the K-sweep paragraph already states it" -- ONE condition per checkpoint, not one per arm).
    # Build-time resolution (disclosed): use the GLOBAL arm's own holds_by_c as the trajectory's
    # holds() input (sec 16.2.1's H_LINK-A causal claim is itself global-arm-centric: "global arm
    # learns to compose at least as well as off, per-token no better").
    classification = phx.classify_trajectory(
        holds_by_c=per_arm["global"]["holds_by_c"],
        stage05_pass_by_c=per_arm["global"]["stage05_pass_by_c"],
        det_arm_global_5000=per_arm["global"]["det_arm_by_c"][phx.TERMINAL_CHECKPOINT],
        det_arm_per_token_5000=per_arm["per_token"]["det_arm_by_c"][phx.TERMINAL_CHECKPOINT],
        agree_5000=agree_by_c[phx.TERMINAL_CHECKPOINT])

    return {"corpus": corpus, "per_arm": per_arm, "agree_by_c": agree_by_c, "classification": classification}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=["openr1-mix-ext", "wikitext-mix-ext"], required=True)
    ap.add_argument("--ckpt-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    result = analyze_corpus(args.ckpt_dir, args.corpus, batch_size=args.batch_size, device=args.device)
    print(json.dumps({"corpus": result["corpus"], "classification": result["classification"],
                       "agree_by_c": result["agree_by_c"]}, indent=2, default=str))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
