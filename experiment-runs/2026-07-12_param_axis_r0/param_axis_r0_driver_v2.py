#!/usr/bin/env python3
"""R0 driver v2 -- multi-rung/multi-corpus orchestration of the sec 9 PINNED
instrument (lm_recall_gap_probe_v2_rd.py), which mode_run's own refusal text
says "is not yet built." This is that driver.

Runs DiD/T1a (run_did_eval + finalize_cell) AND T2b (run_t2_planted_copy +
check_t2b1_mechanism_exists + check_t2b2_ceiling) for every (rung, corpus)
cell, then sec 9.6 admissibility (check_rung_admissible) and the sec 9.5
verdict machinery (bootstrap_trend_ci / tost_flat / classify_recall_trend /
compute_verdict_with_s2), applying sec 9.1's pinned raw_did normalization via
compute_capacity_metric.

ADMISSIBILITY DECISIONS MADE BEFORE THIS RUNS (see the R0 report for the
full reasoning, PARAM_AXIS_SCALING_DESIGN.md sec 10):
  - 1.31B rung EXCLUDED ENTIRELY. The only quiesced 1.31B checkpoint on the
    box (/data/lm_rd_trackc_ckpts/wave3, step 155000, harvested 2026-07-07 per
    SCALE_TRANSFER_DESIGN.md sec 5.11) is a DIFFERENT ARM -- plain DeltaNetLM,
    frozen_bias_arm=None, use_geo3_lm=False -- than the per_token/lambda=0.58
    frozen-bias arm used by the 14M/98M/392M rungs. The CORRECT-arm 1.31B
    checkpoints (/data/queue_1p31b_ckpts/*) are LIVE training jobs (pid
    1860400 on ckpt-dir ..._s0, pid 1036283 on ..._s0_pricefix) and are
    therefore inadmissible under sec 9.6 item 3 regardless. Running the
    wrong-arm checkpoint would bundle a second, unproven architectural axis
    onto the param axis (CLAUDE.md's hold-the-second-axis-fixed rule) --
    EXCLUDED, not fudged, per this task's explicit instruction.
  - 392M rung: admissible-SHAPED (correct arm, quiesced, both corpora present
    at step 20000) but tok_per_param at the sec 9.6-forced 0.328B common slice
    = 327,680,000 / 391,869,440 = 0.8362 < 1.0 -- BELOW the sec 9.6 item 2
    primary-fit floor. Run anyway; reported as a disclosed secondary point,
    excluded from the primary fit and from the admissible set A, per sec 9.6's
    own text ("Rungs below that are reported as disclosed secondary points
    that do not enter the fit").
Only 14M and 98M can therefore ever enter A -- 2 < 3, the sec 9.6 minimum-n
floor for ANY trend verdict. This is known BEFORE any GPU cell is read; the
run below still executes fully (full field set, both corpora, real T1a/T2b)
because the per-rung data is independently required and reusable, and because
compute_verdict's FLOOR branch (n_admissible_rungs < 3) is only correctly
producible by actually running check_rung_admissible on real cells.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

import torch  # noqa: E402

from lm_recall_gap_probe_v2_rd import (  # noqa: E402
    load_checkpoint, build_bigram_mode_table, run_did_eval, finalize_cell,
    resolve_token_matched_checkpoint, run_t2_planted_copy, pick_t2_marker_tokens,
    check_t2b1_mechanism_exists, check_t2b2_ceiling, check_rung_admissible,
    compute_capacity_metric, compute_s1_utilization, bootstrap_trend_ci,
    tost_flat, classify_recall_trend, compute_verdict_with_s2,
    VOCAB_SIZE, TARGET_TOKENS_FORCED_SLICE, TOK_PER_PARAM_PRIMARY_FIT_FLOOR,
)
from lm_pretrain_rd import load_corpus, corpus_fixed_seed, DEFAULT_DATA_DIR  # noqa: E402

DEVICE = "cuda"
CORPORA = ["openr1-mix-ext", "wikitext-mix-ext"]
N_WINDOWS = 2048          # rung-independent, per sec 9.2 (F-4 fix)
C_MAX = 8
EVAL_MICRO_BATCH = 64
N_PLANTS_T2B = 512

RUNGS = [
    {
        "rung": "14M", "batch_size": 32,
        "ckpt_dir": lambda c: f"/data/deltanet_rd_frozenbias_ckpts/frozenbias_lm_per_token_lam0p58_{c}_dm256_ds64_L2_s0",
        "prefix": lambda c: f"lmC_{c}_dm256_ds64_L2_s0",
    },
    {
        "rung": "98M", "batch_size": 32,
        "ckpt_dir": lambda c: f"/data/fixscale_ckpts/train/fixscale_train_arm_per_token_98m_{c}_s0",
        "prefix": lambda c: f"lmC_{c}_dm768_ds64_L12_s0",
    },
    {
        "rung": "392M", "batch_size": 32,
        "ckpt_dir": lambda c: f"/data/fixscale_ckpts/train/fixscale_train_arm_per_token_392m_{c}_s0",
        "prefix": lambda c: f"lmC_{c}_dm1536_ds128_L16_s0",
    },
]

OUT_DIR = "/home/nvidia/chapter2/deltanet_rd/results/param_axis_r0"
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    corpus_cache = {}
    all_cells = {}          # (rung, corpus) -> cell dict
    t2b1_results = {}       # (rung, corpus) -> t2b1 dict
    t2b2_results = {}       # (rung, corpus) -> t2b2 dict
    s1_results = {}         # (rung, corpus) -> s1 dict
    n_params_by_rung = {}
    slice_info_by_cell = {}
    all_deltas_pooled = []   # sec 9.4's "the main metric's own empirical Delta distribution",
                              # pooled across every (rung, corpus) cell's resolved candidates --
                              # consumed by the separate T2a reference-model driver so its plant
                              # distances are drawn from the SAME empirical distribution ours are.

    for spec in RUNGS:
        rung = spec["rung"]
        for corpus in CORPORA:
            t_cell0 = time.time()
            ckpt_dir = spec["ckpt_dir"](corpus)
            prefix = spec["prefix"](corpus)
            slice_info = resolve_token_matched_checkpoint(
                ckpt_dir, spec["batch_size"], 512, TARGET_TOKENS_FORCED_SLICE, prefix)
            assert "error" not in slice_info, f"{rung}/{corpus}: {slice_info}"
            path = slice_info["path"]
            print(f"=== {rung} / {corpus} : step {slice_info['chosen_step']} "
                  f"(miss_tokens={slice_info['miss_tokens']}) ===", flush=True)

            model, ckpt, md5_pin = load_checkpoint(path, DEVICE, require_quiesced=True, wait_s=5.0)
            n_params = sum(p.numel() for p in model.parameters())
            n_params_by_rung[rung] = n_params

            if corpus not in corpus_cache:
                train_tokens, val_tokens, meta, _, _ = load_corpus(DEFAULT_DATA_DIR, corpus, DEVICE)
                mode_next = build_bigram_mode_table(train_tokens, VOCAB_SIZE, DEVICE)
                corpus_cache[corpus] = (train_tokens, val_tokens, mode_next)
            train_tokens, val_tokens, mode_next = corpus_cache[corpus]

            seed = corpus_fixed_seed(corpus) + 424242
            did_result = run_did_eval(model, val_tokens, spec["batch_size"], 512, N_WINDOWS, DEVICE,
                                       mode_next, seed, c_max=C_MAX, eval_micro_batch=EVAL_MICRO_BATCH,
                                       min_sep=2)

            # --- T2b: planted-copy positive control at the MAIN METRIC'S OWN empirical
            # Delta distribution (sec 9.4), computed from THIS cell's own resolved candidates.
            delta_pool = [r["delta"] for r in did_result["records"] if r["delta"] is not None]
            if not delta_pool:
                delta_pool = [20]  # degenerate fallback, flagged below via n_candidates
            all_deltas_pooled.extend(delta_pool)
            tok_a, tok_b = pick_t2_marker_tokens(train_tokens, VOCAB_SIZE, DEVICE)
            t2_seed = seed + 777
            t2_result = run_t2_planted_copy(model, val_tokens, spec["batch_size"], 512, DEVICE,
                                             t2_seed, delta_pool, tok_a, tok_b, n_plants=N_PLANTS_T2B,
                                             vocab_size=VOCAB_SIZE, eval_micro_batch=EVAL_MICRO_BATCH)

            cell = finalize_cell(rung, corpus, did_result, t2b_result=t2_result)
            cell["checkpoint"] = path
            cell["checkpoint_md5"] = md5_pin
            cell["n_params"] = n_params
            cell["ckpt_step"] = ckpt.get("step")
            cell["job_terminated_attested"] = True  # these three rungs' training completed days
                                                      # ago (result JSONs: complete=true / timed_out
                                                      # false at final_step==steps; no matching
                                                      # process in `ps aux`) -- attested, not a live
                                                      # --ckpt-every writer (sec 9.6 item 3, 2nd clause)
            cell["quiesce_check_skipped"] = False
            cell["sec91_normalization_pinned"] = True
            cell["sec91_normalization_name"] = "raw_did"
            cell["tok_per_param_common_slice"] = TARGET_TOKENS_FORCED_SLICE / n_params
            cell["frozen_bias_arm"] = ckpt["config"].get("frozen_bias_arm") if "config" in ckpt else None
            cell["frozen_bias_lambda"] = ckpt["config"].get("frozen_bias_lambda") if "config" in ckpt else None

            t2b1 = check_t2b1_mechanism_exists(t2_result["records"])
            t2b2 = check_t2b2_ceiling(cell["did"], t2_result["acc_copy"] or 0.0, t2_result["acc_copy_se"] or 0.0)
            s1 = compute_s1_utilization(cell)

            all_cells[(rung, corpus)] = cell
            t2b1_results[(rung, corpus)] = t2b1
            t2b2_results[(rung, corpus)] = t2b2
            s1_results[(rung, corpus)] = s1
            slice_info_by_cell[(rung, corpus)] = slice_info

            elapsed = time.time() - t_cell0
            print(f"    n_params={n_params} tok_per_param={cell['tok_per_param_common_slice']:.4f} "
                  f"did={cell['did']:.4f} ci={cell['did_ci']} t1a={cell['t1a_pass_did_ci_excludes_zero']} "
                  f"acc_copy={t2_result['acc_copy']} t2b1_p={t2b1['p_value']:.4g} t2b1_pass={t2b1['passes']} "
                  f"t2b2_pass={t2b2['passes']} n_res={cell['n_candidates_resolved']} "
                  f"elapsed={elapsed:.1f}s", flush=True)

            # free VRAM before next checkpoint
            del model
            torch.cuda.empty_cache()

    # ---- sec 9.6 admissibility, per rung, across BOTH corpora ----
    rung_admissibility = {}
    for spec in RUNGS:
        rung = spec["rung"]
        cell_by_corpus = {c: all_cells[(rung, c)] for c in CORPORA}
        t2b1_by_corpus = {c: t2b1_results[(rung, c)] for c in CORPORA}
        t2b2_by_corpus = {c: t2b2_results[(rung, c)] for c in CORPORA}
        tok_per_param = TARGET_TOKENS_FORCED_SLICE / n_params_by_rung[rung]
        adm = check_rung_admissible(cell_by_corpus, tok_per_param, checkpoint_quiesced=True,
                                     t2b2_by_corpus=t2b2_by_corpus, t2b1_by_corpus=t2b1_by_corpus,
                                     min_candidates=4096)
        rung_admissibility[rung] = adm
        print(f"--- {rung} sec 9.6 admissibility: {adm}", flush=True)

    admissible_rungs = [r for r in rung_admissibility if rung_admissibility[r]["admissible"]]
    n_admissible_rungs = len(admissible_rungs)
    # t1a_positive_count: rungs T2b-1-passing AND T1a-positive on BOTH corpora (independent
    # of the tok/param + corpus-availability + sample-size gates folded into "admissible").
    t1a_positive_rungs = []
    for spec in RUNGS:
        rung = spec["rung"]
        ok = all(
            all_cells[(rung, c)].get("t1a_pass_did_ci_excludes_zero", False)
            and t2b1_results[(rung, c)].get("passes", False)
            for c in CORPORA
        )
        if ok:
            t1a_positive_rungs.append(rung)
    t1a_positive_count = len(t1a_positive_rungs)

    print(f"\nADMISSIBLE RUNGS (A): {admissible_rungs} (n={n_admissible_rungs})", flush=True)
    print(f"T1a+T2b-1-positive rungs (both corpora): {t1a_positive_rungs} (n={t1a_positive_count})",
          flush=True)

    # ---- trend fit, only meaningful if n_admissible_rungs >= 3; compute anyway for
    # disclosure (bootstrap_trend_ci/tost_flat/classify_recall_trend do not themselves
    # enforce the minimum-n gate -- that enforcement lives in compute_verdict). ----
    trend_result = None
    trend_result_s2 = None
    if n_admissible_rungs >= 2:  # need >=2 points for OLS to be non-degenerate at all
        per_rung_row_values = {}
        per_rung_row_values_s2 = {}
        log10_params = {}
        import math as _math
        for r in admissible_rungs:
            # M(r) per row = raw_did normalization applied per-row; pool over BOTH corpora's
            # per-row dicts (disclosed choice -- sec 9 does not pin how corpora combine into
            # one trend point; see the R0 report's flagged judgment call).
            pooled = {}
            pooled_s2 = {}
            for c in CORPORA:
                cell = all_cells[(r, c)]
                for row_idx, val in cell["per_row_did"].items():
                    pooled[f"{c}:{row_idx}"] = val
                for row_idx, val in cell["per_row_did_logp"].items():
                    pooled_s2[f"{c}:{row_idx}"] = val
            per_rung_row_values[r] = pooled
            per_rung_row_values_s2[r] = pooled_s2
            log10_params[r] = _math.log10(n_params_by_rung[r])
        trend_result = bootstrap_trend_ci(per_rung_row_values, log10_params)
        trend_result_s2 = bootstrap_trend_ci(per_rung_row_values_s2, log10_params)

    verdict = None
    factor1_primary = None
    factor1_s2 = None
    tost_primary = None
    tost_s2 = None
    if trend_result is not None:
        # delta = 0.125 * M(r_min) per decade (sec 9.5) -- M(r_min) = the smallest-params
        # admissible rung's OWN point DiD (not pooled row mean -- "M(r_min)" is the metric
        # value AT that rung).
        r_min = min(admissible_rungs, key=lambda r: n_params_by_rung[r])
        m_r_min = all_cells[(r_min, CORPORA[0])]["did"]  # sec 9 does not pin cross-corpus
        # combination for delta either; using corpus[0] as a disclosed, documented choice.
        delta = 0.125 * abs(m_r_min)
        tost_primary = tost_flat(trend_result["boots"], delta)
        tost_s2 = tost_flat(trend_result_s2["boots"], delta)
        factor1_primary = classify_recall_trend(trend_result["ci95"], tost_primary)
        factor1_s2 = classify_recall_trend(trend_result_s2["ci95"], tost_s2)
        verdict = compute_verdict_with_s2(factor1_primary, factor1_s2,
                                           span_frac_monotone_over_A=None,  # T3 not run this session
                                           n_admissible_rungs=n_admissible_rungs,
                                           t1a_positive_count=t1a_positive_count)
    else:
        verdict = "FLOOR" if n_admissible_rungs < 3 else "INDETERMINATE (trend not computed)"

    print(f"\nFactor 1 (primary): {factor1_primary}  Factor 1 (S2): {factor1_s2}")
    print(f"VERDICT (before T2a gate): {verdict}")

    out = {
        "generated_at": time.time(),
        "n_windows": N_WINDOWS, "c_max": C_MAX, "n_plants_t2b": N_PLANTS_T2B,
        "target_tokens_forced_slice": TARGET_TOKENS_FORCED_SLICE,
        "tok_per_param_primary_fit_floor": TOK_PER_PARAM_PRIMARY_FIT_FLOOR,
        "cells": {f"{r}|{c}": {k: v for k, v in all_cells[(r, c)].items()
                                if k not in ("per_row_did", "per_row_did_logp")}
                  for (r, c) in all_cells},
        "per_row_did": {f"{r}|{c}": all_cells[(r, c)]["per_row_did"] for (r, c) in all_cells},
        "per_row_did_logp": {f"{r}|{c}": all_cells[(r, c)]["per_row_did_logp"] for (r, c) in all_cells},
        "t2b1": {f"{r}|{c}": t2b1_results[(r, c)] for (r, c) in t2b1_results},
        "t2b2": {f"{r}|{c}": t2b2_results[(r, c)] for (r, c) in t2b2_results},
        "s1_utilization": {f"{r}|{c}": s1_results[(r, c)] for (r, c) in s1_results},
        "n_params_by_rung": n_params_by_rung,
        "rung_admissibility": rung_admissibility,
        "admissible_rungs": admissible_rungs,
        "t1a_positive_rungs": t1a_positive_rungs,
        "n_admissible_rungs": n_admissible_rungs,
        "t1a_positive_count": t1a_positive_count,
        "trend_result_primary": trend_result,
        "trend_result_s2": trend_result_s2,
        "tost_primary": tost_primary,
        "tost_s2": tost_s2,
        "factor1_primary": factor1_primary,
        "factor1_s2": factor1_s2,
        "verdict_before_t2a_gate": verdict,
        "slice_info": {f"{r}|{c}": slice_info_by_cell[(r, c)] for (r, c) in slice_info_by_cell},
        "all_deltas_pooled": all_deltas_pooled,
        "wall_s": time.time() - t0,
    }
    out_path = os.path.join(OUT_DIR, "r0_v2_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
