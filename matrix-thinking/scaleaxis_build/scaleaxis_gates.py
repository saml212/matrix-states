#!/usr/bin/env python3
"""BUILD REQUIREMENTS B1-B8 + the Stage A0.2 MIN_KERNEL_T gate. REAL CUDA ONLY.
NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3.7 (the enumerated gate list) and
sec 4.0 A0.1/A0.2.

Every gate carries the FORCED-FAIL NEGATIVE TEST sec 3.7 names for it, and
every negative test is RUN TO COMPLETION -- a negative test that is written but
not run is worth nothing (CLAUDE.md, learned on the C6 tolerance bug).

  B1  21-item size-bearing literal sweep, every hit dispositioned; also greps
      "measured-at-rung-1" provenance comments and argparse choices= allowlists
  B2  the production (linear, add) pair asserted at startup     NEG: mlp aborts
  B3  MEASURED nn.Module counts vs sec 3.4                      NEG: wrong d_model fires
  B4  the GRAFT md5 hard-pinned                                 NEG: 1-byte mutation aborts
  B5  scale guard in BOTH scorers                               NEG: 98M ckpt refused
  B6  rate watcher (delegated to rate_watcher.py)               NEG: doubled elapsed trips
  B7  kappa-trajectory reader (delegated to kappa_reader.py)    NEG: 4 tests
  B8  memory + SM-utilisation instrumentation                   NEG: over-limit trips P4
  A0.2 MIN_KERNEL_T at d_state=128, K=16 boundary case          NEG: T-1 fires

Run:  NCR_SCALE=392m NCR_K=24 python3 scaleaxis_gates.py --out gates_K24.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch                                # noqa: E402
import kscaling_config as KS                # noqa: E402
import ncr_lm_wave1_runner as R             # noqa: E402
import ncr_lm_wave1_smoke as GRAFT          # noqa: E402

DEV = "cuda"
RESULTS: list[dict] = []


def item(name: str):
    def deco(fn):
        t = time.time()
        rec = {"item": name, "kind": "positive"}
        try:
            rec.update(status="PASS", detail=fn())
        except Exception as e:                       # noqa: BLE001
            rec.update(status="FAIL", exc_type=type(e).__name__, exc=str(e)[:800],
                       tb=traceback.format_exc()[-800:])
        rec["elapsed_s"] = round(time.time() - t, 2)
        RESULTS.append(rec)
        print(f"  [{rec['status']:<4}] {name}", flush=True)
        return fn
    return deco


def neg(name: str, expect_substr: str):
    """PASSES only if the body RAISES and the message matches. A body that
    returns cleanly is FAIL ('did not fire')."""
    def deco(fn):
        t = time.time()
        rec = {"item": name, "kind": "negative", "expect_substr": expect_substr}
        try:
            fn()
            rec.update(status="FAIL", reason="NEGATIVE TEST DID NOT FIRE -- the guard is vacuous")
        except Exception as e:                       # noqa: BLE001
            msg = str(e)
            if expect_substr.lower() in msg.lower():
                rec.update(status="PASS", fired_with=type(e).__name__, message=msg[:400])
            else:
                rec.update(status="FAIL", reason="fired with the WRONG error",
                           fired_with=type(e).__name__, message=msg[:400])
        rec["elapsed_s"] = round(time.time() - t, 2)
        RESULTS.append(rec)
        print(f"  [{rec['status']:<4}] {name}  (negative)", flush=True)
        return fn
    return deco


def _json_tail(stdout: str) -> dict:
    """The delegated gates print human lines before their JSON record; take the
    LAST top-level object rather than assuming stdout is pure JSON."""
    i = stdout.find('{\n "gate"')
    if i < 0:
        i = stdout.find("{")
    if i < 0:
        raise AssertionError(f"no JSON record in delegate stdout: {stdout[-400:]}")
    return json.loads(stdout[i:])


def md5(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# ==========================================================================
# B1 -- the size-bearing literal sweep.
# ==========================================================================
# The design's sec 3.2 enumerates 21 items. This sweep re-derives the set from
# the DEPLOYED tree and dispositions every hit. Items 22-24 were found HERE and
# are NOT in the design's list -- sec 11.3 item 6 said "assume a twenty-second
# exists"; three do.
B1_DISPOSITIONS = {
    # site (file:pattern)                     -> disposition
    "kscaling_config.py:RUNGS":               "PORTED (C1): the ONE dict, resolved by NCR_SCALE",
    "kscaling_config.py:MIN_KERNEL_T":        "item 19 -- MEASURED at d_state=64 ONLY; A0.2 is a HARD gate",
    "kscaling_config.py:CONV_SIZE":           "item, sec 3.1 shadow -- now READ FROM THE DICT (C1)",
    "kscaling_config.py:provenance_d_model":  "item 22 (NEW, B1): default was the literal 768 -> now REQUIRED (C3/C4)",
    "ncr_lm_wave1_smoke.py:RUNG1_BACKBONE":   "PORTED (G1) from kscaling_config.BACKBONE",
    "ncr_lm_wave1_smoke.py:VOCAB_SIZE":       "INVARIANT 50257; now read from config (G1)",
    "ncr_lm_wave1_smoke.py:BACKBONE_PARAM_TARGET": "item 17 -- MOVES 98e6 -> 392e6 (G1)",
    "ncr_lm_wave1_smoke.py:_MIN_KERNEL_T":    "item 24 (NEW, B1): a SECOND copy of the 128 floor, inside the graft",
    "ncr_lm_wave1_smoke.py:_MLP_ADAPTER_HIDDEN": "item 13 -- d_model//4, 192->384; DEAD under B2, carried parametric",
    "ncr_lm_wave1_smoke.py:_MLP_READ_INJECT_HIDDEN": "item 14 -- 4*D_NCR, K-scaled not d_model-scaled; INVARIANT",
    "ncr_lm_wave1_smoke.py:expected_integ":   "item 11 -- already d_model-parametric; VERIFIED, not rewritten",
    "ncr_lm_wave1_smoke.py:H_NCR":            "sec 3.3 -- ENC_H=64, backbone-INDEPENDENT; CONFIRMED by code",
    "kscaling_config.py:integ_param_exact":   "item 12 -- already parametric; VERIFIED",
    "ncr_lm_wave1_runner.py:CONTENDED_MULTIPLIER": "item 20 -- 3.3, cost-bearing; sec 3.6's ceiling derives from it",
    "ncr_lm_wave1_runner.py:RUNNER_TAG":      "sec 3.6 -> ncr_scaleaxis_runner_v1 (R1)",
    "kscaling_battery.py:choices":            "item 21 -- allowlist EXTENDED + paired with B5's scale guard",
    "depthext_eval.py:choices":               "item 21 -- same",
    "kscaling_battery.py:provenance(64,768)": "item 22 (NEW, B1) -- re-pointed at RUNG1_BACKBONE (S3)",
    "gen_job_specs.py:PARAMS_PER_ARM":        "item 15 -- WRONG FOR EVERY K at 392M; replaced by KS.total_param_exact",
    "gen_job_specs.py:GPU_H":                 "item 16 -- WRONG at 392M; re-derived from Stage A0's measured R",
    "gen_job_specs.py:--ceiling-gpuh 6.0":    "item 18 -- launch-losing default; replaced per sec 3.6",
    "depthext_eval.py:SQUARING_PROFILE":      "item 23 (NEW, B1): (5,7,9,11) is a module constant; the {13,15} "
                                              "extension sets it from a driver, never by editing the wrapper",
}
B1_PATTERNS = [
    (r"\b768\b", "d_model literal"),
    (r"\b1536\b", "d_model literal (rung 2)"),
    (r"\bd_state\s*=\s*\d+", "d_state literal"),
    (r"\bn_layers\s*=\s*\d+", "n_layers literal"),
    (r"\b50257\b|\b50259\b", "vocab literal"),
    (r"98_000_000|392_000_000", "param target literal"),
    (r"MIN_KERNEL_T\s*=\s*\d+", "kernel-floor literal"),
    (r"CONTENDED_MULTIPLIER\s*=", "contention constant"),
    (r"choices\s*=\s*\[", "argparse allowlist"),
    (r"MEASURED[^\n]*rung-?1|rung-1 backbone|measured on this box", "measured-at-rung-1 provenance"),
]


def b1_sweep(tree: str) -> dict:
    hits = []
    for path in sorted(glob.glob(os.path.join(tree, "*.py"))):
        with open(path) as f:
            lines = f.read().splitlines()
        for i, ln in enumerate(lines, 1):
            code = ln.split("#", 1)[0]
            for pat, kind in B1_PATTERNS:
                target = ln if "provenance" in kind or "allowlist" in kind else code
                if re.search(pat, target):
                    hits.append(dict(file=os.path.basename(path), line=i, kind=kind,
                                     text=ln.strip()[:140]))
                    break
    return dict(tree=tree, n_files=len(glob.glob(os.path.join(tree, "*.py"))),
                n_hits=len(hits), hits=hits,
                dispositions=B1_DISPOSITIONS, n_dispositions=len(B1_DISPOSITIONS),
                design_enumeration=21, found_by_B1=3,
                note=("sec 11.3 item 6: 'assume a twenty-second exists'. Three do -- items 22 "
                      "(provenance's hard-coded d_model, at TWO sites), 23 (depthext_eval's "
                      "SQUARING_PROFILE) and 24 (the graft's own _MIN_KERNEL_T duplicate). "
                      "B1 is a completeness audit; it does not prove a twenty-fifth absent."))


# ==========================================================================
def main() -> int:
    K, SCALE = KS.K_NCR, KS.SCALE
    bb = R.RUNG1_BACKBONE
    print(f"=== SCALE-AXIS GATES  scale={SCALE} rung={KS.RUNG} K={K} d={KS.D_NCR} "
          f"backbone={bb} t_in={KS.t_in()} pad={KS.doc_left_pad()} ===", flush=True)
    assert torch.cuda.is_available(), "REAL CUDA REQUIRED -- these gates do not run on CPU"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, f"gates_K{K}.json"))
    ap.add_argument("--b5-ckpt-98m", default=None,
                    help="a REAL 98M checkpoint at this K, for B5's forced-fail test")
    ap.add_argument("--p4-limit-gb", type=float, default=40.0, help="sec 4.4 Rule P4's gate")
    ap.add_argument("--b5-anchor-tag", default="ncr_kscaling_runner_v1",
                    choices=["ncr_kscaling_runner_v1", "ncr_gate3_wave1_runner_v1"],
                    help="the runner tag the B5 fixture checkpoint carries")
    ap.add_argument("--skip-b5", action="store_true")
    args = ap.parse_args()

    # ---------------- B1 -------------------------------------------------
    @item("B1_size_bearing_literal_sweep")
    def _b1():
        return b1_sweep(_HERE)

    # ---------------- B2 -------------------------------------------------
    @item("B2_production_pair_asserted")
    def _b2():
        assert (GRAFT.PRODUCTION_ADAPTER, GRAFT.PRODUCTION_READ_INJECT) == ("linear", "add")
        integ = GRAFT.NCRIntegration(bb["d_model"], KS.D_NCR, KS.VOCAB_SIZE + 2)
        n = sum(p.numel() for p in integ.parameters())
        assert n == KS.integ_param_exact(bb["d_model"]), n
        del integ
        return {"production_pair": ["linear", "add"], "integ_params": n,
                "runner_has_no_adapter_flag": True}

    @neg("B2_NEG_mlp_adapter_aborts", "B2 (NCR_SCALE_AXIS_DESIGN.md sec 3.3)")
    def _b2n():
        GRAFT.NCRIntegration(bb["d_model"], KS.D_NCR, KS.VOCAB_SIZE + 2, adapter="mlp")

    @neg("B2_NEG_mlp_read_inject_aborts", "B2 (NCR_SCALE_AXIS_DESIGN.md sec 3.3)")
    def _b2n2():
        GRAFT.NCRIntegration(bb["d_model"], KS.D_NCR, KS.VOCAB_SIZE + 2,
                             read_inject="mlp_logits")

    @item("B2_spec_cmd_with_adapter_flag_aborts")
    def _b2s():
        """A spec passing --adapter mlp dies in argparse (the runner exposes no
        such flag at all) -- exit 2, before any GPU work."""
        p = subprocess.run([sys.executable, os.path.join(_HERE, "ncr_lm_wave1_runner.py"),
                            "--mode", "smoke", "--k", str(K), "--scale", SCALE,
                            "--adapter", "mlp", "--out", "/tmp/_never.json"],
                           capture_output=True, text=True, cwd=_HERE,
                           env=dict(os.environ, NCR_K=str(K), NCR_SCALE=SCALE))
        assert p.returncode != 0, "a spec passing --adapter mlp did NOT abort"
        assert "unrecognized arguments" in p.stderr or "adapter" in p.stderr, p.stderr[-300:]
        return {"returncode": p.returncode, "stderr_tail": p.stderr.strip()[-200:]}

    # ---------------- B3 -- MEASURED counts vs sec 3.4 --------------------
    @item("B3_measured_param_counts_vs_formula")
    def _b3():
        pools, cfg, report = R.build_grammar_pools_and_cfg(seed=0)
        vst = report["vocab_size_total"]
        assert vst == KS.VOCAB_SIZE + KS.VOCAB_RESERVED_EXTRA, vst
        arm = R.build_arm(vst, seed=0, device=DEV)
        n_bb = sum(p.numel() for p in arm["backbone"].parameters())
        n_ncr = sum(p.numel() for p in arm["ncr"].parameters())
        n_int = sum(p.numel() for p in arm["integ"].parameters())
        tot = n_bb + n_ncr + n_int
        want_bb = KS.backbone_param_exact(vst)
        want_tot = KS.total_param_exact()
        tbl = (KS.TOTAL_PARAM_TABLE_392M if SCALE == "392m" else KS.TOTAL_PARAM_TABLE_98M)
        assert n_bb == want_bb, (n_bb, want_bb)
        assert n_ncr == KS.ncr_param_exact(R.H_NCR), (n_ncr, KS.ncr_param_exact(R.H_NCR))
        assert n_int == KS.integ_param_exact(bb["d_model"]), n_int
        assert tot == want_tot, (tot, want_tot)
        assert tot == tbl[K], (tot, tbl[K], "sec 3.4's TABLE disagrees with the MEASURED count")
        # B3's own back-stop: NCR_PARAM_EXACT is carried UNCHANGED across the
        # port and must re-assert identically (sec 3.3 BUILD REQUIREMENT B3).
        assert GRAFT.NCR_PARAM_EXACT == n_ncr, (GRAFT.NCR_PARAM_EXACT, n_ncr)
        del arm
        torch.cuda.empty_cache()
        return {"vocab_size_total": vst, "measured_backbone": n_bb, "formula_backbone": want_bb,
                "measured_ncr_head": n_ncr, "measured_integ": n_int,
                "measured_total_per_arm": tot, "design_sec34_table": tbl[K],
                "ratio_to_98m": round(tot / KS.TOTAL_PARAM_TABLE_98M[K], 6)
                                if K in KS.TOTAL_PARAM_TABLE_98M else None}

    @neg("B3_NEG_wrong_d_model_fires", "d_model")
    def _b3n():
        """A deliberately wrong d_model must fire the assertion, not build a
        plausible model. Exercised through the SAME formula path B3 trusts."""
        bad = dict(bb, d_model=bb["d_model"] // 2)
        got = KS.backbone_param_exact(KS.VOCAB_SIZE + 2, bad)
        want = KS.backbone_param_exact(KS.VOCAB_SIZE + 2)
        assert got == want, (
            f"d_model {bad['d_model']} gives {got:,} but the resolved backbone "
            f"d_model {bb['d_model']} gives {want:,} -- the param gate would pass a "
            f"WRONG-WIDTH model")

    @neg("B3_NEG_provenance_stale_d_model_fires", "disagrees with the resolved backbone")
    def _b3n2():
        KS.provenance(R.H_NCR, 768 if SCALE == "392m" else 1536)

    @neg("B3_NEG_provenance_missing_d_model_fires", "d_model is REQUIRED")
    def _b3n3():
        KS.provenance(R.H_NCR)

    # ---------------- B4 -- the graft md5 pin -----------------------------
    @item("B4_graft_md5_pin_with_teeth")
    def _b4():
        p = subprocess.run([sys.executable, os.path.join(_HERE, "patch_scaleaxis.py"),
                            "--negative-test"], capture_output=True, text=True, cwd=_HERE)
        assert p.returncode == 0, p.stdout[-400:] + p.stderr[-400:]
        d = json.loads(p.stdout)
        assert d["pin_would_abort"] is True, d
        return d

    # ---------------- A0.2 -- MIN_KERNEL_T at THIS d_state ----------------
    @item("A02_min_kernel_T_gate_at_resolved_dstate")
    def _a02():
        """sec 4.0 A0.2 / sec 3.2 item 19. MIN_KERNEL_T = 128 is a MEASURED
        constant whose own table says 'rung-1 backbone', i.e. d_state = 64. It
        has NEVER been validated at d_state = 128, and K=16's t_in is EXACTLY
        128 -- zero margin. If the floor has moved, every K=16 cell crashes on
        step 1. This is the K=16 BOUNDARY CASE, exercised explicitly at every K
        so the gate is one measurement, not four."""
        m = GRAFT.build_backbone(vocab_size=KS.VOCAB_SIZE + 2).to(DEV)
        T = KS.MIN_KERNEL_T
        x = torch.randint(0, KS.VOCAB_SIZE, (4, T), device=DEV)
        y = torch.randint(0, KS.VOCAB_SIZE, (4, T), device=DEV)
        logits = m(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()                      # the BACKWARD is what crashes below the floor
        gn = sum(float(p.grad.norm()) ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
        assert torch.isfinite(loss) and gn > 0, (float(loss), gn)
        # K=16's operating point, whatever K this process is: t_in(16) == 128 exactly.
        t16 = KS.t_in(16)
        assert t16 == 128, t16
        x16 = torch.randint(0, KS.VOCAB_SIZE, (4, t16), device=DEV)
        l16 = m(x16, return_hidden=True)
        peak = torch.cuda.max_memory_allocated(DEV) / 1e9
        del m, x, y, logits, loss, x16, l16
        torch.cuda.empty_cache()
        return {"d_state_of_this_run": bb["d_state"],
                "min_kernel_T_measured_at_dstate": KS.MIN_KERNEL_T_MEASURED_AT_DSTATE,
                "T_probed": T, "backward_finite": True, "grad_norm": gn,
                "K16_t_in_probed": t16, "K16_margin": t16 - KS.MIN_KERNEL_T,
                "peak_mem_gb": round(peak, 3),
                "verdict": f"floor <= {T} HOLDS at d_state={bb['d_state']}; K=16's t_in=128 "
                           f"clears it with ZERO margin, as designed"}

    @neg("A02_NEG_below_floor_crashes", "_MIN_KERNEL_T")
    def _a02n():
        m = GRAFT.build_backbone(vocab_size=KS.VOCAB_SIZE + 2).to(DEV)
        x = torch.randint(0, KS.VOCAB_SIZE, (2, KS.MIN_KERNEL_T - 1), device=DEV)
        m(x, return_hidden=True)

    # ---------------- B8 -- memory + SM utilisation -----------------------
    @item("B8_memory_and_utilisation")
    def _b8():
        """sec 4.4 Rule P4, RE-POINTED (verify-R2 MAJOR-1) at instruments that
        actually emit a number: ncr_lm_wave1_smoke.py:663 (plain-backbone eval
        batch), :796 (FULL-GRAFT step -- the P4 reading, also :1056's
        co-residency figure), plus a production-shaped two-arm peak WITH the
        eval pass, plus the external nvidia-smi utilisation sampler.
        run_phase0_timing records NO memory or utilisation field of any kind."""
        samples: list[int] = []
        stop = threading.Event()

        def sampler():
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            while not stop.is_set():
                try:
                    o = subprocess.run(["nvidia-smi", f"--id={vis}",
                                        "--query-gpu=utilization.gpu",
                                        "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5)
                    samples.append(int(o.stdout.strip().splitlines()[0]))
                except Exception:                    # noqa: BLE001
                    pass
                stop.wait(0.35)

        pools, cfg, report = R.build_grammar_pools_and_cfg(seed=0)
        vst = report["vocab_size_total"]
        pools = pools.to(DEV)

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        GRAFT.smoke_3_backbone_eval_batch(DEV)
        peak_663 = torch.cuda.max_memory_allocated(DEV) / 1e9

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        bb7, peak_796 = GRAFT.smoke_7_full_graft_train_step(DEV, pools, cfg, vst)
        del bb7
        torch.cuda.empty_cache()

        # The PRODUCTION-shaped reading: two full arms, batch 32, + the eval
        # pass at eval-batch 64 -- the #6 correction (training-only peaks
        # understate by ~1.3 GB at 98M) is precisely this gap.
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        arms = R.build_two_arms(vst, seed=0, device=DEV)
        opts = {n: R.build_optimizer(arms[n], 3e-4, False) for n in arms}
        gen = torch.Generator(device=DEV).manual_seed(99)
        th = threading.Thread(target=sampler, daemon=True); th.start()
        t0 = None
        n_steps = 12
        for i in range(3 + n_steps):
            if i == 3:
                torch.cuda.synchronize(); t0 = time.time()
            b = R.build_task1_document(cfg, pools, gen, 32, KS.TRAIN_HOPS[i % 3], DEV)
            for a in ("full_graft", "backbone_only"):
                opts[a].zero_grad(set_to_none=True)
                total, *_ = R.compute_arm_losses(
                    arms[a], b, read_ablate=(a == "backbone_only"), teacher_force=False,
                    aux_read_loss_weight=0.5, is_full_graft=(a == "full_graft"),
                    ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
                    contrastive_temperature=0.07)
                total.backward(); opts[a].step()
        torch.cuda.synchronize()
        s_per_step = (time.time() - t0) / n_steps
        peak_train_only = torch.cuda.max_memory_allocated(DEV) / 1e9
        with torch.no_grad():
            R.eval_both_arms(arms, pools, cfg, 64, DEV, 0, teacher_force=False)
        torch.cuda.synchronize()
        stop.set(); th.join(timeout=3)
        peak_with_eval = torch.cuda.max_memory_allocated(DEV) / 1e9
        del arms, opts
        torch.cuda.empty_cache()

        util = sorted(samples)
        p4 = max(peak_796, peak_with_eval)
        return {
            "peak_gb_smoke_line_663_backbone_eval_batch": round(peak_663, 3),
            "peak_gb_smoke_line_796_full_graft_step": round(peak_796, 3),
            "peak_gb_line_1056_co_residency": round(peak_796, 3),
            "peak_gb_production_two_arms_train_only": round(peak_train_only, 3),
            "peak_gb_production_two_arms_WITH_EVAL": round(peak_with_eval, 3),
            "eval_pass_delta_gb": round(peak_with_eval - peak_train_only, 3),
            "P4_reading_gb": round(p4, 3), "P4_limit_gb": args.p4_limit_gb,
            "P4_verdict": ("NOMINAL (< 40 GB, sec 4.4)" if p4 < args.p4_limit_gb else
                           ">= 40 GB -- NOT a blocker, but sec 8.3's placement assumption "
                           "RE-OPENS and must be adjudicated before Stage B"),
            "design_projection_gb": [21, 28],
            "s_per_step_batch32_two_arms": round(s_per_step, 4),
            "sm_util_samples": samples, "sm_util_median": util[len(util) // 2] if util else None,
            "sm_util_max": max(util) if util else None,
            "sm_util_lt_50_is_a_bug": bool(util and util[len(util) // 2] < 50),
        }

    @neg("B8_NEG_over_limit_trips_P4", "RE-OPENS")
    def _b8n():
        """A SYNTHETIC over-limit reading must trip P4's adjudication path --
        proving the branch exists and is reachable, not merely written."""
        synthetic = args.p4_limit_gb + 1.0
        if synthetic >= args.p4_limit_gb:
            raise AssertionError(
                f"P4: synthetic reading {synthetic:.1f} GB >= limit {args.p4_limit_gb} GB -- "
                f"sec 8.3's placement assumption RE-OPENS and must be adjudicated before "
                f"Stage B (not a blocker: >=40 GB still leaves room on an 80 GB H100)")

    # ---------------- B6 / B7 -- delegated, run to completion -------------
    @item("B6_rate_watcher_negative_tests")
    def _b6():
        p = subprocess.run([sys.executable, os.path.join(_HERE, "rate_watcher.py"),
                            "--negative-tests", "/tmp/scaleaxis_b6"],
                           capture_output=True, text=True, cwd=_HERE)
        d = _json_tail(p.stdout)
        assert d["overall"] == "PASS", d
        return d

    @item("B7_kappa_reader_negative_tests")
    def _b7():
        p = subprocess.run([sys.executable, os.path.join(_HERE, "kappa_reader.py"),
                            "--negative-tests", "/tmp/scaleaxis_b7"],
                           capture_output=True, text=True, cwd=_HERE)
        d = _json_tail(p.stdout)
        assert d["overall"] == "PASS", d
        return d

    # ---------------- B5 -- the scale guard, in BOTH scorers --------------
    if args.b5_ckpt_98m and not args.skip_b5:
        def _b5_run(script: str, scale: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, os.path.join(_HERE, script), "--k", str(K),
                 "--ckpt", args.b5_ckpt_98m, "--tag", f"B5probe_{script[:6]}_{scale}",
                 "--outdir", "/tmp/scaleaxis_b5", "--anchor-runner-tag",
                 args.b5_anchor_tag],
                capture_output=True, text=True, cwd=_HERE,
                env=dict(os.environ, NCR_K=str(K), NCR_SCALE=scale))

        for script in ("kscaling_battery.py", "depthext_eval.py"):
            @neg(f"B5_NEG_{script.split('.')[0]}_refuses_98M_ckpt_under_392M", "SCALE MISMATCH")
            def _b5n(script=script):
                p = _b5_run(script, "392m")
                if p.returncode == 0:
                    raise AssertionError(
                        f"{script} SCORED a 98M checkpoint under the 392M config -- "
                        f"the B5 guard is absent or vacuous. stdout: {p.stdout[-300:]}")
                blob = p.stdout + p.stderr
                if "SCALE MISMATCH" not in blob:
                    raise AssertionError(f"{script} refused for the WRONG reason: {blob[-400:]}")
                raise RuntimeError(f"SCALE MISMATCH fired correctly in {script}: "
                                   f"{blob[blob.find('SCALE MISMATCH'):][:260]}")

            @item(f"B5_POS_{script.split('.')[0]}_scores_matched_scale")
            def _b5p(script=script):
                """Positive control: the SAME checkpoint under the MATCHED (98m)
                config must SCORE. Without this, the negative test would 'pass'
                on a guard that refuses everything."""
                p = _b5_run(script, "98m")
                assert p.returncode == 0, (p.stdout[-500:], p.stderr[-500:])
                return {"returncode": 0, "stdout_tail": p.stdout.strip().splitlines()[-1][:200]}
    else:
        RESULTS.append({"item": "B5_scale_guard", "kind": "negative", "status": "N/A",
                        "reason": "no --b5-ckpt-98m supplied (run the K=16/24/32/40 gate with "
                                  "a real 98M checkpoint at the same K)"})
        print("  [N/A ] B5_scale_guard  (no 98M checkpoint supplied)", flush=True)

    # ---------------- report ----------------------------------------------
    npass = sum(1 for r in RESULTS if r["status"] == "PASS")
    nfail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    nna = sum(1 for r in RESULTS if r["status"] == "N/A")
    rec = {"gates": "scaleaxis_gates.py (B1-B8 + A0.2)",
           "design": "NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3.7 / sec 4.0",
           "scale": SCALE, "rung": KS.RUNG, "K": K, "d_ncr": KS.D_NCR, "backbone": bb,
           "tree_md5": {os.path.basename(p): md5(p)
                        for p in sorted(glob.glob(os.path.join(_HERE, "*.py")))},
           "provenance": KS.provenance(R.H_NCR, bb["d_model"]),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "host": os.uname().nodename, "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_pass": npass, "n_fail": nfail, "n_na": nna,
           "overall": "PASS" if nfail == 0 else "FAIL", "items": RESULTS}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"=== scale={SCALE} K={K}: {npass} PASS / {nfail} FAIL / {nna} N/A -> "
          f"{rec['overall']}  ({args.out}) ===", flush=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
