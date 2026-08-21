#!/usr/bin/env python3
"""K-SCALING BUILD SMOKE -- NCR_KSCALING_DESIGN.md sec 8. REAL CUDA ONLY.

Runs at one K per invocation (env NCR_K). Every POSITIVE item exercises the
PRODUCTION path (the patched runner's own functions, not reimplementations);
every NEGATIVE item must be shown to FIRE -- a negative test that is written
but not run to completion is worth nothing (CLAUDE.md, learned the hard way
on the C6 tolerance bug).

  A  derivation          D=K+1, param formula, ladder soundness, provenance
  B  NEG: pinned ladder  the carried (5,12,20,29,40,61) is REJECTED at this K
  C  NEG: silent dupe    a residue-colliding ladder that PASSES the pinned
                         guard is REJECTED by the new one   <-- the gate finding
  D  NEG: identity       an h == 0 mod K ladder is REJECTED (old check has teeth)
  E  NEG: train residue  an h == 1 mod K ladder is REJECTED (old check has teeth)
  F  params              MEASURED ncr/integ param counts == the derived formulas
  G  document geometry   shape, t_in, pad, and the KEY/VALUE/QUERY position
                         identities survive the left pad
  H  fwd/bwd/grad        both arms, P1b and P0, finite loss and finite grads
  I  ckpt/resume         save -> load -> restore -> BIT-IDENTICAL logits
  J  NEG: read-ablation  the pinned exact-zero invariant still holds
  K  NEG: unpadded T     the raw 7K+6 length crashes the kernel floor (fires
                         only where the pad is load-bearing, K in {12,16})
  L  throughput/util     s/step, peak memory, sampled SM utilisation -> the
                         placement plan's saturation-packing input
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch                               # noqa: E402
import kscaling_config as KS                # noqa: E402
import ncr_lm_wave1_runner as R             # noqa: E402

DEV = "cuda"
RESULTS: list[dict] = []


def item(name: str, kind: str = "positive"):
    def deco(fn):
        t = time.time()
        rec = {"item": name, "kind": kind}
        try:
            detail = fn()
            rec.update(status="PASS", detail=detail)
        except Exception as e:                       # noqa: BLE001
            rec.update(status="FAIL", exc_type=type(e).__name__, exc=str(e)[:600],
                       tb=traceback.format_exc()[-600:])
        rec["elapsed_s"] = round(time.time() - t, 2)
        RESULTS.append(rec)
        print(f"  [{rec['status']:<4}] {name}", flush=True)
        return fn
    return deco


def neg(name: str, expect_substr: str):
    """A NEGATIVE test PASSES only if the body RAISES and the message matches.
    A body that returns cleanly is recorded as FAIL ('did not fire')."""
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


def main() -> int:
    K = KS.K_NCR
    print(f"=== K-SCALING SMOKE  K={K} d={KS.D_NCR} t_in={KS.t_in()} pad={KS.doc_left_pad()} "
          f"ladder={KS.DEEP_LADDER} ===", flush=True)
    assert torch.cuda.is_available(), "REAL CUDA REQUIRED -- this smoke does not run on CPU"

    # ---------------- A: derivation -------------------------------------
    @item("A_derivation")
    def _a():
        assert KS.D_NCR == K + 1
        assert R.D_NCR == K + 1 and R.K_NCR == K
        # NCR_PARAM_EXACT is not re-exported by the runner; read it off the graft
        # module the runner imports, which is where patch S1 derives it.
        assert R.graft.NCR_PARAM_EXACT == KS.ncr_param_exact(R.H_NCR)
        assert R.graft.D_NCR == R.D_NCR == K + 1
        assert (K != 24) or (R.graft.NCR_PARAM_EXACT == 173_209)
        KS.assert_ladder_sound(KS.DEEP_LADDER, K)
        assert KS.H_TOP % K == K // 2, "top rung is not antipodal"
        assert KS.FIXED_DIST_PROBE % K == KS.FIXED_DIST_RESIDUE
        assert KS.n_squarings(KS.FIXED_DIST_PROBE) == KS.n_squarings(KS.H_TOP)
        return KS.provenance(R.H_NCR, R.RUNG1_BACKBONE["d_model"])

    # ---------------- B-E: ladder negatives ------------------------------
    @neg("B_NEG_pinned_ladder_rejected", "")
    def _b():
        KS.assert_ladder_sound(KS.REFERENCE_K24_LADDER, K)

    # A ladder that PASSES the pinned guard (no identity, no train residue) but
    # has two rungs on the SAME residue -- exactly the silent failure the pinned
    # guard misses. Built from this K's own ladder by moving one rung a full
    # period, which provably preserves its residue.
    dupe = tuple(sorted(KS.DEEP_LADDER[:-1] + (KS.DEEP_LADDER[0] + K,)))

    @neg("C_NEG_silent_residue_collision_rejected", "PAIRWISE residue collisions")
    def _c():
        train = {h % K for h in KS.TRAIN_HOPS}
        for h in dupe:                       # prove it PASSES the pinned guard first
            assert h % K != 0 and h % K not in train, (
                f"fixture {dupe} does not pass the pinned guard, so it cannot demonstrate "
                f"the pinned guard's blind spot")
        KS.assert_ladder_sound(dupe, K)

    @neg("D_NEG_identity_residue_rejected", "IDENTITY mod K")
    def _d():
        KS.assert_ladder_sound(KS.DEEP_LADDER[:-1] + (2 * K,), K)

    @neg("E_NEG_train_residue_rejected", "colliding with a train-residue")
    def _e():
        KS.assert_ladder_sound(KS.DEEP_LADDER[:-1] + (2 * K + 1,), K)

    # ---------------- F: params -----------------------------------------
    pools, cfg, report = R.build_grammar_pools_and_cfg(seed=0)
    pools = pools.to(DEV)
    arms = R.build_two_arms(report["vocab_size_total"], seed=0, device=DEV)
    arm = arms["full_graft"]

    @item("F_param_counts_exact")
    def _f():
        n_ncr = sum(p.numel() for p in arm["ncr"].parameters())
        n_integ = sum(p.numel() for p in arm["integ"].parameters())
        n_bb = sum(p.numel() for p in arm["backbone"].parameters())
        assert n_ncr == KS.ncr_param_exact(R.H_NCR), (n_ncr, KS.ncr_param_exact(R.H_NCR))
        assert n_integ == KS.integ_param_exact(R.RUNG1_BACKBONE["d_model"]), n_integ
        return {"ncr": n_ncr, "integ": n_integ, "backbone": n_bb, "total_per_arm": n_ncr + n_integ + n_bb}

    # ---------------- G: document geometry ------------------------------
    gen = torch.Generator(device=DEV).manual_seed(1234)
    batch = R.build_task1_document(cfg, pools, gen, 8, KS.H_TOP, DEV)

    @item("G_document_geometry_and_pad")
    def _g():
        doc = batch["doc"]
        assert doc.shape[1] == KS.doc_len() + KS.doc_left_pad(), (doc.shape, KS.doc_len())
        assert doc.shape[1] - 1 == KS.t_in() >= KS.MIN_KERNEL_T
        assert batch["query_mark_col"] == doc.shape[1] - 2
        ents = batch["entity_ids"]                                   # (B,K)
        # The KEY/VALUE tokens at the recorded positions must all be this row's
        # own entities, and the query-key token must be one of them. If the left
        # pad shifted anything wrongly these read BUFFER tokens and fail.
        for field in ("key_pos", "val_pos"):
            got = torch.gather(doc, 1, batch[field])                 # (B,K)
            assert bool((got.unsqueeze(2) == ents.unsqueeze(1)).any(2).all()), \
                f"{field} does not index this row's entities after a pad of {KS.doc_left_pad()}"
        qk = doc[:, batch["query_key_col"]]                          # (B,)
        assert bool((qk.unsqueeze(1) == ents).any(1).all()), \
            f"query_key_col does not index an entity after a pad of {KS.doc_left_pad()}"
        if KS.doc_left_pad():
            assert bool((doc[:, :KS.doc_left_pad()] == int(pools.buffer_id)).all()), \
                "left pad is not made of BUFFER tokens"
        return {"doc_shape": list(doc.shape), "pad": KS.doc_left_pad(), "t_in": KS.t_in(),
                "query_key_col": int(batch["query_key_col"]),
                "query_mark_col": int(batch["query_mark_col"])}

    # ---------------- H: fwd/bwd/grad, both regimes ----------------------
    @item("H_forward_backward_grad")
    def _h():
        out = {}
        for regime, tf in (("P1b", True), ("P0", False)):
            for a_name in ("full_graft", "backbone_only"):
                a = arms[a_name]
                for p in a["backbone"].parameters():
                    p.grad = None
                for p in a["ncr"].parameters():
                    p.grad = None
                total, ce, aux, ortho, o_raw, comps = R.compute_arm_losses(
                    a, batch, read_ablate=(a_name == "backbone_only"),
                    teacher_force=(tf and a_name == "full_graft"),
                    aux_read_loss_weight=0.5, is_full_graft=(a_name == "full_graft"),
                    ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
                    contrastive_temperature=0.07)
                total.backward()
                torch.cuda.synchronize()
                gnorm = sum(float(p.grad.norm()) for p in a["backbone"].parameters()
                            if p.grad is not None)
                nnz_ncr = sum(1 for p in a["ncr"].parameters() if p.grad is not None)
                assert torch.isfinite(total), f"{regime}/{a_name} loss not finite"
                assert gnorm == gnorm and gnorm > 0, f"{regime}/{a_name} backbone grad {gnorm}"
                out[f"{regime}/{a_name}"] = {
                    "loss": float(total.detach()), "ce": float(ce.detach()),
                    "grad_norm_backbone": gnorm,
                    "ncr_params_with_grad": nnz_ncr, "o_raw_shape": list(o_raw.shape)}
                assert o_raw.shape[-1] == KS.D_NCR, o_raw.shape
        return out

    # ---------------- J: read-ablation exact zero ------------------------
    @item("J_read_ablation_exact_zero")
    def _j():
        d = R.assert_read_ablation_is_exact_zero(
            arms["backbone_only"]["backbone"], arms["backbone_only"]["ncr"],
            arms["backbone_only"]["integ"], batch)
        return {"max_abs_diff": float(d)}

    # ---------------- I: checkpoint / resume -----------------------------
    @item("I_checkpoint_resume_bit_identical")
    def _i():
        ck_dir = os.environ.get("KSCALING_SMOKE_CKPT_DIR", "/ephemeral/kscaling_smoke")
        os.makedirs(ck_dir, exist_ok=True)
        path = os.path.join(ck_dir, f"smoke_K{K}.ckpt.pt")
        opts = {n: R.build_optimizer(arms[n], 3e-4, False) for n in arms}
        dg = torch.Generator(device=DEV).manual_seed(7)
        R.save_checkpoint(path, 20000, arms, opts, dg, 12.5, f"smoke_K{K}", seed=0,
                          freeze_entity_adapter=False)
        with torch.no_grad():
            before = R.ncr_lm_forward_ablatable(arm["backbone"], arm["ncr"], arm["integ"],
                                                 batch, read_ablate=False, teacher_force=True)[0]
        ck = R.load_checkpoint(path, DEV)
        assert ck is not None, "load_checkpoint rejected the checkpoint just written"
        assert ck["step"] == 20000 and ck["seed"] == 0
        assert ck["full_graft"]["ncr_config"]["d"] == KS.D_NCR, ck["full_graft"]["ncr_config"]
        r_arms, _o, _g = R.restore_arms_and_opts(ck, report["vocab_size_total"], 3e-4, DEV, False)
        ra = r_arms["full_graft"]
        for m in ("backbone", "ncr", "integ"):
            ra[m].eval()
            arm[m].eval()
        with torch.no_grad():
            after = R.ncr_lm_forward_ablatable(ra["backbone"], ra["ncr"], ra["integ"],
                                                batch, read_ablate=False, teacher_force=True)[0]
        for m in ("backbone", "ncr", "integ"):
            arm[m].train()
        maxdiff = float((before - after).abs().max())
        assert maxdiff == 0.0, f"resume is not bit-identical: max|diff|={maxdiff}"
        os.remove(path)
        return {"ckpt_dir": ck_dir, "max_abs_logit_diff": maxdiff,
                "recorded_d_ncr": ck["full_graft"]["ncr_config"]["d"]}

    # ---------------- K: the unpadded length really does crash ------------
    if KS.doc_left_pad() > 0:
        @neg("K_NEG_unpadded_T_crashes_kernel_floor", "_MIN_KERNEL_T")
        def _k():
            raw_t = KS.doc_len() - 1                      # what it would be with no pad
            x = torch.randint(0, R.VOCAB_SIZE, (2, raw_t), device=DEV)
            arm["backbone"](x, return_hidden=True)
    else:
        RESULTS.append({"item": "K_NEG_unpadded_T_crashes_kernel_floor", "kind": "negative",
                        "status": "N/A",
                        "reason": f"pad is 0 at K={K} (t_in={KS.t_in()} >= {KS.MIN_KERNEL_T}); "
                                  f"the pad is not load-bearing here, so there is nothing to fire"})
        print("  [N/A ] K_NEG_unpadded_T_crashes_kernel_floor  (pad not load-bearing at this K)",
              flush=True)

    # ---------------- L: throughput / memory / utilisation ----------------
    @item("L_throughput_memory_utilisation")
    def _l():
        opts = {n: R.build_optimizer(arms[n], 3e-4, False) for n in arms}
        gen2 = torch.Generator(device=DEV).manual_seed(99)
        torch.cuda.reset_peak_memory_stats()
        samples: list[int] = []
        stop = threading.Event()

        def sampler():
            vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            while not stop.is_set():
                try:
                    o = subprocess.run(
                        ["nvidia-smi", f"--id={vis}", "--query-gpu=utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5)
                    samples.append(int(o.stdout.strip().splitlines()[0]))
                except Exception:                        # noqa: BLE001
                    pass
                stop.wait(0.35)

        th = threading.Thread(target=sampler, daemon=True)
        n_steps, bs = 30, 32
        for i in range(5 + n_steps):                     # 5 warmup
            if i == 5:
                torch.cuda.synchronize()
                th.start()
                t0 = time.time()
            b = R.build_task1_document(cfg, pools, gen2, bs, KS.TRAIN_HOPS[i % 3], DEV)
            for a_name in ("full_graft", "backbone_only"):
                a = arms[a_name]
                opts[a_name].zero_grad(set_to_none=True)
                total, *_ = R.compute_arm_losses(
                    a, b, read_ablate=(a_name == "backbone_only"), teacher_force=False,
                    aux_read_loss_weight=0.5, is_full_graft=(a_name == "full_graft"),
                    ortho_reg_weight=0.1, aux_loss_type="contrastive+cosine",
                    contrastive_temperature=0.07)
                total.backward()
                opts[a_name].step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        stop.set()
        th.join(timeout=3)
        s_per_step = dt / n_steps
        peak = torch.cuda.max_memory_allocated() / 1e9
        util = sorted(samples)
        return {
            "batch_size": bs, "steps_timed": n_steps, "s_per_step": round(s_per_step, 4),
            "projected_gpu_h_20k_steps": round(s_per_step * 20000 / 3600, 3),
            "peak_mem_gb": round(peak, 3),
            "sm_util_samples": samples,
            "sm_util_median": util[len(util) // 2] if util else None,
            "sm_util_max": max(util) if util else None,
            "packing_headroom_by_mem_80gb": int(75 / peak) if peak > 0 else None}

    # ---------------- report ---------------------------------------------
    npass = sum(1 for r in RESULTS if r["status"] == "PASS")
    nfail = sum(1 for r in RESULTS if r["status"] == "FAIL")
    nna = sum(1 for r in RESULTS if r["status"] == "N/A")
    rec = {"smoke": "kscaling_smoke.py", "K": K, "d_ncr": KS.D_NCR,
           "provenance": KS.provenance(R.H_NCR, R.RUNG1_BACKBONE["d_model"]),
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "host": os.uname().nodename, "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_pass": npass, "n_fail": nfail, "n_na": nna,
           "overall": "PASS" if nfail == 0 else "FAIL", "items": RESULTS}
    outdir = os.environ.get("KSCALING_SMOKE_OUT", os.path.join(_HERE, "smoke_results"))
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"kscaling_smoke_K{K}.json")
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"=== K={K}: {npass} PASS / {nfail} FAIL / {nna} N/A -> {rec['overall']}  ({out}) ===",
          flush=True)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
