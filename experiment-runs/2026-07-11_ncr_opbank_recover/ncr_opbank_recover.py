"""NCR operator-bank CONVERGENCE-RECOVERY diagnosis -- NOVEL_ARCH_WATERFALL.md
§8.7. ADDITIVE build: imports ncr_opbank_{task,models} and run_ncr_opbank
VERBATIM, modifies NONE of them (the §9 agent shares the ncr/ tree --
single-writer discipline).

Question: WHY does the raw-matmul exact-composition contender fail to
converge at R=3 (§8.5a: flat chance at the proven 256/80K budget) while
single-relation NCR and fwm-bank (identical R=3 write load) both converge?
Four training-recipe arms on the SAME contender (pure-matmul EXACT read at
EVAL always -- exactness never compromised):

  baseline    control: flat Adam 3e-4, axis_b_frac=0.5, no LN. MUST
              reproduce §8.5a non-convergence or the comparison is confounded.
  warmup      (b-i)  linear-warmup (4K) + cosine-decay LR.
  earlyln     (b-ii) parameter-free inter-hop LN blended into the TRAIN read
              with weight α annealed 1.0→0.0 over the first half, 0 after.
              At α=0 the forward is BIT-IDENTICAL to the plain contender
              (closed-form-tested); EVAL always uses the parent's pure exact
              read, so the final model is the exact contender.
  curriculum  (b-iii) axis_b_frac ramped 0.0→0.5 over the first half
              (single-block first, 2-block chains added gradually).

Verdict map (§8.7, pinned): primary = in-dist (h=1,2,3) recovered@0.9
min-over-3-relations. RECOVERED ≥0.9 (+A_eff_rank→8 +swap gap>0.3);
PARTIAL [0.5,0.9); NONE-RECOVER <0.5 (honest negative). Control: baseline
must stay <0.5.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
for p in (CHAPTER2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import ncr_opbank_task as ot            # noqa: E402 (verbatim)
import ncr_opbank_models as om          # noqa: E402 (verbatim)
import run_ncr_opbank as ro             # noqa: E402 (verbatim -- instrument reuse)

RECOVER_ARMS = ("baseline", "warmup", "earlyln", "curriculum")
BASE_LR = ro.TRAIN_LR                   # 3e-4, same as §8.5a
WARMUP_STEPS = 4000
LR_FLOOR_FRAC = 0.1
RUNNER_TAG = "ncr_opbank_recover_v1"


class NCRBankRecoverModel(om.NCRBankModel):
    """The §8.5 contender, EXACT at eval; the ONLY change is an optional
    parameter-free inter-hop LN blended into the TRAIN read step, weight
    `self._ln_alpha` (default 0.0). At α=0 forward() is bit-identical to
    NCRBankModel.forward (asserted in the self-test). arm stays 'ncr-bank'
    so every run_ncr_opbank instrument routes through the pure-matmul
    ncr-bank branch unchanged."""

    def __init__(self, d: int = om.D_PIN, R: int = om.R_PIN):
        super().__init__(d, R)
        self._ln_alpha = 0.0     # runner sets this per step for the earlyln arm

    def _blend(self, stepped):
        a = self._ln_alpha
        if a <= 0.0:
            return stepped        # bit-identical to the plain contender
        return a * F.layer_norm(stepped, (self.d,)) + (1.0 - a) * stepped

    def forward(self, batch: dict):
        Z = self.encode(batch["keys"], batch["values"], batch["rel_ids"])
        v = batch["query_keys"]
        Zr1 = om._select_Z(Z, batch["r1"])
        max_h1 = int(batch["h1"].max().item())
        cur = v
        for t in range(1, max_h1 + 1):
            stepped = self._blend(torch.einsum("bqij,bqj->bqi", Zr1, cur))
            cur = torch.where((batch["h1"] >= t).unsqueeze(-1), stepped, cur)
        mid = cur
        Zr2 = om._select_Z(Z, batch["r2"])
        max_h2 = int(batch["h2"].max().item())
        cur2 = mid
        for t in range(1, max_h2 + 1):
            stepped = self._blend(torch.einsum("bqij,bqj->bqi", Zr2, cur2))
            cur2 = torch.where((batch["h2"] >= t).unsqueeze(-1), stepped, cur2)
        return cur2, Z


# ---------------------------------------------------------------------------
# Per-arm schedules (pure functions; pinned in §8.7)
# ---------------------------------------------------------------------------

def lr_at(step: int, total: int, arm: str) -> float:
    if arm != "warmup":
        return BASE_LR
    if step <= WARMUP_STEPS:
        return BASE_LR * step / WARMUP_STEPS
    prog = (step - WARMUP_STEPS) / max(1, total - WARMUP_STEPS)
    return BASE_LR * (LR_FLOOR_FRAC + (1 - LR_FLOOR_FRAC) * 0.5 * (1 + math.cos(math.pi * prog)))


def ln_alpha_at(step: int, total: int, arm: str) -> float:
    if arm != "earlyln":
        return 0.0
    half = total // 2
    if step >= half:
        return 0.0
    return 1.0 - step / half        # 1.0 → 0.0 over the first half


def axis_b_frac_at(step: int, total: int, arm: str) -> float:
    if arm != "curriculum":
        return 0.5
    half = total // 2
    if step >= half:
        return 0.5
    return 0.5 * step / half        # 0.0 → 0.5 over the first half


# ---------------------------------------------------------------------------
# Training (mirrors run_ncr_opbank.train_cell_bank + the §8.7 schedules)
# ---------------------------------------------------------------------------

def train_recover_cell(model, cfg, device, steps, seed, arm, stop_file, ceiling_s,
                       train_batch, log_every=500):
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    model.train()
    n_skipped, loss_hist = 0, []
    t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, arm)
        model._ln_alpha = ln_alpha_at(step, steps, arm)
        frac = axis_b_frac_at(step, steps, arm)
        b = ot.sample_train_batch(cfg, train_batch, gen, device, axis_b_frac=frac)
        for k in ("keys", "values", "rel_ids", "query_keys", "r1", "h1", "r2", "h2", "targets"):
            b[k] = b[k].to(device)
        pred, _ = model(b)
        loss = ro.cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            loss_hist.append([step, float(loss.item())])
            print(f"  [{arm}] step {step:6d}  loss {loss.item():.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"ln_a {model._ln_alpha:.2f}  abf {frac:.2f}  {elapsed:.0f}s", flush=True)
            if ro.stop_requested(stop_file):
                print(f"  [{arm}] STOP -- exiting 3"); sys.exit(3)
            if elapsed > ceiling_s:
                return dict(status="ABORTED-BUDGET", step=step, elapsed_s=elapsed,
                           ceiling_s=ceiling_s, n_skipped_steps=n_skipped, loss_history=loss_hist)
    model._ln_alpha = 0.0     # final model is the EXACT pure-matmul contender
    return dict(status="COMPLETED", step=steps, elapsed_s=time.time() - t0,
               ceiling_s=ceiling_s, n_skipped_steps=n_skipped, loss_history=loss_hist)


def cell_id(arm: str) -> str:
    return f"ncropbank_recover_{arm}"


def run_recover_cell(arm: str, seed: int, steps: int, device: str, outdir: str,
                     stop_file: str = "", ceiling_gpuh: float = 2.0,
                     train_batch: int = 256) -> dict:
    assert arm in RECOVER_ARMS, arm
    os.makedirs(outdir, exist_ok=True)
    cid = cell_id(arm)
    out_path = os.path.join(outdir, f"{cid}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "COMPLETED":
            print(f"  [{cid}] already COMPLETED -- skipping (resume-safe)", flush=True)
            return prev
    cfg = ot.BankConfig()
    torch.manual_seed(seed)
    model = NCRBankRecoverModel().to(device)
    rec = dict(cell_id=cid, recover_arm=arm, arm=model.arm, seed=seed, runner_tag=RUNNER_TAG,
              git_commit=ro.git_commit(), params=om.n_params(model), train_batch=train_batch,
              host=socket.gethostname(), device=device, torch_version=torch.__version__,
              started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ceiling_s = ceiling_gpuh * 3600.0 if device == "cuda" else float("inf")
    t0 = time.time()

    tr = train_recover_cell(model, cfg, device, steps, seed, arm, stop_file, ceiling_s, train_batch)
    rec["train"] = tr
    if tr["status"] != "COMPLETED":
        rec["status"] = tr["status"]
        rec["elapsed_s"] = time.time() - t0
        ro.atomic_write_json(out_path, rec)
        return rec

    model.eval()
    model._ln_alpha = 0.0            # exact read for ALL instruments (belt-and-suspenders)
    h_star = ot.GRID8["h_star"]
    zd = ro.z_dump_bank(model, cfg, device, seed)
    rec["deep_probe_per_relation"] = ro.deep_probe_bank(zd, cfg, outdir, cid, hops=[1, 2, 3, h_star])
    rec["blank_out"] = ro.blank_out_check_bank(model, cfg, device, seed)
    assert rec["blank_out"]["passed"], f"blank-out FAILED for {cid}: {rec['blank_out']}"
    rec["relation_id_swap"] = ro.relation_id_swap_ablation(model, cfg, device, seed, h=h_star)
    rec["eval"] = ro.eval_cell_bank_small(model, cfg, device, seed, h_star)
    rec["elapsed_s"] = time.time() - t0
    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
    rec["status"] = "COMPLETED"
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    ro.atomic_write_json(out_path, rec)
    print(f"  [{cid}] COMPLETED in {rec['elapsed_s']:.0f}s -> {out_path}", flush=True)
    return rec


def harvest(outdir: str) -> dict:
    """§8.8 verdict from the 4 recover cells against the §8.7 pinned map."""
    cells = {}
    for arm in RECOVER_ARMS:
        p = os.path.join(outdir, f"{cell_id(arm)}.json")
        if not os.path.exists(p):
            cells[arm] = dict(status="MISSING")
            continue
        d = json.load(open(p))
        if d.get("status") != "COMPLETED":
            cells[arm] = dict(status=d.get("status"))
            continue
        pts = d["eval"]["points"]
        # Primary = min over ALL 9 (h in {1,2,3} x r in {0,1,2}) cells -- STRICTER
        # than §8.7's literal "min over 3 relations per-h" (it can only LOWER a
        # verdict, never inflate). The per-h min-over-r breakdown is reported
        # alongside for transparency (§8.8a audit MINOR-1).
        per_h_min_over_r = {h: min(pts[str(h)][str(r)]["recovered_frac_0_9"] for r in range(3))
                            for h in (1, 2, 3)}
        indist_min = min(per_h_min_over_r.values())
        indist_meancos = min(pts[str(h)][str(r)]["mean_cos"] for h in (1, 2, 3) for r in range(3))
        dp = d.get("deep_probe_per_relation", {})
        aer = [x for r in ("0", "1", "2") for x in dp.get(r, {}).get("A_eff_rank", [])]
        sw = d.get("relation_id_swap", {})
        cells[arm] = dict(status="COMPLETED", train_final_loss=d["train"]["loss_history"][-1][1],
                          indist_min_recovered_at_0_9=indist_min,
                          per_h_min_over_r={str(h): v for h, v in per_h_min_over_r.items()},
                          indist_min_mean_cos=indist_meancos,
                          A_eff_rank_min=(min(aer) if aer else None),
                          A_eff_rank_max=(max(aer) if aer else None),
                          swap_gap=sw.get("right_minus_wrong_gap"), gpu_h=d.get("gpu_h"))
    trained = {a: c for a, c in cells.items() if c.get("status") == "COMPLETED"}
    best_arm, best = None, -1.0
    for a, c in trained.items():
        if a == "baseline":
            continue
        if c["indist_min_recovered_at_0_9"] > best:
            best, best_arm = c["indist_min_recovered_at_0_9"], a
    baseline_ok = trained.get("baseline", {}).get("indist_min_recovered_at_0_9", 1.0) < 0.5
    if best >= 0.9:
        verdict = "RECOVERED"
    elif best >= 0.5:
        verdict = "PARTIAL"
    else:
        verdict = "NONE-RECOVER"
    return dict(cells=cells, best_arm=best_arm, best_indist_recovered=best,
                baseline_reproduces_nonconvergence=baseline_ok, verdict=verdict,
                control_valid=baseline_ok)


# ---------------------------------------------------------------------------
# Self-test (CPU; executed kill-proofs)
# ---------------------------------------------------------------------------

def _self_test():
    torch.manual_seed(0)
    cfg = ot.BankConfig()
    gen = torch.Generator().manual_seed(0)
    b = ot.sample_train_batch(cfg, 8, gen, "cpu", axis_b_frac=0.5)

    # t1 CLOSED-FORM: α=0 recover forward is BIT-IDENTICAL to the plain contender
    torch.manual_seed(1)
    rm = NCRBankRecoverModel(); rm._ln_alpha = 0.0
    plain = om.NCRBankModel()
    plain.load_state_dict(rm.state_dict())     # same weights (no new params)
    with torch.no_grad():
        pr, _ = rm(b); pp, _ = plain(b)
    assert torch.equal(pr, pp), f"t1 FAILED: α=0 recover != plain (maxdiff {(pr-pp).abs().max()})"
    # kill-proof: α>0 MUST change the output (else the LN blend is a dead no-op)
    rm._ln_alpha = 1.0
    with torch.no_grad():
        pr1, _ = rm(b)
    assert not torch.equal(pr1, pp), "t1 kill-proof FAILED: α=1 did not change output"
    print(f"t1 PASS: α=0 bit-identical to plain contender; α=1 changes output "
          f"(maxdiff {(pr1-pp).abs().max():.4f}) -- LN blend is live and eval-inert at α=0")

    # t2 SCHEDULES: correct at boundaries, and NON-recover arms are inert
    T = 80000
    assert abs(lr_at(WARMUP_STEPS, T, "warmup") - BASE_LR) < 1e-12
    assert lr_at(1, T, "warmup") < BASE_LR and lr_at(T, T, "warmup") < BASE_LR
    assert lr_at(500, T, "baseline") == BASE_LR
    assert ln_alpha_at(0, T, "earlyln") == 1.0 and ln_alpha_at(T // 2, T, "earlyln") == 0.0
    assert ln_alpha_at(100, T, "baseline") == 0.0
    assert axis_b_frac_at(0, T, "curriculum") == 0.0 and axis_b_frac_at(T // 2, T, "curriculum") == 0.5
    assert axis_b_frac_at(100, T, "baseline") == 0.5
    # kill-proof: earlyln α must MONOTONE-decrease (not constant)
    assert ln_alpha_at(0, T, "earlyln") > ln_alpha_at(T // 4, T, "earlyln") > ln_alpha_at(T // 2 - 1, T, "earlyln")
    print("t2 PASS: LR/ln_alpha/axis_b_frac schedules correct at boundaries; "
          "non-recover arms inert; earlyln α monotone-decreasing")

    # t3 END-TO-END all 4 arms (the §7e 'all arms' lesson), 4 steps CPU.
    # rmtree first so the resume-skip cache can never make t3 assert against
    # stale COMPLETED jsons instead of a genuine fresh run (§8.8a audit NIT-1).
    import shutil
    _t3_dir = "/tmp/ncropbank_recover_selftest"
    shutil.rmtree(_t3_dir, ignore_errors=True)
    for arm in RECOVER_ARMS:
        rec = run_recover_cell(arm, seed=0, steps=4, device="cpu", outdir=_t3_dir)
        assert rec["status"] == "COMPLETED", (arm, rec.get("status"))
        assert rec["blank_out"]["passed"], (arm, rec["blank_out"])
    print(f"t3 PASS: end-to-end micro cell COMPLETED for all {len(RECOVER_ARMS)} arms "
          f"({', '.join(RECOVER_ARMS)}) with blank-out P=1 holding")

    # t4 EVAL PURITY: after training, the recover model's eval read is the EXACT
    # pure-matmul read (α forced 0) -- confirm eval_read matches the fp64 power
    torch.manual_seed(2)
    m = NCRBankRecoverModel(); m.eval(); m._ln_alpha = 0.0
    Zb = torch.randn(3, om.R_PIN, om.D_PIN, om.D_PIN, dtype=torch.float64) * 0.4
    q = torch.randn(3, 5, om.D_PIN, dtype=torch.float64)
    ref = q
    for _ in range(21):
        ref = torch.einsum("bij,bqj->bqi", Zb[:, 0], ref)
    ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    o = NCRBankRecoverModel.eval_read(Zb, q, 0, 21)["o"]     # inherited exact read
    assert torch.all((o * ref).sum(-1) > 1 - 1e-9), "t4 FAILED: eval read not exact"
    print("t4 PASS: post-train eval read is the inherited EXACT pure-matmul read "
          "(matches fp64 power at h=21) -- exactness axis preserved")

    print("\nncr_opbank_recover self-test: ALL 4/4 PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cell", choices=list(RECOVER_ARMS), default=None)
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=80000)
    ap.add_argument("--train-batch", type=int, default=256)
    ap.add_argument("--ceiling-gpuh", type=float, default=2.0)
    ap.add_argument("--outdir", type=str, default=os.path.join(_HERE, "results_opbank_recover"))
    ap.add_argument("--stop-file", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.smoke:
        for arm in RECOVER_ARMS:
            r = run_recover_cell(arm, 0, 4, args.device, f"/tmp/ncropbank_recover_smoke_{args.device.replace(':','_')}")
            assert r["status"] == "COMPLETED", (arm, r.get("status"))
        print(f"ncr_opbank_recover smoke PASSED (4-step {args.device} cells, all 4 arms)")
        return
    if args.harvest:
        h = harvest(args.outdir)
        print(json.dumps(h, indent=1, default=float))
        return
    if args.cell:
        rec = run_recover_cell(args.cell, args.seed, args.steps, args.device, args.outdir,
                               args.stop_file, args.ceiling_gpuh, args.train_batch)
        print(f"RECOVER CELL {args.cell} status={rec.get('status')}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
