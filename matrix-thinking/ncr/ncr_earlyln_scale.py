"""NCR EARLY-LN K-SCALING -- NOVEL_ARCH_WATERFALL.md S11. ADDITIVE build:
imports ncr_task/ncr_models/ncr_spectral/run_ncr VERBATIM, modifies none of
them (the only shared-file edit anywhere in this program is the additive
GRIDS[24] entry landed directly in ncr_task.py, S11's pinned exception).

Question (S11, verdict-first framing). S9.10's write-capacity diagnostic
left K=15/16 DEAD on the plain single-relation recipe (n=1, trainability-
confounded -- the model never left the loss plateau, A_eff_rank collapsed to
O(1)). S8.9 independently RECOVERED convergence at R=3 (operator bank) via
a parameter-free inter-hop LayerNorm blended into the TRAIN read with
weight alpha annealed 1.0->0.0 over the first half of training (EVAL always
the inherited pure-matmul exact read -- exactness never compromised). This
module applies that SAME recipe to the single-relation K-scaling axis and
asks TWO separately-scored questions per K in {14, 15, 16, 24}, n=4 seeds:

  GATE 1 (CONVERGENCE): does earlyln reach in-dist (h=1,2,3) recovered@0.9
    (min over the 3 depths) >= 0.9, with A_eff_rank climbing toward K (bar:
    mean A_eff_rank >= 0.9*K)? Per-cell: CONVERGED / PARTIAL ([0.5,0.9)) /
    DEAD (<0.5). Per-K rate = #CONVERGED / 4; label CONVERGED-ROBUST
    (>=3/4) / CONVERGED-PARTIAL (1-2/4) / TRAINABILITY-DEAD (0/4).

  GATE 2 (FAR-DEPTH), scored ONLY on CONVERGED cells: recovered@0.9 at the
    cell's own pinned h* (ncr_task.GRIDS[K]["h_star"]) and along the ladder,
    banded per the standing S3.2a bands (HOLD>=0.9 / DEGRADED(0.5,0.9) /
    FAIL<=0.5).

NCREarlyLNModel is an nn.Module subclass of ncr_models.NCRModel overriding
ONLY forward() with the per-hop LN blend; encode()/eval_read()/arm are
INHERITED UNCHANGED, so every existing run_ncr.py instrument (z_dump, the
ncr_spectral deep probe, Axis-C lock, trust screen, blank_out_check,
eval_cell) runs against it verbatim post-training -- exactly the pattern
ncr_opbank_recover.py already established and had independently audited for
the operator-bank (R=3) case, generalized here to R=1.

EVAL-PURITY (the S8.8 safeguard, re-verified for THIS forward pass, not
inherited): after training, `model._ln_alpha` is forced to 0.0 and every
downstream instrument uses the inherited, UNMODIFIED binexp_read/loop_read
-- "recovery" at any depth can therefore never be an LN artifact. Verified
both as a closed-form CPU self-test (t2 below) and, per S11's launch gate,
as an independent opus audit kill-proof before deploy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
for p in (CHAPTER2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import task_e as te                     # noqa: E402 (verbatim)
import ncr_task as nt                   # noqa: E402 (verbatim; GRIDS[24] additive entry)
import ncr_models as nm                 # noqa: E402 (verbatim)
import ncr_spectral as ns               # noqa: E402 (verbatim)
import run_ncr as rn                    # noqa: E402 (verbatim -- instrument reuse)

RUNNER_TAG = "ncr_earlyln_scale_v1"
BASE_LR = rn.TRAIN_LR                   # 3e-4, unchanged from every wave-1/S9 cell
STEPS_DEFAULT = 80_000                  # S9.10's own per-cell budget, unchanged
TRAIN_BATCH = rn.TRAIN_BATCH            # 256

# S11 grid: (K, d, h) per the pre-registered convention (spare-probe for
# K=14/15, Condition A / proportional-headroom d=2K for K=16/24).
GRID_SHAPES = {
    14: dict(d=16, h=64),
    15: dict(d=16, h=64),
    16: dict(d=32, h=64),
    24: dict(d=48, h=64),
    # Queue-system K-ladder extension (2026-07-11, matrix-thinking/queue/):
    # additive only, K=14/15/16/24 above byte-identical. Condition-A
    # proportional-headroom convention (d=2K, h=64) applied verbatim, the
    # SAME convention already pinned for K=16/24 -- see ncr_task._gen_grid
    # for the matching GRIDS[K] extension (same K set).
    20: dict(d=40, h=64),
    32: dict(d=64, h=64),
    48: dict(d=96, h=64),
    64: dict(d=128, h=64),
    96: dict(d=192, h=64),
    128: dict(d=256, h=64),
    192: dict(d=384, h=64),
    256: dict(d=512, h=64),
}
DEFAULT_SEEDS = (0, 1, 2, 3)
CONVERGED_INDIST_BAR = 0.9
PARTIAL_INDIST_BAR = 0.5
AEFF_RANK_FRAC_BAR = 0.9                # mean A_eff_rank >= 0.9*K


# ---------------------------------------------------------------------------
# The model: NCRModel + a train-time-only, parameter-free inter-hop LN blend
# ---------------------------------------------------------------------------

class NCREarlyLNModel(nm.NCRModel):
    """S11: NCRModel with an optional inter-hop LayerNorm blended into the
    TRAIN read, weight `self._ln_alpha` (default 0.0). At alpha=0 forward()
    is bit-identical to NCRModel.forward() (asserted in the self-test --
    mirrors compose()'s own per-step loop exactly, single relation, no r1/r2
    split). `arm` stays 'ncr' so every run_ncr.py instrument (arm_state,
    arm_reads_from_state, eval_cell, blank_out_check) routes through the
    pure-matmul ncr branch unchanged -- encode()/eval_read() are inherited,
    untouched."""

    def __init__(self, d: int = nm.D_PIN, h: int = nm.ENC_H):
        super().__init__(d=d, h=h)
        self._ln_alpha = 0.0     # runner sets this per step for the earlyln schedule

    def _blend(self, stepped: torch.Tensor) -> torch.Tensor:
        a = self._ln_alpha
        if a <= 0.0:
            return stepped        # bit-identical to the plain contender
        return a * F.layer_norm(stepped, (self.d,)) + (1.0 - a) * stepped

    def forward(self, batch: dict):
        """TRAIN path only (h <= 3). Mirrors model_e.MatrixCompositionModel
        .compose's per-step loop EXACTLY (torch.where masking over t in
        1..max_h), with one extra line: the optional LN blend after each
        matvec. At alpha=0 the two are bit-identical by construction."""
        Z = self.encode(batch["keys"], batch["values"])
        cur = batch["query_keys"]
        hops = batch["hops"]
        result = torch.where(hops.unsqueeze(-1) == 0, cur, torch.zeros_like(cur))
        max_h = int(hops.max().item()) if hops.numel() > 0 else 0
        for t in range(1, max_h + 1):
            stepped = self._blend(torch.einsum("bij,bqj->bqi", Z, cur))
            cur = stepped
            mask = (hops == t).unsqueeze(-1)
            result = torch.where(mask, cur, result)
        return result, Z


# ---------------------------------------------------------------------------
# Per-step schedule (pure function; the S8.7-pinned earlyln shape, generalized
# to an arbitrary step budget -- identical formula, no drift)
# ---------------------------------------------------------------------------

def ln_alpha_at(step: int, total: int) -> float:
    """1.0 -> 0.0 linearly over the first half of training, 0.0 thereafter
    (S8.7/S8.9's pinned earlyln schedule, verbatim)."""
    half = total // 2
    if half <= 0 or step >= half:
        return 0.0
    return 1.0 - step / half


# ---------------------------------------------------------------------------
# Training (mirrors run_ncr.train_cell; adds the ln_alpha schedule; no
# mid-cell checkpoint/resume -- cells are short (~0.5 GPU-h), whole-cell
# skip-if-COMPLETED is the resume-safety unit, the ncr_opbank_recover.py
# precedent)
# ---------------------------------------------------------------------------

def train_earlyln_cell(model, cfg, device: str, steps: int, seed: int,
                       stop_file: str, ceiling_s: float,
                       log_every: int = 500) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=BASE_LR)
    model.train()
    n_skipped, loss_hist = 0, []
    t0 = time.time()
    for step in range(1, steps + 1):
        model._ln_alpha = ln_alpha_at(step, steps)
        b = te.sample_batch(cfg, TRAIN_BATCH, gen, hop_set=cfg.H_train,
                            device=device, assert_injective=(step == 1))
        pred, _ = model(b)
        loss = rn.cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            loss_hist.append([step, float(loss.item())])
            print(f"  [K-earlyln] step {step:6d}  loss {loss.item():.4f}  "
                  f"ln_a {model._ln_alpha:.2f}  {elapsed:.0f}s", flush=True)
            if rn.stop_requested(stop_file):
                print("  STOP file seen -- exiting 3"); sys.exit(3)
            if elapsed > ceiling_s:
                return dict(status="ABORTED-BUDGET", step=step, elapsed_s=elapsed,
                           ceiling_s=ceiling_s, n_skipped_steps=n_skipped, loss_history=loss_hist)
    model._ln_alpha = 0.0     # final model IS the exact pure-matmul contender
    return dict(status="COMPLETED", step=steps, elapsed_s=time.time() - t0,
               ceiling_s=ceiling_s, n_skipped_steps=n_skipped, loss_history=loss_hist)


def cell_id(K: int, seed: int) -> str:
    return f"earlyln_K{K}_s{seed}"


def run_earlyln_cell(K: int, seed: int, steps: int, device: str, outdir: str,
                     stop_file: str = "", ceiling_gpuh: float = 2.0) -> dict:
    """Full pipeline for one (K, seed) cell: train (earlyln schedule) then
    the IDENTICAL post-train instrument sequence run_ncr.run_cell uses for
    its 'ncr' arm branch (z_dump -> deep probe -> Axis-C lock -> trust
    screen -> blank_out_check -> eval_cell), called against this module's
    NCREarlyLNModel instance with _ln_alpha forced to 0 first."""
    shape = GRID_SHAPES[K]
    d_eff, h_eff = shape["d"], shape["h"]
    os.makedirs(outdir, exist_ok=True)
    cid = cell_id(K, seed)
    out_path = os.path.join(outdir, f"{cid}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "COMPLETED":
            print(f"  [{cid}] already COMPLETED -- skipping (resume-safe)", flush=True)
            return prev

    cfg = nt.claim_config(K, d=d_eff)
    torch.manual_seed(seed)
    model = NCREarlyLNModel(d=d_eff, h=h_eff).to(device)
    rec = dict(cell_id=cid, K=K, d=d_eff, h=h_eff, seed=seed, runner_tag=RUNNER_TAG,
              git_commit=rn.git_commit(), params=nm.n_params(model),
              train_batch=TRAIN_BATCH, host=socket.gethostname(), device=device,
              torch_version=torch.__version__,
              started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ceiling_s = ceiling_gpuh * 3600.0 if device == "cuda" else float("inf")
    t0 = time.time()

    tr = train_earlyln_cell(model, cfg, device, steps, seed, stop_file, ceiling_s)
    rec["train"] = tr
    if tr["status"] != "COMPLETED":
        rec["status"] = tr["status"]
        rec["elapsed_s"] = time.time() - t0
        rn.atomic_write_json(out_path, rec)
        return rec

    model.eval()
    model._ln_alpha = 0.0    # EVAL-PURITY (S8.8 safeguard, re-verified this pass):
                             # forced 0 -- downstream reads are the inherited,
                             # unmodified pure-matmul binexp_read/loop_read.

    zd = rn.z_dump(model, cfg, device, seed)
    rec["z_dump"] = zd
    all_h = [p.h for p in nt.eval_points(K, d=d_eff)]
    probe = ns.analyze_zdump_arrays(zd["Z"], zd["z_ideal"], all_h)
    rec["deep_probe"] = dict(
        phase_resid_max_per_example=[ex["phase_resid_max"] for ex in probe["per_example"]],
        phase_resid_max_mean=probe["phase_resid_max_mean"],
        c_star_per_example=[ex["c_star"] for ex in probe["per_example"]],
        scale_corrected_residual=[ex["scale_corrected_residual"] for ex in probe["per_example"]],
        A_eff_rank=[ex["A_eff_rank"] for ex in probe["per_example"]])
    lock_path = os.path.join(outdir, f"{cid}.axis_c_lock.json")
    write_axis = ns.write_axis_c_lock(lock_path, cid, K, probe)
    lock_content = ns.verify_axis_c_lock(lock_path)
    rec["axis_c_lock_sha256"] = write_axis["lock_sha256"]

    screens = [ns.trust_screen(np.array(ex["blocks"]["A"]), np.array(ex["blocks"]["B"]),
                               np.array(ex["blocks"]["C"]), np.array(ex["blocks"]["D"]), all_h)
              for ex in probe["per_example"]]
    trust_per_h = {}
    for hh in screens[0]["per_h"]:
        ts = [s["per_h"][hh]["T"] for s in screens]
        t_worst = max(float("inf") if t == "inf" else float(t) for t in ts)
        trust_per_h[hh] = dict(T=t_worst if math.isfinite(t_worst) else "inf",
                               rule_trusted=bool(all(s["per_h"][hh]["rule_trusted"] for s in screens)))
    rec["trust_screen"] = dict(tau=ns.TAU, per_h=trust_per_h)

    rec["blank_out"] = rn.blank_out_check(model, cfg, device, seed)
    assert rec["blank_out"]["passed"], f"blank-out FAILED for {cid}: {rec['blank_out']}"

    rec["eval"] = rn.eval_cell(model, cfg, device, seed, K, lock_content, trust_per_h, d=d_eff)
    rec["elapsed_s"] = time.time() - t0
    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
    rec["status"] = "COMPLETED"
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rn.atomic_write_json(out_path, rec)
    print(f"  [{cid}] COMPLETED in {rec['elapsed_s']:.0f}s -> {out_path}", flush=True)
    return rec


# ---------------------------------------------------------------------------
# Harvest -- Gate 1 (convergence, per K rate) + Gate 2 (far-depth, converged
# cells only), verdict-first per S11
# ---------------------------------------------------------------------------

def _cell_gate1(rec: dict, K: int) -> dict:
    pts = [e for e in rec["eval"]["points"] if e["component"] == "train_support"]
    indist_min = min(float(e["reads"]["binexp"]["recovered_frac@0.9"]) for e in pts)
    aer = rec["deep_probe"]["A_eff_rank"]
    aer_mean = sum(aer) / len(aer) if aer else 0.0
    if indist_min >= CONVERGED_INDIST_BAR and aer_mean >= AEFF_RANK_FRAC_BAR * K:
        v = "CONVERGED"
    elif indist_min >= PARTIAL_INDIST_BAR:
        v = "PARTIAL"
    else:
        v = "DEAD"
    return dict(indist_min_recovered=indist_min, A_eff_rank_mean=aer_mean,
               A_eff_rank_bar=AEFF_RANK_FRAC_BAR * K, verdict=v)


def _band(x: float) -> str:
    if x >= 0.9:
        return "HOLD"
    if x > 0.5:
        return "DEGRADED"
    return "FAIL"


def _cell_gate2(rec: dict, K: int) -> dict | None:
    hstar = nt.GRIDS[K]["h_star"]
    pts = [e for e in rec["eval"]["points"] if e["component"] == "h_star"
           or (e["component"] == "ladder" and e["h"] == hstar)]
    if not pts:
        return None
    r = float(pts[0]["reads"]["binexp"]["recovered_frac@0.9"])
    return dict(h_star=hstar, recovered_at_h_star=r, band=_band(r),
               failure_front_h=rec["eval"].get("failure_front_h"))


def discover_seeds_by_K(outdir: str) -> dict:
    """Audit MAJOR-2 fix: derive each K's ACTUAL seed set from the cell
    JSONs present on disk, rather than trusting a uniform --seeds value
    that can silently mismatch a per-K trim (e.g. K=24 dropped to n=3 by
    the §11 budget-fit rule while K=14/15/16 stayed at n=4 -- a bare
    `--harvest` with the default uniform seed list would otherwise treat
    K=24's missing 4th seed as MISSING-not-trimmed, landing at rate=3/4
    and the WRONG gate_eligible=True). Globs `earlyln_K{K}_s{seed}.json`
    (excluding `.axis_c_lock.json` siblings) for every K in GRID_SHAPES."""
    import glob
    import re
    seeds_by_K = {K: [] for K in GRID_SHAPES}
    for K in GRID_SHAPES:
        for p in sorted(glob.glob(os.path.join(outdir, f"{cell_id(K, '*')}.json"))):
            if p.endswith(".axis_c_lock.json"):
                continue
            m = re.search(rf"earlyln_K{K}_s(\d+)\.json$", p)
            if m:
                seeds_by_K[K].append(int(m.group(1)))
        seeds_by_K[K] = tuple(sorted(seeds_by_K[K]))
    return seeds_by_K


def harvest(outdir: str, seeds_by_K: dict) -> dict:
    per_K = {}
    for K, seeds in seeds_by_K.items():
        cells = {}
        for seed in seeds:
            p = os.path.join(outdir, f"{cell_id(K, seed)}.json")
            if not os.path.exists(p):
                cells[seed] = dict(status="MISSING")
                continue
            rec = json.load(open(p))
            if rec.get("status") != "COMPLETED":
                cells[seed] = dict(status=rec.get("status"))
                continue
            g1 = _cell_gate1(rec, K)
            entry = dict(status="COMPLETED", gate1=g1,
                        train_final_loss=rec["train"]["loss_history"][-1][1],
                        gpu_h=rec.get("gpu_h"))
            if g1["verdict"] == "CONVERGED":
                entry["gate2"] = _cell_gate2(rec, K)
            cells[seed] = entry
        n_converged = sum(1 for c in cells.values() if c.get("gate1", {}).get("verdict") == "CONVERGED")
        n_seeds = len(seeds)
        rate = n_converged / n_seeds if n_seeds else 0.0
        # S9.5 hard-rule bakein, reused verbatim (S9.2 item 2 / S11's own
        # pinned budget-fit trim rule): a sub-4-seed rung is NEVER gated as
        # CONVERGED-ROBUST/TRAINABILITY-DEAD and NEVER folded into the
        # pooled worst-of-K label -- it is disclosed data only. This is the
        # planned fate of a trimmed K=24 (the ONLY rung this program's own
        # pre-registration permits to drop below n=4).
        gate_eligible = n_seeds >= 4
        if not gate_eligible:
            gate1_label = f"SUB4-DISCLOSED-ONLY(n={n_seeds})"
        elif rate >= 0.75:
            gate1_label = "CONVERGED-ROBUST"
        elif n_converged >= 1:
            gate1_label = "CONVERGED-PARTIAL"
        else:
            gate1_label = "TRAINABILITY-DEAD"
        converged_bands = [c["gate2"]["band"] for c in cells.values()
                           if c.get("gate2") is not None]
        # §11 pins the MEDIAN of converged cells' h*-recovery, not "every
        # converged cell must individually HOLD" -- audit MAJOR-1: an
        # earlier version banded per-cell and required ALL to be HOLD,
        # which is strictly stricter than the pre-registration and could
        # under-claim SCALES on a single outlier seed. Fixed here; t8 below
        # is the kill-proof (a [0.95,0.95,0.95,0.30] converged set must
        # still read SCALES, median 0.95 = HOLD, even though one cell FAILs).
        converged_recovered = sorted(
            c["gate2"]["recovered_at_h_star"] for c in cells.values()
            if c.get("gate2") is not None)
        median_recovered = (
            converged_recovered[len(converged_recovered) // 2] if len(converged_recovered) % 2
            else (converged_recovered[len(converged_recovered) // 2 - 1]
                  + converged_recovered[len(converged_recovered) // 2]) / 2
        ) if converged_recovered else None
        median_band = _band(median_recovered) if median_recovered is not None else None
        if not gate_eligible:
            ladder_verdict = None    # disclosed-only; never scored (see above)
        elif gate1_label == "CONVERGED-ROBUST" and median_band is not None:
            ladder_verdict = ("SCALES" if median_band == "HOLD"
                              else "CONVERGES-BUT-FAR-DEPTH-LIMITED")
        else:
            ladder_verdict = "TRAINABILITY-STILL-LIMITED"
        per_K[str(K)] = dict(n_seeds=n_seeds, n_converged=n_converged, rate=rate,
                             gate_eligible=gate_eligible, gate1_label=gate1_label,
                             gate2_bands=converged_bands,
                             gate2_median_recovered_at_h_star=median_recovered,
                             gate2_median_band=median_band,
                             ladder_verdict=ladder_verdict, cells=cells)

    # K=14 is a continuity check, not pooled; sub-4-seed rungs are disclosed
    # data only (gate_eligible=False -> ladder_verdict=None) and are
    # EXCLUDED from the pooled worst-of-K label -- never silently treated
    # as a pass by omission (the label is None, not a default-good value).
    scored_Ks = [K for K in seeds_by_K
                if K != 14 and per_K[str(K)]["gate_eligible"]]
    excluded_sub4_Ks = [K for K in seeds_by_K
                        if K != 14 and not per_K[str(K)]["gate_eligible"]]
    order = {"SCALES": 0, "CONVERGES-BUT-FAR-DEPTH-LIMITED": 1, "TRAINABILITY-STILL-LIMITED": 2}
    pooled = None
    if scored_Ks:
        worst = max((per_K[str(K)]["ladder_verdict"] for K in scored_Ks), key=lambda v: order[v])
        pooled = worst
    return dict(per_K=per_K, pooled_verdict_K_gt_14=pooled,
               scored_Ks=scored_Ks, excluded_sub4_Ks=excluded_sub4_Ks,
               continuity_K=14 if 14 in seeds_by_K else None)


# ---------------------------------------------------------------------------
# CPU self-test (executed kill-proofs)
# ---------------------------------------------------------------------------

def _self_test():
    torch.manual_seed(0)
    cfg = nt.claim_config(14, d=16)
    gen = torch.Generator().manual_seed(0)
    b = te.sample_batch(cfg, 8, gen, hop_set=cfg.H_train, device="cpu")

    # t1 CLOSED-FORM: alpha=0 earlyln forward is BIT-IDENTICAL to plain NCRModel
    torch.manual_seed(1)
    em = NCREarlyLNModel(d=16, h=64); em._ln_alpha = 0.0
    plain = nm.NCRModel(d=16, h=64)
    plain.load_state_dict(em.state_dict())      # same weights, no new params
    with torch.no_grad():
        pe, _ = em(b); pp, _ = plain(b)
    assert torch.equal(pe, pp), f"t1 FAILED: alpha=0 earlyln != plain (maxdiff {(pe-pp).abs().max()})"
    em._ln_alpha = 1.0
    with torch.no_grad():
        pe1, _ = em(b)
    assert not torch.equal(pe1, pp), "t1 kill-proof FAILED: alpha=1 did not change output"
    assert nm.n_params(em) == nm.n_params(plain), "earlyln must add ZERO parameters"
    print(f"t1 PASS: alpha=0 bit-identical to plain NCRModel (0 new params); "
          f"alpha=1 changes output (maxdiff {(pe1-pp).abs().max():.4f}) -- LN blend live, eval-inert at 0")

    # t2 EVAL-PURITY (S8.8 safeguard): post-"train" (alpha forced 0), the
    # inherited eval_read (binexp) matches the literal fp64 power exactly --
    # kill-proof: leaving alpha>0 during eval WOULD change binexp's *input*
    # Z is irrelevant (eval_read never calls forward()) -- eval_read is a
    # pure function of Z, untouched by _ln_alpha by construction. Verified
    # directly: eval_read's output is bit-for-bit independent of _ln_alpha.
    torch.manual_seed(2)
    m = NCREarlyLNModel(d=16, h=64); m.eval()
    Zb = torch.randn(3, 16, 16, dtype=torch.float64) * 0.4
    q = torch.randn(3, 5, 16, dtype=torch.float64)
    ref = q
    for _ in range(21):
        ref = torch.einsum("bij,bqj->bqi", Zb, ref)
    ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-30)
    for alpha_probe in (0.0, 1.0, 0.7):   # kill-proof: eval_read ignores _ln_alpha entirely
        m._ln_alpha = alpha_probe
        o = NCREarlyLNModel.eval_read(Zb, q, 21, "binexp")["o"]
        assert torch.all((o * ref).sum(-1) > 1 - 1e-9), (alpha_probe, "t2 FAILED: eval read not exact")
    print("t2 PASS: eval_read (inherited binexp_read) matches the fp64 literal power at h=21 "
          "REGARDLESS of _ln_alpha (eval reads are pure functions of Z, structurally immune to "
          "the LN blend) -- exactness axis provably preserved, not just disciplined by convention")

    # t3 SCHEDULE: correct at boundaries, monotone-decreasing
    T = 80000
    assert ln_alpha_at(0, T) == 1.0
    assert ln_alpha_at(T // 2, T) == 0.0
    assert ln_alpha_at(T, T) == 0.0
    assert ln_alpha_at(0, T) > ln_alpha_at(T // 4, T) > ln_alpha_at(T // 2 - 1, T)
    print("t3 PASS: ln_alpha_at schedule correct at boundaries (1.0 at step 0, 0.0 from "
          "the half-point on), monotone-decreasing over the annealed half")

    # t4 GRIDS[24] regression + new-key invariants (independently re-typed
    # snapshot of the OTHER four keys -- an in-place corruption of any of
    # them cannot pass by comparing the module against itself)
    _G8 = dict(ladder=(5, 13, 21, 29, 61, 125, 253, 509, 1021), h_star=61,
              sweep=tuple(range(57, 65)), cost_probe=(2**10+5, 2**14+5, 2**17+5, 2**20+5),
              ladder_residue=5)
    _G12 = dict(ladder=(9, 21, 45, 93, 189, 381, 765, 1533), h_star=57,
               sweep=tuple(range(49, 61)), cost_probe=(), ladder_residue=9)
    _G14 = dict(ladder=(11, 25, 53, 109, 221, 445, 893, 1789), h_star=109,
               sweep=tuple(range(99, 113)), cost_probe=(), ladder_residue=11)
    _G15 = dict(ladder=(12, 27, 57, 117, 237, 477, 957, 1917), h_star=117,
               sweep=tuple(range(106, 121)), cost_probe=(), ladder_residue=12)
    _G16 = dict(ladder=(13, 29, 61, 125, 253, 509, 1021, 2045), h_star=125,
               sweep=tuple(range(113, 129)), cost_probe=(), ladder_residue=13)
    assert nt.GRIDS[8] == _G8 and nt.GRIDS[12] == _G12 and nt.GRIDS[14] == _G14 \
        and nt.GRIDS[15] == _G15 and nt.GRIDS[16] == _G16, \
        "t4 FAILED: an existing GRIDS key changed -- S11's GRIDS[24] add must be additive-only"
    g24 = nt.GRIDS[24]
    assert g24["ladder_residue"] == 21 and g24["h_star"] == 189
    assert all(h % 24 == 21 for h in g24["ladder"])
    assert len(g24["sweep"]) == 24 and sorted(h % 24 for h in g24["sweep"]) == list(range(24))
    assert nt.residue_label(g24["h_star"] + 3, 24) == "identity"
    assert nt.residue_label(g24["ladder_residue"], 24) == "novel"
    print("t4 PASS: GRIDS[8]/[12]/[14]/[15]/[16] byte-identical (GRIDS[24] add is additive-only); "
          "GRIDS[24] residue/identity/sweep invariants hold")

    # t4b QUEUE-SYSTEM K-LADDER EXTENSION (2026-07-11): same additive-only
    # discipline, one more block of K's (20/32/48/64/96/128/192/256). GRIDS[8]
    # through [24] re-checked byte-identical (t4's own dicts, no drift from
    # this module's own edit); every new K's ladder/h_star/sweep/residue
    # invariants hold via the SAME formula nt._gen_grid implements (verified
    # against nt.GRIDS directly here, not against _gen_grid itself, so this
    # catches a divergence in EITHER the formula or its call site); GRID_SHAPES
    # d=2K/h=64 convention verified for every new key.
    assert nt.GRIDS[8] == _G8 and nt.GRIDS[12] == _G12 and nt.GRIDS[14] == _G14 \
        and nt.GRIDS[15] == _G15 and nt.GRIDS[16] == _G16 and nt.GRIDS[24] == g24, \
        "t4b FAILED: an existing GRIDS key changed by the K-ladder extension"
    for _K in (20, 32, 48, 64, 96, 128, 192, 256):
        _g = nt.GRIDS[_K]
        assert _g["ladder_residue"] == _K - 3, (_K, _g)
        assert _g["h_star"] == 8 * _K - 3, (_K, _g)
        assert all(h % _K == _g["ladder_residue"] for h in _g["ladder"]), (_K, _g)
        assert len(_g["sweep"]) == _K and sorted(h % _K for h in _g["sweep"]) == list(range(_K)), (_K, _g)
        assert nt.residue_label(_g["h_star"] + 3, _K) == "identity", _K
        assert nt.residue_label(_g["ladder_residue"], _K) == "novel", _K
        assert GRID_SHAPES[_K] == dict(d=2 * _K, h=64), (_K, "GRID_SHAPES must be d=2K,h=64")
    print("t4b PASS: GRIDS[8..24] byte-identical after the K-ladder extension; "
          "K in {20,32,48,64,96,128,192,256} residue/identity/sweep invariants hold; "
          "GRID_SHAPES d=2K/h=64 convention verified for every new key")

    # t5 END-TO-END all 4 K shapes, tiny eval grid (mirrors ncr_wcap_selftest
    # t05's tiny-grid pattern so CPU cost stays trivial)
    import shutil
    _t5_dir = "/tmp/ncr_earlyln_scale_selftest"
    shutil.rmtree(_t5_dir, ignore_errors=True)
    real_pts = nt.eval_points
    real_batches, real_bs = rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE

    def tiny_points(K_, d=nt.D_PIN, _real=real_pts):
        g = nt.GRIDS[K_]
        keep = {1, 2, 3, g["ladder"][0], g["h_star"]}
        return [p for p in _real(K_, d=d) if p.h in keep and p.component != "cost_probe"]
    try:
        nt.eval_points = tiny_points
        rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE = 2, 16
        for K, shape in GRID_SHAPES.items():
            rec = run_earlyln_cell(K, seed=0, steps=4, device="cpu", outdir=_t5_dir)
            assert rec["status"] == "COMPLETED", (K, rec.get("status"))
            assert rec["blank_out"]["passed"], (K, rec["blank_out"])
            assert rec["params"] == 40 * shape["h"]**2 + 4 * shape["d"] * shape["h"] \
                + 46 * shape["h"] + shape["d"], (K, "param formula mismatch")
    finally:
        nt.eval_points = real_pts
        rn.EVAL_BATCHES, rn.EVAL_BATCH_SIZE = real_batches, real_bs
    print(f"t5 PASS: end-to-end micro cell (train->instruments->eval->JSON) COMPLETED for all "
          f"{len(GRID_SHAPES)} K shapes ({sorted(GRID_SHAPES)}), blank-out P=1 holding, "
          f"params match the S9.3-corrected closed form")

    # t6 HARVEST kill-proof: a synthetic 2-cell (1 CONVERGED @ HOLD, 1 DEAD)
    # K reads CONVERGED-PARTIAL / gate2 HOLD -- exercise both branches
    _t6_dir = "/tmp/ncr_earlyln_scale_harvest_selftest"
    shutil.rmtree(_t6_dir, ignore_errors=True)
    os.makedirs(_t6_dir)

    def _fake_cell(K, seed, indist, aer, hstar_rec, dir_=_t6_dir):
        pts = [dict(component="train_support", h=h,
                    reads=dict(binexp=dict(**{"recovered_frac@0.9": indist})))
              for h in (1, 2, 3)]
        pts.append(dict(component="h_star", h=nt.GRIDS[K]["h_star"],
                        reads=dict(binexp=dict(**{"recovered_frac@0.9": hstar_rec}))))
        rec = dict(status="COMPLETED", K=K, seed=seed,
                  train=dict(loss_history=[[1, 1.0], [10, 0.01]]),
                  deep_probe=dict(A_eff_rank=[aer, aer]),
                  eval=dict(points=pts, failure_front_h=None), gpu_h=0.01)
        with open(os.path.join(dir_, f"{cell_id(K, seed)}.json"), "w") as f:
            json.dump(rec, f)

    for s in DEFAULT_SEEDS:
        _fake_cell(16, s, indist=0.95, aer=15.0, hstar_rec=0.95)   # every seed HOLDS
    h6 = harvest(_t6_dir, {16: DEFAULT_SEEDS})
    assert h6["per_K"]["16"]["gate1_label"] == "CONVERGED-ROBUST", h6["per_K"]["16"]
    assert h6["per_K"]["16"]["ladder_verdict"] == "SCALES", h6["per_K"]["16"]
    assert h6["pooled_verdict_K_gt_14"] == "SCALES"

    shutil.rmtree(_t6_dir, ignore_errors=True)
    os.makedirs(_t6_dir)
    for s in DEFAULT_SEEDS:
        _fake_cell(16, s, indist=0.0, aer=1.5, hstar_rec=0.0)      # every seed DEAD
    h6b = harvest(_t6_dir, {16: DEFAULT_SEEDS})
    assert h6b["per_K"]["16"]["gate1_label"] == "TRAINABILITY-DEAD", h6b["per_K"]["16"]
    assert h6b["per_K"]["16"]["ladder_verdict"] == "TRAINABILITY-STILL-LIMITED"
    assert h6b["pooled_verdict_K_gt_14"] == "TRAINABILITY-STILL-LIMITED"

    shutil.rmtree(_t6_dir, ignore_errors=True)
    os.makedirs(_t6_dir)
    for s in DEFAULT_SEEDS:
        _fake_cell(16, s, indist=0.95, aer=15.0, hstar_rec=0.2)    # converges, far-depth FAILs
    h6c = harvest(_t6_dir, {16: DEFAULT_SEEDS})
    assert h6c["per_K"]["16"]["gate1_label"] == "CONVERGED-ROBUST"
    assert h6c["per_K"]["16"]["ladder_verdict"] == "CONVERGES-BUT-FAR-DEPTH-LIMITED", h6c["per_K"]["16"]
    assert h6c["pooled_verdict_K_gt_14"] == "CONVERGES-BUT-FAR-DEPTH-LIMITED"

    # pooled = worst-of: one SCALES K + one TRAINABILITY-STILL-LIMITED K -> pooled is the LIMITED one
    shutil.rmtree(_t6_dir, ignore_errors=True)
    os.makedirs(_t6_dir)
    for s in DEFAULT_SEEDS:
        _fake_cell(15, s, indist=0.95, aer=14.0, hstar_rec=0.95)
        _fake_cell(16, s, indist=0.0, aer=1.5, hstar_rec=0.0)
    h6d = harvest(_t6_dir, {15: DEFAULT_SEEDS, 16: DEFAULT_SEEDS})
    assert h6d["per_K"]["15"]["ladder_verdict"] == "SCALES"
    assert h6d["per_K"]["16"]["ladder_verdict"] == "TRAINABILITY-STILL-LIMITED"
    assert h6d["pooled_verdict_K_gt_14"] == "TRAINABILITY-STILL-LIMITED", (
        "t6 kill-proof FAILED: pooled label must take the WORST per-K label, never cherry-pick "
        f"got {h6d['pooled_verdict_K_gt_14']}")
    shutil.rmtree(_t6_dir, ignore_errors=True)
    print("t6 PASS: harvest Gate-1/Gate-2 classification correct on synthetic SCALES / "
          "TRAINABILITY-STILL-LIMITED / CONVERGES-BUT-FAR-DEPTH-LIMITED cells; pooled label "
          "kill-proof confirms it takes the WORST per-K label (no cherry-picking a good K)")

    # t7 SUB-4-SEED EXCLUSION kill-proof (S9.5 hard-rule bakein / S11 budget-fit
    # rule): a trimmed rung (n<4, e.g. a trimmed K=24) must NEVER be labeled
    # CONVERGED-ROBUST/TRAINABILITY-DEAD and must NEVER enter the pooled
    # worst-of-K label -- even when every one of its few seeds converges.
    os.makedirs(_t6_dir)
    for s in DEFAULT_SEEDS:
        _fake_cell(15, s, indist=0.95, aer=14.0, hstar_rec=0.95)   # K=15 SCALES, n=4
    for s in (0, 1, 2):
        _fake_cell(24, s, indist=0.95, aer=23.0, hstar_rec=0.95)   # K=24 trimmed to n=3, all "converge"
    h7 = harvest(_t6_dir, {15: DEFAULT_SEEDS, 24: (0, 1, 2)})
    assert h7["per_K"]["24"]["gate_eligible"] is False, h7["per_K"]["24"]
    assert h7["per_K"]["24"]["gate1_label"] == "SUB4-DISCLOSED-ONLY(n=3)", h7["per_K"]["24"]
    assert h7["per_K"]["24"]["ladder_verdict"] is None, h7["per_K"]["24"]
    assert 24 in h7["excluded_sub4_Ks"] and 24 not in h7["scored_Ks"], h7
    assert h7["pooled_verdict_K_gt_14"] == "SCALES", (
        "t7 kill-proof FAILED: pooled label must come from K=15 alone -- a trimmed, "
        f"all-converged K=24 must never enter it, got {h7['pooled_verdict_K_gt_14']}")
    shutil.rmtree(_t6_dir, ignore_errors=True)
    print("t7 PASS: a sub-4-seed rung (trimmed K=24, n=3, even with 3/3 'converged') is "
          "correctly excluded from gate1_label/ladder_verdict/pooled scoring -- disclosed "
          "data only, per S9.5's sub-4-seed hard rule, never a silent pass")

    # t8 MEDIAN-RULE kill-proof (independent audit MAJOR-1): §11 pins the
    # MEDIAN of converged cells' h*-recovery, not "every converged cell
    # must individually HOLD". [0.95,0.95,0.95,0.30] has median 0.95 = HOLD
    # -> must read SCALES even though one seed FAILs far-depth.
    os.makedirs(_t6_dir)
    for s, r in zip(DEFAULT_SEEDS, (0.95, 0.95, 0.95, 0.30)):
        _fake_cell(16, s, indist=0.95, aer=15.0, hstar_rec=r)
    h8 = harvest(_t6_dir, {16: DEFAULT_SEEDS})
    assert h8["per_K"]["16"]["gate2_median_recovered_at_h_star"] == 0.95, h8["per_K"]["16"]
    assert h8["per_K"]["16"]["ladder_verdict"] == "SCALES", (
        "t8 kill-proof FAILED (audit MAJOR-1 regression): median of "
        "[0.95,0.95,0.95,0.30] is 0.95 (HOLD) -- one FAILing seed among 4 "
        f"converged must not sink SCALES, got {h8['per_K']['16']}")
    shutil.rmtree(_t6_dir, ignore_errors=True)
    print("t8 PASS: ladder_verdict uses the MEDIAN of converged cells' h*-recovery per §11 "
          "(not 'every cell must individually HOLD') -- a single outlier FAIL among 4 "
          "converged, HOLD-median seeds still reads SCALES")

    # t9 SEED AUTO-DISCOVERY kill-proof (independent audit MAJOR-2): a
    # per-K trim (K=24 dropped to n=3 while K=15/16 stay n=4) must be
    # discovered from the JSONs on disk, not silently mismatched against a
    # stale uniform seed assumption (which would read K=24's missing 4th
    # seed as MISSING-not-trimmed and wrongly flip gate_eligible to True).
    _t9_dir = "/tmp/ncr_earlyln_scale_discover_selftest"
    shutil.rmtree(_t9_dir, ignore_errors=True)
    os.makedirs(_t9_dir)
    for s in DEFAULT_SEEDS:
        _fake_cell(15, s, indist=0.95, aer=14.0, hstar_rec=0.95, dir_=_t9_dir)
    for s in (0, 1, 2):     # K=24 trimmed to n=3 on disk
        _fake_cell(24, s, indist=0.95, aer=23.0, hstar_rec=0.95, dir_=_t9_dir)
    discovered = discover_seeds_by_K(_t9_dir)
    assert discovered[15] == (0, 1, 2, 3), discovered
    assert discovered[24] == (0, 1, 2), (
        "t9 kill-proof FAILED (audit MAJOR-2 regression): discover_seeds_by_K must read the "
        f"ACTUAL trimmed seed set (0,1,2) off disk, not assume a uniform default, got {discovered}")
    h9 = harvest(_t9_dir, discovered)
    assert h9["per_K"]["24"]["gate_eligible"] is False, (
        "t9 kill-proof FAILED: auto-discovered K=24 (n=3, trimmed) must stay sub-4-excluded, "
        f"got {h9['per_K']['24']}")
    shutil.rmtree(_t9_dir, ignore_errors=True)
    print("t9 PASS: discover_seeds_by_K correctly recovers a per-K trim (K=24 n=3 while "
          "K=15 stays n=4) from the JSONs on disk -- a stale uniform seed assumption can "
          "no longer silently mis-score a trimmed rung as gate-eligible")

    print("\nncr_earlyln_scale self-test: ALL 9/9 PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cell", action="store_true")
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--K", type=int, default=None, choices=sorted(GRID_SHAPES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--ceiling-gpuh", type=float, default=2.0)
    ap.add_argument("--outdir", type=str, default=os.path.join(_HERE, "results_earlyln_scale"))
    ap.add_argument("--stop-file", type=str, default="")
    ap.add_argument("--seeds", type=str, default=None,
                    help="override the auto-discovered seed list for EVERY K (rarely needed -- "
                         "--harvest auto-discovers each K's actual completed seeds from the JSONs "
                         "on disk by default, audit MAJOR-2 fix, so a per-K trim (e.g. K=24 dropped "
                         "to n=3) is never silently mismatched against a uniform assumption)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.smoke:
        for K in GRID_SHAPES:
            r = run_earlyln_cell(K, 0, 4, args.device,
                                 f"/tmp/ncr_earlyln_scale_smoke_{args.device.replace(':','_')}")
            assert r["status"] == "COMPLETED", (K, r.get("status"))
        print(f"ncr_earlyln_scale smoke PASSED (4-step {args.device} cells, all K shapes)")
        return
    if args.harvest:
        if args.seeds is not None:
            seeds = tuple(int(s) for s in args.seeds.split(","))
            seeds_by_K = {K: seeds for K in GRID_SHAPES}
            print(f"WARNING: --seeds override in use ({seeds}) -- auto-discovery bypassed; "
                  f"only use this if you have independently confirmed every K ran this exact "
                  f"seed set (audit MAJOR-2: a stale uniform assumption after a per-K trim "
                  f"silently mis-scores a rung).", file=sys.stderr)
        else:
            seeds_by_K = discover_seeds_by_K(args.outdir)
            print(f"auto-discovered seeds by K (from {args.outdir}): {seeds_by_K}", file=sys.stderr)
        h = harvest(args.outdir, seeds_by_K)
        print(json.dumps(h, indent=1, default=float))
        return
    if args.cell:
        assert args.K is not None, "--cell requires --K"
        rec = run_earlyln_cell(args.K, args.seed, args.steps, args.device, args.outdir,
                               args.stop_file, args.ceiling_gpuh)
        print(f"EARLYLN CELL K={args.K} seed={args.seed} status={rec.get('status')}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
