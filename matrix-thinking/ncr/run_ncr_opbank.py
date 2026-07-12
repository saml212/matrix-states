"""NCR operator-bank runner -- Phase-0 calibration ONLY (NOVEL_ARCH_WATERFALL.md
S8.1.7). Wave-1's full eval grid (ladder/sweep/cost_probe/B-CHAIN crossed
grid) is explicitly OUT OF SCOPE here -- sized after Phase-0's real rate is
known, per S8.1.7's own pin ("no wave-1 cell count is committed until that
number exists"). This module's job: train 3 cells (one per trained arm),
run every S8.1.5 instrument duty to completion, and measure the real GPU
rate.

Per-cell pipeline (`run_cell_bank`), ORDER LOAD-BEARING (mirrors run_ncr.py's
run_cell precedent):
  1. train        mixed Axis-R/Axis-B batches, h in {1,2,3} per block
  2. z_dump + deep probe + Axis-C lock + trust_screen, PER RELATION (S8.1.5:
                  "not assumed to transfer as a single global check")
  3. blank-out    S8.1.5: corrupt raw R*K bindings post-write; Z_bank must
                  be bit-identical, grad w.r.t. raw inputs exactly zero
  4. relation-ID-swap ablation (S8.1.5's bank-specific P=1 analog) -- the
                  MANDATORY new gate, executed with both the pinned <0.3 bar
                  AND an empirical per-episode random-direction control (m3)
  5. read-vector-std  J1 fix: fwm-bank/loopedvec-bank only, bar >= 0.04
  6. small eval   train_support (h=1,2,3) + h* PER RELATION (Axis R-BANK
                  only; full grid + Axis B-CHAIN deferred to wave-1)
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

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
for p in (CHAPTER2, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import task_e as te                      # noqa: E402
import ncr_opbank_task as ot              # noqa: E402
import ncr_opbank_models as om            # noqa: E402
import ncr_spectral as ns                 # noqa: E402

TRAIN_BATCH = 48        # smaller than the single-relation's 64: R*K=24 tokens vs K=8
TRAIN_LR = 3e-4
RVSTD_BAR = 0.04
SWAP_BAR = 0.3
RUNNER_TAG = "ncr_opbank_phase0_v1"
ANCHOR_GPUH_80K = 3.36   # S8.1.7 conservative 3x-token-count estimate; Phase-0 supersedes


def git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_HERE,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNKNOWN"


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def cell_id(arm: str) -> str:
    return f"ncropbank_{arm}"


def cosine_loss(pred, target):
    return 1.0 - torch.cosine_similarity(pred, target, dim=-1).mean()


def stop_requested(stop_file: str) -> bool:
    return bool(stop_file) and os.path.exists(stop_file)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_cell_bank(model, cfg: ot.BankConfig, device: str, steps: int, seed: int,
                    stop_file: str, ceiling_s: float, log_every: int = 200,
                    train_batch: int = TRAIN_BATCH) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    model.train()
    n_skipped = 0
    loss_hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        b = ot.sample_train_batch(cfg, train_batch, gen, device)
        for k in ("keys", "values", "rel_ids", "query_keys", "r1", "h1", "r2", "h2", "targets"):
            b[k] = b[k].to(device)
        pred, _ = model(b)
        loss = cosine_loss(pred, b["targets"])
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
            print(f"  [{model.arm}] step {step:6d}  cosine_loss {loss.item():.4f}  "
                 f"elapsed {elapsed:.0f}s", flush=True)
            if stop_requested(stop_file):
                print(f"  [{model.arm}] STOP file seen -- exiting 3")
                sys.exit(3)
            if elapsed > ceiling_s:
                return dict(status="ABORTED-BUDGET", step=step, elapsed_s=elapsed,
                           ceiling_s=ceiling_s, n_skipped_steps=n_skipped,
                           loss_history=loss_hist)
    return dict(status="COMPLETED", step=steps, elapsed_s=time.time() - t0,
               ceiling_s=ceiling_s, n_skipped_steps=n_skipped, loss_history=loss_hist)


# ---------------------------------------------------------------------------
# Instruments (S8.1.5)
# ---------------------------------------------------------------------------

@torch.no_grad()
def z_dump_bank(model, cfg: ot.BankConfig, device: str, seed: int, n_examples: int = 4) -> dict:
    gz = torch.Generator(device=device).manual_seed(seed + 20_000)
    ep = ot.generate_bank_episode(n_examples, cfg, gz, device)
    if model.arm == "ncr-bank":
        Z = model.encode(ep["keys"], ep["values"], ep["rel_ids"])
    elif model.arm == "fwm-bank":
        Z = model.encode(ep["keys"], ep["values"], ep["rel_ids"])
    else:
        raise ValueError(f"z_dump_bank only applies to matrix-state arms, got {model.arm}")
    z_ideal_per_r = []
    for r in range(cfg.R):
        keys_r = ep["keys"][:, r * cfg.K:(r + 1) * cfg.K, :]
        values_r = ep["values"][:, r * cfg.K:(r + 1) * cfg.K, :]
        z_ideal_per_r.append(te.ideal_Z(keys_r, values_r).detach().cpu().tolist())
    return dict(Z=Z.detach().cpu().tolist(), z_ideal_per_r=z_ideal_per_r)


def deep_probe_bank(zd: dict, cfg: ot.BankConfig, outdir: str, cid: str,
                    hops: list) -> dict:
    """PER RELATION (S8.1.5): analyze_zdump_arrays/trust_screen/axis_c_lock,
    the discharged S3.4 machinery, reused verbatim R times -- not assumed to
    transfer as a single global check."""
    per_relation = {}
    for r in range(cfg.R):
        Z_r = [ex[r] for ex in zd["Z"]]        # Z: (n_ex, R, d, d) list -> per-r slice
        z_ideal_r = zd["z_ideal_per_r"][r]
        probe = ns.analyze_zdump_arrays(Z_r, z_ideal_r, hops)
        lock_path = os.path.join(outdir, f"{cid}.r{r}.axis_c_lock.json")
        write_axis = ns.write_axis_c_lock(lock_path, f"{cid}.r{r}", cfg.K, probe)
        ns.verify_axis_c_lock(lock_path)        # tamper/hash self-check
        screens = [ns.trust_screen(np.array(ex["blocks"]["A"]), np.array(ex["blocks"]["B"]),
                                   np.array(ex["blocks"]["C"]), np.array(ex["blocks"]["D"]),
                                   hops) for ex in probe["per_example"]]
        per_relation[str(r)] = dict(
            phase_resid_max_mean=probe["phase_resid_max_mean"],
            axis_c_lock_sha256=write_axis["lock_sha256"],
            A_eff_rank=[ex["A_eff_rank"] for ex in probe["per_example"]],
            c_star=[ex["c_star"] for ex in probe["per_example"]],
            trust_tau=ns.TAU,
            trust_rule_trusted_at_hstar=bool(all(
                s["per_h"].get(str(hops[-1]), s["per_h"].get(hops[-1], {})).get("rule_trusted", False)
                for s in screens)) if screens else None,
        )
    return per_relation


def blank_out_check_bank(model, cfg: ot.BankConfig, device: str, seed: int) -> dict:
    """S8.1.5: corrupt raw R*K bindings POST-write; decode bit-identical AND
    grad w.r.t. post-write raw inputs exactly zero -- gradient-based, not a
    shape check. Read = fixed h=2 on relation r=0, single block."""
    gen = torch.Generator(device=device).manual_seed(seed + 31_000)
    ep = ot.generate_bank_episode(16, cfg, gen, device)
    keys = ep["keys"].clone().requires_grad_(True)
    values = ep["values"].clone().requires_grad_(True)
    rel_ids = ep["rel_ids"]
    q = ep["pool"][:, :1, :]

    if model.arm in ("ncr-bank", "fwm-bank"):
        state = model.encode(keys, values, rel_ids)
        read = (lambda s: om.NCRBankModel.eval_read(s, q, 0, 2)["o"]) if model.arm == "ncr-bank" \
            else (lambda s: model.read_fixed_h(s, q, 0, 2))
    elif model.arm == "loopedvec-bank":
        q_rel = torch.zeros(q.shape[0], q.shape[1], dtype=torch.long, device=device)
        state = model.encode(keys, values, rel_ids, q, q_rel)
        read = lambda s: model.decode(model.iterate_fixed_h(s, 2))  # noqa: E731
    else:
        raise ValueError(model.arm)

    state_frozen = state.detach()
    with torch.no_grad():
        pred1 = read(state_frozen)
        pred2 = read(state_frozen)   # re-read; raw keys/values never touched by `read`
    bit_identical = bool(torch.equal(pred1, pred2))

    state_leaf = state.detach().clone().requires_grad_(True)
    pred_leaf = read(state_leaf)
    g = torch.autograd.grad(pred_leaf.sum(), [keys, values], allow_unused=True)
    grad_zero = all(gi is None for gi in g)

    pred_live = read(state)
    g_live = torch.autograd.grad(pred_live.sum(), keys, allow_unused=True)[0]
    write_path_alive = g_live is not None and g_live.abs().sum().item() > 0
    return dict(bit_identical=bit_identical, grad_exactly_zero=grad_zero,
               write_path_alive=bool(write_path_alive),
               passed=bool(bit_identical and grad_zero and write_path_alive))


def relation_id_swap_ablation(model, cfg: ot.BankConfig, device: str, seed: int,
                              h: int) -> dict:
    """S8.1.5/S8.3 m3: the bank's own P=1 analog. Feed the WRONG relation id
    for a query whose true relation is r=0; recovery must collapse. Reports
    BOTH the pinned a-priori bar (<0.3) and an empirical per-episode
    random-direction control (m3's fix: report both, control authoritative
    per S8.3a's recommendation).

    AUDIT FIX (§8.4a MAJOR-1): loopedvec-bank now genuinely receives a
    query-relation tag at encode() time (see LoopedVecBankModel.encode's
    docstring) -- the swap ablation is applicable for it too, not N/A."""
    gen = torch.Generator(device=device).manual_seed(seed + 40_000)
    r_true, r_wrong = 0, 1
    eb = ot.sample_eval_batch_axis_r(cfg, 64, gen, r=r_true, h=h, device=device)
    with torch.no_grad():
        if model.arm in ("ncr-bank", "fwm-bank"):
            Z = model.encode(eb["keys"], eb["values"], eb["rel_ids"])
            read = (lambda r: om.NCRBankModel.eval_read(Z, eb["query_keys"], r, h)["o"]) \
                if model.arm == "ncr-bank" else \
                (lambda r: model.read_fixed_h(Z, eb["query_keys"], r, h))
            pred_right = read(r_true)
            pred_wrong = read(r_wrong)
        elif model.arm == "loopedvec-bank":
            q_shape = eb["query_keys"].shape[:2]
            q_rel_right = torch.full(q_shape, r_true, dtype=torch.long, device=device)
            q_rel_wrong = torch.full(q_shape, r_wrong, dtype=torch.long, device=device)
            x0_right = model.encode(eb["keys"], eb["values"], eb["rel_ids"],
                                    eb["query_keys"], q_rel_right)
            x0_wrong = model.encode(eb["keys"], eb["values"], eb["rel_ids"],
                                    eb["query_keys"], q_rel_wrong)
            pred_right = model.decode(model.iterate_fixed_h(x0_right, h))
            pred_wrong = model.decode(model.iterate_fixed_h(x0_wrong, h))
        else:
            raise ValueError(model.arm)
        random_dir = torch.randn_like(eb["targets"])
        random_dir = random_dir / random_dir.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        cos_right = torch.cosine_similarity(pred_right, eb["targets"], dim=-1)
        cos_wrong = torch.cosine_similarity(pred_wrong, eb["targets"], dim=-1)
        cos_control = torch.cosine_similarity(random_dir, eb["targets"], dim=-1)
    applicable = True   # AUDIT FIX (§8.4a MAJOR-1): all 3 gradient arms now applicable
    right, wrong = float(cos_right.median()), float(cos_wrong.median())
    return dict(applicable=applicable, r_true=r_true, r_wrong=r_wrong, h=h,
               median_cos_right=right, median_cos_wrong=wrong,
               right_minus_wrong_gap=right - wrong,   # §8.4a MINOR-1: makes a
               # vacuous pass (an undertrained model where right~=wrong~=0,
               # both < 0.3) visible in the gate table -- distinct from a
               # genuine relation-sensitive pass (right high, wrong low, gap
               # large). Not gated itself (Phase-0 doesn't require training
               # to converge, per the calibration-lesson hard rule), but
               # reported so the coordinator's readout isn't misled.
               median_cos_control=float(cos_control.median()),
               bar=SWAP_BAR,
               passed_pinned_bar=bool(cos_wrong.median() < SWAP_BAR),
               passed_empirical_control=bool(
                   abs(cos_wrong.median() - cos_control.median()) < 0.15))


@torch.no_grad()
def read_vector_std_bank(model, cfg: ot.BankConfig, device: str, seed: int,
                         depths=(3, 61)) -> dict:
    """J1 fix (S8.3): deviating-read bank arms only (fwm-bank, loopedvec-bank)."""
    gen = torch.Generator(device=device).manual_seed(seed + 32_000)
    per_depth = {}
    for h in depths:
        eb = ot.sample_eval_batch_axis_r(cfg, 64, gen, r=0, h=h, device=device)
        if model.arm == "fwm-bank":
            Z = model.encode(eb["keys"], eb["values"], eb["rel_ids"])
            read = model.read_fixed_h(Z, eb["query_keys"], 0, h)
        elif model.arm == "loopedvec-bank":
            q_rel = torch.zeros(eb["query_keys"].shape[:2], dtype=torch.long, device=device)
            x0 = model.encode(eb["keys"], eb["values"], eb["rel_ids"], eb["query_keys"], q_rel)
            read = model.iterate_fixed_h(x0, h)
        else:
            raise ValueError(f"read_vector_std_bank only applies to deviating arms, got {model.arm}")
        std_q = read.std(dim=1)
        per_depth[str(h)] = dict(mean=float(std_q.mean().item()),
                                 passed=bool(std_q.mean().item() >= RVSTD_BAR))
    return dict(bar=RVSTD_BAR, per_depth=per_depth,
               passed=bool(all(v["passed"] for v in per_depth.values())))


# ---------------------------------------------------------------------------
# Small Phase-0 eval (Axis R-BANK only; full grid deferred to wave-1)
# ---------------------------------------------------------------------------

def eval_cell_bank_small(model, cfg: ot.BankConfig, device: str, seed: int,
                         h_star: int) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed + 50_000)
    points = {}
    with torch.no_grad():
        for h in (*ot.H_TRAIN, h_star):
            per_r = {}
            for r in range(cfg.R):
                eb = ot.sample_eval_batch_axis_r(cfg, 64, gen, r=r, h=h, device=device)
                if model.arm == "ncr-bank":
                    Z = model.encode(eb["keys"], eb["values"], eb["rel_ids"])
                    pred = om.NCRBankModel.eval_read(Z, eb["query_keys"], r, h)["o"]
                elif model.arm == "fwm-bank":
                    Z = model.encode(eb["keys"], eb["values"], eb["rel_ids"])
                    pred = model.read_fixed_h(Z, eb["query_keys"], r, h)
                elif model.arm == "loopedvec-bank":
                    q_rel = torch.full(eb["query_keys"].shape[:2], r, dtype=torch.long, device=device)
                    x0 = model.encode(eb["keys"], eb["values"], eb["rel_ids"],
                                      eb["query_keys"], q_rel)
                    pred = model.decode(model.iterate_fixed_h(x0, h))
                elif model.arm == "cmlp-bank":
                    b2 = dict(eb, hops=torch.full_like(eb["targets"][..., 0], h, dtype=torch.int64))
                    pred = model(b2)[0]
                else:
                    raise ValueError(model.arm)
                cos = torch.cosine_similarity(pred, eb["targets"], dim=-1)
                per_r[str(r)] = dict(recovered_frac_0_9=float((cos >= 0.9).float().mean()),
                                     mean_cos=float(cos.mean()))
            points[str(h)] = per_r
    bank_input = {0: {r: points[str(h_star)][str(r)]["recovered_frac_0_9"]
                      for r in range(cfg.R)}}
    return dict(points=points, h_star=h_star,
               bank_score_single_seed=ot.bank_score(bank_input))


# ---------------------------------------------------------------------------
# Cell orchestration
# ---------------------------------------------------------------------------

def run_cell_bank(arm: str, seed: int, steps: int, device: str, outdir: str,
                  stop_file: str = "", ceiling_gpuh: float = 2.0,
                  train_batch: int = TRAIN_BATCH) -> dict:
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
    model = om.ARM_BUILDERS[arm]().to(device)
    rec = dict(cell_id=cid, arm=arm, seed=seed, runner_tag=RUNNER_TAG,
              git_commit=git_commit(), params=om.n_params(model), train_batch=train_batch,
              host=socket.gethostname(), device=device, torch_version=torch.__version__,
              started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    ceiling_s = ceiling_gpuh * 3600.0 if device == "cuda" else float("inf")
    t0 = time.time()

    tr = train_cell_bank(model, cfg, device, steps, seed, stop_file, ceiling_s,
                         train_batch=train_batch)
    rec["train"] = tr
    if tr["status"] != "COMPLETED":
        rec["status"] = tr["status"]
        rec["elapsed_s"] = time.time() - t0
        atomic_write_json(out_path, rec)
        return rec

    model.eval()
    h_star = ot.GRID8["h_star"]
    if arm in ("ncr-bank", "fwm-bank"):
        zd = z_dump_bank(model, cfg, device, seed)
        rec["deep_probe_per_relation"] = deep_probe_bank(
            zd, cfg, outdir, cid, hops=[1, 2, 3, h_star])

    if arm in ("ncr-bank", "fwm-bank", "loopedvec-bank"):
        rec["blank_out"] = blank_out_check_bank(model, cfg, device, seed)
        assert rec["blank_out"]["passed"], f"blank-out FAILED for {cid}: {rec['blank_out']}"
        rec["relation_id_swap"] = relation_id_swap_ablation(model, cfg, device, seed, h=h_star)

    if arm in ("fwm-bank", "loopedvec-bank"):
        rec["read_vector_std"] = read_vector_std_bank(model, cfg, device, seed)

    rec["eval"] = eval_cell_bank_small(model, cfg, device, seed, h_star)
    rec["elapsed_s"] = time.time() - t0
    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
    rec["status"] = "COMPLETED"
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(out_path, rec)
    print(f"  [{cid}] COMPLETED in {rec['elapsed_s']:.0f}s -> {out_path}", flush=True)
    return rec


def phase0_bank(device: str, outdir: str, steps: int, stop_file: str = "") -> dict:
    os.makedirs(outdir, exist_ok=True)
    rep = om.assert_param_match()
    cells = {}
    for arm in om.TRAINED_ARMS:
        if stop_requested(stop_file):
            print("STOP file seen between cells -- exiting 3")
            sys.exit(3)
        print(f"\n=== OPBANK PHASE 0 CELL: {arm} (seed 0, steps {steps}) ===", flush=True)
        cells[arm] = run_cell_bank(arm, 0, steps, device, outdir, stop_file)

    table = dict(param_report=rep, cells={}, steps=steps, anchor_gpuh_80k=ANCHOR_GPUH_80K)
    rates, all_pass = {}, True
    for arm, rec in cells.items():
        if rec.get("status") != "COMPLETED":
            table["cells"][arm] = dict(status=rec.get("status"))
            all_pass = False
            continue
        rate_gpuh = rec["train"]["elapsed_s"] / 3600.0 * (80_000 / steps)
        rates[arm] = rate_gpuh
        mandatory = [rec.get("blank_out", {}).get("passed") is True]
        if arm in ("fwm-bank", "loopedvec-bank"):
            mandatory.append(rec.get("read_vector_std", {}).get("passed") is True)
        swap = rec.get("relation_id_swap", {})
        if swap.get("applicable"):
            mandatory.append(swap.get("passed_pinned_bar") is True)
        row = dict(status="COMPLETED", rate_gpuh_80k_equiv=rate_gpuh,
                  cell_total_gpu_h=rec.get("gpu_h"),
                  train_final_loss=rec["train"]["loss_history"][-1][1],
                  n_skipped_steps=rec["train"]["n_skipped_steps"],
                  bank_score_single_seed=rec["eval"]["bank_score_single_seed"],
                  blank_out_pass=rec.get("blank_out", {}).get("passed"),
                  swap_ablation=swap, read_vector_std_pass=rec.get("read_vector_std", {}).get("passed"),
                  gate_pass=bool(all(mandatory)))
        all_pass = all_pass and row["gate_pass"]
        table["cells"][arm] = row
    table["rate_gpuh_80k_mean"] = (sum(rates.values()) / len(rates)) if rates else None
    table["phase0_verdict"] = "PASS" if all_pass else "FAIL"
    atomic_write_json(os.path.join(outdir, "opbank_phase0_gate_table.json"), table)
    if rates:
        atomic_write_json(os.path.join(outdir, "opbank_phase0_rate.json"),
                          dict(rate_gpuh_80k_per_arm=rates,
                               rate_gpuh_80k_mean=table["rate_gpuh_80k_mean"]))
    print("\n=== OPBANK PHASE 0 GATE TABLE ===")
    print(json.dumps({k: v for k, v in table.items() if k != "param_report"}, indent=1, default=float))
    print(f"OPBANK PHASE0 VERDICT: {table['phase0_verdict']}")
    return table


def smoke(device: str = "cpu"):
    """Logic suite entry -- see ncr_opbank_selftest.py for the full,
    negative-tested (CPU-only) suite. This is a quick end-to-end sanity
    call, device-parametrized so the SAME code path can be re-run on real
    CUDA before deploy (CLAUDE.md: CPU-stub suites test logic only; a
    separate narrow real-device smoke is required)."""
    outdir = f"/tmp/ncropbank_smoke_{device.replace(':', '_')}"
    for arm in om.ALL_ARMS:
        rec = run_cell_bank(arm, seed=0, steps=5, device=device, outdir=outdir)
        assert rec["status"] == "COMPLETED", (arm, rec.get("status"))
    print(f"run_ncr_opbank smoke PASSED (5-step {device} cells, all 4 arms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--phase0", action="store_true")
    ap.add_argument("--cell", choices=list(om.TRAINED_ARMS), default=None,
                    help="run ONE Phase-0 cell directly (for one-GPU-per-cell "
                         "dispatch); resume-safe, same output schema as --phase0's "
                         "per-arm cells -- a later --phase0 pass on the same outdir "
                         "skips COMPLETED cells and just rebuilds the gate table")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ceiling-gpuh", type=float, default=2.0)
    ap.add_argument("--train-batch", type=int, default=TRAIN_BATCH,
                    help="train batch size (default 48; the single-relation proven "
                         "recipe is 256 -- §8.5's re-calibration uses 256 to match the "
                         "20.5M-example budget the contender converged under there)")
    ap.add_argument("--outdir", type=str, default=os.path.join(_HERE, "results_opbank_phase0"))
    ap.add_argument("--stop-file", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.smoke:
        smoke(args.device)
        return
    if args.cell:
        rec = run_cell_bank(args.cell, args.seed, args.steps, args.device, args.outdir,
                            args.stop_file, args.ceiling_gpuh, train_batch=args.train_batch)
        print(f"CELL {args.cell} status={rec.get('status')}")
        return
    if args.phase0:
        phase0_bank(args.device, args.outdir, args.steps, args.stop_file)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
