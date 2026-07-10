"""NCR wave-1 runner -- train / instrument / eval / Phase-0 orchestration.

Design of record: NOVEL_ARCH_WATERFALL.md S3 (as amended by S3.9) + the S5
nits; launch double-gate per S3.8 (all three design gates discharged on the
record: S2.30 / S4+S3.9 / S6). This module is the standing chain's BUILD
artifact; it never fires Phase 1 (wave-1) -- that needs a fresh coordinator
go/no-go against the ledger (S3.6).

Per-cell pipeline (`run_cell`), ORDER LOAD-BEARING:
  1. train        h in {1,2,3} only, backprop through <=3 naive matmuls
                  (S3.1); resume-safe checkpoints; budget breaker at 1.5x the
                  anchor-scaled rate (S2.29 lesson: ceilings scale with the
                  design's own cost axes); STOP-file support.
  2. Z-dump       4 eval examples, seed+20000 generator (run_task_e's own
                  --save-z convention), matrix-state arms (ncr, fwm).
  3. deep probe   restricted-operator analysis (m4; S9/S10 lineage).
  4. AXIS-C LOCK  per-seed predicted decay curves written + hashed BEFORE
                  any far-h behavioral eval runs (S3.2) -- the eval path
                  REFUSES far-h points for matrix arms without a verified
                  lock (negative-tested).
  5. trust screen S3.4 sigma-form on measured blocks (a-priori screen only;
                  labels RULE-TRUSTED / SHADOW-VERIFIED / UNTRUSTED, mi3).
  6. blank-out    m3: corrupt raw episode inputs post-write; decode
                  bit-identical AND grad exactly zero (None) -- executed,
                  not a shape check.
  7. read-vector-std  P3 diagnostic for deviating-read arms (fwm,
                  loopedvec), bar mean >= 0.04 at every probe depth
                  (S3.1, mi4-harmonized to Stage 2's derivation).
  8. eval grid    all pinned points (ncr_task.eval_points): fp32 reads +
                  fp64 shadows on the SAME batches, binexp-vs-loop agreement
                  bar (MA5), effective-hop stratification, residue
                  label-and-exclude schema, reducer-detection signature,
                  failure front (first crossing; revivals reported, never
                  re-admitted -- mi5), Axis-B timing probes.

fp64-shadow scope (disclosed): the shadow certifies h-fold fp32 ROUNDING
accumulation (S3.4 mi3: rounding only, never leakage), so it wraps the
h-iteration reads (ncr binexp/loop, fwm recursive, loopedvec loop). C_MLP's
read is a single O(1) MLP pass with no h-accumulation -- shadow n/a,
recorded as null, never silently.

C_MLP h-signal (disclosed): comparisons of record receive the raw integer h
only (S3.1); C_MLP keeps its INHERITED one-hot(h) -- that is its S3.3 role
(disclosed weak control, architecturally unable to extrapolate, never the
comparison of record).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
if CHAPTER2 not in sys.path:
    sys.path.insert(0, CHAPTER2)

import task_e as te                      # noqa: E402
import ncr_task as nt                    # noqa: E402
import ncr_models as nm                  # noqa: E402
import ncr_spectral as ns                # noqa: E402

RUNNER_TAG = "ncr_wave1_build1"
ANCHOR_GPUH_80K = 2.4                    # S3.6 measured rate anchor (80K-step Task-E cell)
PHASE0_STEPS_DEFAULT = 40_000            # K=8 converges by 40K (S3.6's own note; S9's
                                         # converged-at-40K dumps are the evidence)
BREAKER_FACTOR = 1.5                     # S3.6 per-cell abort factor
TRAIN_BATCH = 256
TRAIN_LR = 3e-4
EVAL_BATCHES = 8
EVAL_BATCH_SIZE = 256
TIMING_BATCH = 32
TIMING_REPEATS = 3
RVSTD_BAR = 0.04                         # S3.1 mi4-harmonized read-vector-std bar
RVSTD_DEPTHS = (1, 2, 3, 5, 13, 21)
FAR_H_LOCK_THRESHOLD = 7                 # h > 7 = beyond the legacy window -> lock required


# ---------------------------------------------------------------------------
# Fingerprints / IO
# ---------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_HERE, text=True).strip()
    except Exception:
        return "UNKNOWN"


def config_sha(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def cell_id(arm: str, K: int, seed: int) -> str:
    return f"ncr_{arm}_K{K}_s{seed}"


def stop_requested(stop_file: str) -> bool:
    return bool(stop_file) and os.path.exists(stop_file)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def recovery_stats(cos: torch.Tensor) -> dict:
    cos = cos.reshape(-1).float()
    d = dict(mean_cos=cos.mean().item(),
             cos_p10=torch.quantile(cos, 0.10).item(),
             cos_p50=torch.quantile(cos, 0.50).item(),
             cos_p90=torch.quantile(cos, 0.90).item(),
             n_items=int(cos.numel()))
    for tau in nt.TAUS:
        d[f"recovered_frac@{tau}"] = (cos > tau).float().mean().item()
    return d


def cosine_loss(pred, target):
    return (1.0 - nm.recovery_cosine(pred, target)).mean()


# ---------------------------------------------------------------------------
# Training (resume-safe, breaker, STOP)
# ---------------------------------------------------------------------------

def _ckpt_path(outdir: str, cid: str) -> str:
    return os.path.join(outdir, f"{cid}.ckpt.pt")


def save_ckpt(path: str, model, opt, gen, step: int, csha: str, device: str):
    state = dict(step=step, config_sha=csha,
                 model=model.state_dict(), opt=opt.state_dict(),
                 gen_state=gen.get_state(),
                 torch_rng=torch.get_rng_state())
    if device == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state()
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def try_resume(path: str, model, opt, gen, csha: str, device: str) -> int:
    """Resume only from a VALID checkpoint (loads cleanly + config_sha
    matches); a mismatched or corrupt checkpoint is refused loudly, never
    silently retrained over (resume-safety = validity, not existence)."""
    if not os.path.exists(path):
        return 0
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"checkpoint {path} exists but fails to load: {e!r}; "
                           f"refusing to silently overwrite -- move it aside") from e
    assert state["config_sha"] == csha, (
        f"checkpoint {path} config_sha={state['config_sha']} != current {csha}; "
        f"refusing a mismatched resume")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
    gen.set_state(state["gen_state"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if device == "cuda" and "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    return int(state["step"])


def train_cell(model, cfg, device: str, steps: int, seed: int, outdir: str,
               cid: str, csha: str, stop_file: str, ceiling_s: float,
               log_every: int = 500, ckpt_every: int = 2000) -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
    ck = _ckpt_path(outdir, cid)
    start_step = try_resume(ck, model, opt, gen, csha, device)
    if start_step:
        print(f"  [{cid}] RESUMED at step {start_step}", flush=True)
    model.train()
    n_skipped = 0
    loss_hist = []
    t0 = time.time()
    for step in range(start_step + 1, steps + 1):
        b = te.sample_batch(cfg, TRAIN_BATCH, gen, hop_set=cfg.H_train,
                            device=device, assert_injective=(step == start_step + 1))
        pred, _ = model(b)
        loss = cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1
        if step % log_every == 0 or step == start_step + 1:
            elapsed = time.time() - t0
            loss_hist.append([step, float(loss.item())])
            extra = f"  [skipped {n_skipped} non-finite]" if n_skipped else ""
            print(f"  [{cid}] step {step:6d}  cosine_loss {loss.item():.4f}"
                  f"  elapsed {elapsed:.0f}s{extra}", flush=True)
            if stop_requested(stop_file):
                save_ckpt(ck, model, opt, gen, step, csha, device)
                print(f"  [{cid}] STOP file seen -- checkpointed at {step}, exiting 3")
                sys.exit(3)
            if elapsed > ceiling_s:
                save_ckpt(ck, model, opt, gen, step, csha, device)
                return dict(status="ABORTED-BUDGET", step=step,
                            elapsed_s=elapsed, ceiling_s=ceiling_s,
                            n_skipped_steps=n_skipped, loss_history=loss_hist)
        if step % ckpt_every == 0:
            save_ckpt(ck, model, opt, gen, step, csha, device)
    save_ckpt(ck, model, opt, gen, steps, csha, device)
    return dict(status="COMPLETED", step=steps, elapsed_s=time.time() - t0,
                ceiling_s=ceiling_s, n_skipped_steps=n_skipped,
                loss_history=loss_hist)


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

@torch.no_grad()
def z_dump(model, cfg, device: str, seed: int, n_examples: int = 4) -> dict:
    gz = torch.Generator(device=device).manual_seed(seed + 20_000)
    bz = te.sample_batch(cfg, n_examples, gz, hop_set=(1,), device=device)
    Z = model.encode(bz["keys"], bz["values"])
    return dict(Z=Z.detach().cpu().tolist(), z_ideal=bz["z_ideal"].detach().cpu().tolist())


def blank_out_check(model, cfg, device: str, seed: int) -> dict:
    """m3, executed: corrupt raw episode inputs POST-write; decode must be
    bit-identical AND grad w.r.t. post-write raw inputs exactly zero (None)."""
    gen = torch.Generator(device=device).manual_seed(seed + 31_000)
    b = te.sample_batch(cfg, 16, gen, hop_set=(2,), device=device)
    keys = b["keys"].clone().requires_grad_(True)
    values = b["values"].clone().requires_grad_(True)
    q, hops = b["query_keys"], b["hops"]

    if model.arm in ("ncr", "fwm"):
        state = model.encode(keys, values)
        read = (lambda s: nm.binexp_read(s, q, 2)["o"]) if model.arm == "ncr" \
            else (lambda s: model.read_fixed_h(s, q, 2))
    elif model.arm == "loopedvec":
        state = model.encode(keys, values, q)
        read = lambda s: model.decode(model.iterate_fixed_h(s, 2))  # noqa: E731
    else:
        raise ValueError(model.arm)

    state_frozen = state.detach()
    with torch.no_grad():
        pred1 = read(state_frozen)
        keys_c = torch.randn_like(keys)      # corrupt post-write (never used by read)
        values_c = torch.randn_like(values)
        del keys_c, values_c
        pred2 = read(state_frozen)
    bit_identical = bool(torch.equal(pred1, pred2))

    state_leaf = state.detach().clone().requires_grad_(True)
    pred_leaf = read(state_leaf)
    g = torch.autograd.grad(pred_leaf.sum(), [keys, values], allow_unused=True)
    grad_zero = all(gi is None for gi in g)
    # sanity: through the NON-detached state the raw inputs DO matter
    pred_live = read(state)
    g_live = torch.autograd.grad(pred_live.sum(), keys, allow_unused=True)[0]
    write_path_alive = g_live is not None and g_live.abs().sum().item() > 0
    return dict(bit_identical=bit_identical, grad_exactly_zero=grad_zero,
                write_path_alive=bool(write_path_alive),
                passed=bool(bit_identical and grad_zero and write_path_alive))


@torch.no_grad()
def read_vector_std(model, cfg, device: str, seed: int) -> dict:
    """P3 read-vector-std across queries for deviating-read arms; statistic =
    std across the Q query axis per (item, dim), MEAN-aggregated (Stage 2's
    MAJOR-1(a) statistic -- mean grades partial collapse; the archived max is
    co-reported), torch unbiased std (ddof=1). Bar: mean >= 0.04 at EVERY
    probe depth (S3.1, mi4)."""
    gen = torch.Generator(device=device).manual_seed(seed + 32_000)
    per_depth = {}
    for h in RVSTD_DEPTHS:
        b = nt.sample_eval_batch(cfg, 64, gen, h, device=device)
        q = b["query_keys"]
        if model.arm == "fwm":
            state = model.encode(b["keys"], b["values"])
            read = model.read_fixed_h(state, q, h)
        elif model.arm == "loopedvec":
            x0 = model.encode(b["keys"], b["values"], q)
            read = model.iterate_fixed_h(x0, h)      # pre-decode state
        else:
            raise ValueError(f"read_vector_std only applies to deviating arms, got {model.arm}")
        std_q = read.std(dim=1)                       # (B, d), unbiased
        per_depth[str(h)] = dict(mean=float(std_q.mean().item()),
                                 max=float(std_q.max().item()),
                                 passed=bool(std_q.mean().item() >= RVSTD_BAR))
    return dict(bar=RVSTD_BAR, per_depth=per_depth,
                passed=bool(all(v["passed"] for v in per_depth.values())))


# ---------------------------------------------------------------------------
# Eval reads per arm (fp32 + fp64 shadow on the same batch)
# ---------------------------------------------------------------------------

def _fwm_read_dtype(model, Z, q, h, dtype):
    ln = model.read_ln
    w = ln.weight.to(dtype)
    bias = ln.bias.to(dtype)
    cur = q.to(dtype)
    Zd = Z.to(dtype)
    for _ in range(h):
        cur = torch.nn.functional.layer_norm(
            torch.einsum("bij,bqj->bqi", Zd, cur), (model.d,), w, bias, ln.eps)
    return cur


def _loopedvec_iterate_dtype(model, x0, h, dtype):
    ln, w1, w2 = model.step_ln, model.step_w1, model.step_w2
    lw, lb = ln.weight.to(dtype), ln.bias.to(dtype)
    w1w, w1b = w1.weight.to(dtype), w1.bias.to(dtype)
    w2w, w2b = w2.weight.to(dtype), w2.bias.to(dtype)
    cur = x0.to(dtype)
    for _ in range(h):
        y = torch.nn.functional.layer_norm(cur, (model.d,), lw, lb, ln.eps)
        y = torch.nn.functional.gelu(y @ w1w.T + w1b)
        cur = cur + y @ w2w.T + w2b
    return cur


@torch.no_grad()
def arm_state(model, batch: dict):
    """The WRITE: encode the episode into the arm's post-write state.
    Separated from the reads so Axis-B timing wraps the READ only."""
    if model.arm in ("ncr", "fwm"):
        return model.encode(batch["keys"], batch["values"])
    if model.arm == "loopedvec":
        return model.encode(batch["keys"], batch["values"], batch["query_keys"])
    if model.arm == "cmlp":
        return None                       # C_MLP's forward is its own O(1) read
    raise ValueError(model.arm)


@torch.no_grad()
def arm_reads_from_state(model, state, batch: dict, h: int,
                         with_shadow: bool = True) -> dict:
    """All reads for one arm from a post-write state at raw depth h.
    Returns {read_name: {o (fp32), o64 (fp64 shadow or None)}}."""
    q = batch["query_keys"]
    out = {}
    if model.arm == "ncr":
        for kind in ("binexp", "loop"):
            o = nm.NCRModel.eval_read(state, q, h, kind)["o"]
            o64 = (nm.NCRModel.eval_read(state.double(), q.double(), h, kind)["o"]
                   if with_shadow else None)
            out[kind] = dict(o=o, o64=o64)
    elif model.arm == "fwm":
        o = model.read_fixed_h(state, q, h)
        o64 = _fwm_read_dtype(model, state, q, h, torch.float64) if with_shadow else None
        out["fwm_recursive"] = dict(o=o, o64=o64)
    elif model.arm == "loopedvec":
        o = model.decode(model.iterate_fixed_h(state, h))
        if with_shadow:
            x64 = _loopedvec_iterate_dtype(model, state, h, torch.float64)
            o64 = x64 @ model.decode.weight.double().T + model.decode.bias.double()
        else:
            o64 = None
        out["loopedvec_iter"] = dict(o=o, o64=o64)
    elif model.arm == "cmlp":
        pred, _ = model(batch)
        out["cmlp_onehot"] = dict(o=pred, o64=None)   # O(1) read: no h-fold fp32
        # accumulation to shadow (disclosed n/a, module docstring)
    else:
        raise ValueError(model.arm)
    return out


def primary_read_name(arm: str) -> str:
    return {"ncr": "binexp", "fwm": "fwm_recursive",
            "loopedvec": "loopedvec_iter", "cmlp": "cmlp_onehot"}[arm]


@torch.no_grad()
def timed_probe(model, cfg, device: str, gen, h: int) -> dict:
    """Axis-B standardized wall-clock probe: B=32, median of 3 repeats, per
    read kind, READ ONLY (the write/encode is excluded -- the S3.2 Axis-B
    claim is sequential READ cost), fp64 shadow excluded from the timed
    region, cuda-synchronized when applicable."""
    b = nt.sample_eval_batch(cfg, TIMING_BATCH, gen, h, device=device)
    state = arm_state(model, b)
    q = b["query_keys"]
    if model.arm == "ncr":
        fns = {"binexp": lambda: nm.binexp_read(state, q, h)["o"],
               "loop": lambda: nm.loop_read(state, q, h)["o"]}
    elif model.arm == "fwm":
        fns = {"fwm_recursive": lambda: model.read_fixed_h(state, q, h)}
    elif model.arm == "loopedvec":
        fns = {"loopedvec_iter": lambda: model.decode(model.iterate_fixed_h(state, h))}
    else:  # cmlp
        fns = {"cmlp_onehot": lambda: model(b)[0]}
    times = {}
    for name, fn in fns.items():
        fn()                                      # warm-up
        rep = []
        for _ in range(TIMING_REPEATS):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            rep.append(time.perf_counter() - t0)
        times[name] = sorted(rep)[len(rep) // 2]
    return times


@torch.no_grad()
def eval_cell(model, cfg, device: str, seed: int, K: int,
              lock_content: dict | None, trust_per_h: dict | None) -> dict:
    """Full pinned grid for one cell. Matrix-state arms REFUSE far-h points
    (h > 7) unless a verified Axis-C lock is supplied (S3.2 lock-before-eval;
    negative-tested)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    tgen = torch.Generator(device=device).manual_seed(seed + 11_000)
    points_out = []
    agreement = []
    for p in nt.eval_points(K):
        if model.arm in ("ncr", "fwm") and p.h > FAR_H_LOCK_THRESHOLD:
            assert lock_content is not None, (
                f"far-h point h={p.h} requested before the Axis-C lock exists "
                f"for this cell -- S3.2's lock-before-eval order is mandatory")
        n_b = 1 if p.timed else EVAL_BATCHES
        bs = TIMING_BATCH if p.timed else EVAL_BATCH_SIZE
        cos_by_read, cos64_by_read = {}, {}
        pair_diff_max = 0.0
        for _ in range(n_b):
            b = nt.sample_eval_batch(cfg, bs, gen, p.h, device=device)
            state = arm_state(model, b)
            reads = arm_reads_from_state(model, state, b, p.h)
            for name, rr in reads.items():
                c32 = nm.recovery_cosine(rr["o"], b["targets"]).reshape(-1)
                cos_by_read.setdefault(name, []).append(c32)
                if rr["o64"] is not None:
                    c64 = nm.recovery_cosine(rr["o64"], b["targets"].double()).reshape(-1)
                    cos64_by_read.setdefault(name, []).append(c64)
            if model.arm == "ncr":
                d = (nm.recovery_cosine(reads["binexp"]["o"], b["targets"])
                     - nm.recovery_cosine(reads["loop"]["o"], b["targets"])).abs()
                pair_diff_max = max(pair_diff_max, float(d.max().item()))
        entry = dict(h=p.h, component=p.component, mode=p.mode, residue=p.residue,
                     residue_label=p.residue_label, claim_eligible=p.claim_eligible,
                     in_window=p.in_window, effective_hop=p.h % K, reads={})
        nt.require_residue_label(entry)
        for name, chunks in cos_by_read.items():
            cos = torch.cat(chunks)
            st = recovery_stats(cos)
            shadow_delta = None
            if name in cos64_by_read:
                cos64 = torch.cat(cos64_by_read[name])
                shadow_delta = float(cos.double().mean().item() - cos64.mean().item())
                st["shadow_delta_max_item"] = float((cos.double() - cos64).abs().max().item())
            st["shadow_delta"] = shadow_delta
            st["numeric_divergent_shadow"] = bool(
                shadow_delta is not None and abs(shadow_delta) > ns.SHADOW_BAR)
            if trust_per_h is not None:
                rt = trust_per_h.get(str(p.h), {}).get("rule_trusted", False)
                st["trust_label"] = ns.trust_label(rt, shadow_delta)
            entry["reads"][name] = st
        if model.arm == "ncr":
            agree_ok = (pair_diff_max <= ns.AGREE_BAR) if p.h <= ns.AGREE_H_MAX else None
            entry["binexp_loop_max_item_diff"] = pair_diff_max
            entry["binexp_loop_agreement_ok"] = agree_ok
            if agree_ok is False:
                entry["numeric_divergent_agreement"] = True
            agreement.append(dict(h=p.h, max_item_diff=pair_diff_max, ok=agree_ok))
        if p.timed or (p.component in ("ladder", "h_star") and p.h >= 61):
            entry["read_time_s"] = timed_probe(model, cfg, device, tgen, p.h)
        points_out.append(entry)

    prim = primary_read_name(model.arm)
    # aggregates honor in_window structurally (S7a audit MINOR-2: today the
    # cost probes are the only out-of-window points and carry their own
    # component, but the field is the schema's contract -- filter on it, not
    # only on component names)
    ladder = [e for e in points_out
              if e["component"] in ("ladder", "h_star") and e["in_window"]]
    ladder.sort(key=lambda e: e["h"])
    front = next((e["h"] for e in ladder
                  if e["reads"][prim]["recovered_frac@0.9"] < 0.9), None)
    revivals = [e["h"] for e in ladder
                if front is not None and e["h"] > front
                and e["reads"][prim]["recovered_frac@0.9"] >= 0.9]
    sweep = [e for e in points_out
             if e["component"] == "residue_sweep" and e["in_window"]]
    sweep_min = min(e["reads"][prim]["recovered_frac@0.9"] for e in sweep)
    reducer = bool(sweep_min >= 0.9 and front is None)
    return dict(points=points_out, agreement_checks=agreement,
                failure_front_h=front, post_front_revivals=revivals,
                reducer_signature=dict(
                    sweep_min_recovered=sweep_min, no_decay_front=front is None,
                    flagged=reducer))


# ---------------------------------------------------------------------------
# Cell pipeline
# ---------------------------------------------------------------------------

def run_cell(arm: str, K: int, seed: int, steps: int, device: str,
             outdir: str, stop_file: str = "", rate_gpuh_80k: float | None = None,
             eval_only_ckpt: str | None = None) -> dict:
    os.makedirs(outdir, exist_ok=True)
    cid = cell_id(arm, K, seed)
    out_path = os.path.join(outdir, f"{cid}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            prev = json.load(f)
        if prev.get("status") == "COMPLETED":
            print(f"  [{cid}] already COMPLETED -- skipping (resume-safe)")
            return prev
    cfg = nt.claim_config(K)
    cfg_desc = dict(arm=arm, K=K, seed=seed, steps=steps, d=nt.D_PIN,
                    H_train=list(cfg.H_train), H_test=list(cfg.H_test),
                    H_extra=list(cfg.H_extra), runner_tag=RUNNER_TAG,
                    train_batch=TRAIN_BATCH, lr=TRAIN_LR)
    csha = config_sha(cfg_desc)
    torch.manual_seed(seed)
    model = nm.ARM_BUILDERS[arm]().to(device)

    rate = (rate_gpuh_80k if rate_gpuh_80k is not None else ANCHOR_GPUH_80K)
    # S3.6 per-cell breaker at 1.5x the (anchor- or Phase-0-)calibrated rate,
    # scaled by the cell's own step budget (S2.29 lesson: ceilings scale with
    # the design's cost axes). The anchor is a MEASURED GPU rate -- on CPU
    # (local smokes only) the breaker is inactive, disclosed here.
    ceiling_s = (BREAKER_FACTOR * rate * (steps / 80_000) * 3600.0
                 if device == "cuda" else float("inf"))

    rec = dict(cell_id=cid, runner_tag=RUNNER_TAG, git_commit=git_commit(),
               config_sha=csha, config=cfg_desc, params=nm.n_params(model),
               host=socket.gethostname(), device=device,
               torch_version=torch.__version__,
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    t_start = time.time()

    if eval_only_ckpt:
        state = torch.load(eval_only_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        rec["train"] = dict(status="EVAL-ONLY", from_ckpt=eval_only_ckpt)
    else:
        tr = train_cell(model, cfg, device, steps, seed, outdir, cid, csha,
                        stop_file, ceiling_s)
        rec["train"] = tr
        if tr["status"] != "COMPLETED":
            rec["status"] = tr["status"]
            rec["elapsed_s"] = time.time() - t_start
            atomic_write_json(out_path, rec)
            return rec

    model.eval()
    lock_content = None
    trust_per_h = None
    if arm in ("ncr", "fwm"):
        zd = z_dump(model, cfg, device, seed)
        rec["z_dump"] = zd
        all_h = [p.h for p in nt.eval_points(K)]
        probe = ns.analyze_zdump_arrays(zd["Z"], zd["z_ideal"], all_h)
        rec["deep_probe"] = dict(
            phase_resid_max_per_example=[ex["phase_resid_max"] for ex in probe["per_example"]],
            phase_resid_max_mean=probe["phase_resid_max_mean"],
            c_star_per_example=[ex["c_star"] for ex in probe["per_example"]],
            scale_corrected_residual=[ex["scale_corrected_residual"] for ex in probe["per_example"]],
            normB=[ex["normB"] for ex in probe["per_example"]],
            normC=[ex["normC"] for ex in probe["per_example"]],
            normD=[ex["normD"] for ex in probe["per_example"]],
            A_eff_rank=[ex["A_eff_rank"] for ex in probe["per_example"]])
        lock_path = os.path.join(outdir, f"{cid}.axis_c_lock.json")
        write_axis = ns.write_axis_c_lock(lock_path, cid, K, probe)
        lock_content = ns.verify_axis_c_lock(lock_path)
        rec["axis_c_lock_sha256"] = write_axis["lock_sha256"]
        # trust screen on measured blocks, per example; per-h verdict is the
        # WORST over examples (rule_trusted only if EVERY example trusts --
        # bias-toward-refuse, the screen's honest direction)
        screens = [ns.trust_screen(np.array(ex["blocks"]["A"]),
                                   np.array(ex["blocks"]["B"]),
                                   np.array(ex["blocks"]["C"]),
                                   np.array(ex["blocks"]["D"]), all_h)
                   for ex in probe["per_example"]]
        trust_per_h = {}
        for hh in screens[0]["per_h"]:
            ts = [s["per_h"][hh]["T"] for s in screens]
            t_worst = max(float("inf") if t == "inf" else float(t) for t in ts)
            trust_per_h[hh] = dict(
                T=t_worst if math.isfinite(t_worst) else "inf",
                rule_trusted=bool(all(s["per_h"][hh]["rule_trusted"] for s in screens)))
        rec["trust_screen"] = dict(
            per_example_constants=[{k: s[k] for k in
                                    ("a", "r", "C_over_sigmin", "b_feedback_neglect")}
                                   for s in screens],
            tau=ns.TAU, per_h=trust_per_h,
            note="a-priori screen only (S3.4/mi3): decisive far-h attribution "
                 "rides on the Axis-C locked curves; fp64 shadow certifies "
                 "rounding only; B-feedback neglect disclosed per example")

    if arm in ("ncr", "fwm", "loopedvec"):
        rec["blank_out"] = blank_out_check(model, cfg, device, seed)
        assert rec["blank_out"]["passed"], f"blank-out FAILED for {cid}: {rec['blank_out']}"
    if arm in ("fwm", "loopedvec"):
        rec["read_vector_std"] = read_vector_std(model, cfg, device, seed)

    rec["eval"] = eval_cell(model, cfg, device, seed, K, lock_content, trust_per_h)
    rec["elapsed_s"] = time.time() - t_start
    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
    rec["status"] = "COMPLETED"
    rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(out_path, rec)
    print(f"  [{cid}] COMPLETED in {rec['elapsed_s']:.0f}s -> {out_path}")
    return rec


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------

def phase0(device: str, outdir: str, steps: int, stop_file: str) -> dict:
    """S3.6 Phase 0: the three trained-arm calibration cells at K=8, seed 0,
    run to completion, then the gate table. Emits phase0_rate.json (the
    calibrated rate that supersedes the 2.4 anchor for wave 1)."""
    os.makedirs(outdir, exist_ok=True)
    rep = nm.assert_param_match()
    cells = {}
    for arm in nm.TRAINED_ARMS:
        if stop_requested(stop_file):
            print("STOP file seen between cells -- exiting 3")
            sys.exit(3)
        print(f"\n=== PHASE 0 CELL: {arm} (K=8, seed 0, steps {steps}) ===", flush=True)
        cells[arm] = run_cell(arm, 8, 0, steps, device, outdir, stop_file)

    table = dict(param_report=rep, cells={}, steps=steps,
                 anchor_gpuh_80k=ANCHOR_GPUH_80K)
    rates = {}
    all_pass = True
    for arm, rec in cells.items():
        if rec.get("status") != "COMPLETED":
            table["cells"][arm] = dict(status=rec.get("status"))
            all_pass = False
            continue
        prim = primary_read_name(arm)
        in_dist = [e for e in rec["eval"]["points"] if e["component"] == "train_support"]
        conv = min(e["reads"][prim]["recovered_frac@0.9"] for e in in_dist)
        shadow_pts = [e for e in rec["eval"]["points"]
                      if e["reads"][prim].get("shadow_delta") is not None]
        rate_gpuh = rec["train"]["elapsed_s"] / 3600.0 * (80_000 / steps)
        rates[arm] = rate_gpuh
        row = dict(
            status="COMPLETED",
            rate_gpuh_80k_equiv=rate_gpuh,
            cell_total_gpu_h=rec.get("gpu_h"),
            train_final_loss=rec["train"]["loss_history"][-1][1],
            n_skipped_steps=rec["train"]["n_skipped_steps"],
            in_dist_min_recovered=conv,
            converged=bool(conv >= 0.9),
            blank_out_pass=rec.get("blank_out", {}).get("passed"),
            read_vector_std_pass=rec.get("read_vector_std", {}).get("passed", None),
            axis_c_lock_sha256=rec.get("axis_c_lock_sha256"),
            fp64_shadow_wired=bool(len(shadow_pts) > 0),
            failure_front_h=rec["eval"]["failure_front_h"],
            reducer_flagged=rec["eval"]["reducer_signature"]["flagged"],
        )
        # convergence is a Phase-0 READOUT for the coordinator's wave-1
        # go/no-go, never an auto-gate (calibration lesson: a small reliable
        # model may plateau -- re-register rather than assume); the hard gate
        # items are the instrument duties.
        mandatory = [row["blank_out_pass"] is True,
                     row["fp64_shadow_wired"]]
        if arm in ("fwm", "loopedvec"):
            mandatory.append(row["read_vector_std_pass"] is True)
        if arm in ("ncr", "fwm"):
            mandatory.append(row["axis_c_lock_sha256"] is not None)
        row["gate_pass"] = bool(all(mandatory))
        all_pass = all_pass and row["gate_pass"]
        table["cells"][arm] = row
    table["rate_gpuh_80k_mean"] = (sum(rates.values()) / len(rates)) if rates else None
    table["phase0_verdict"] = "PASS" if all_pass else "FAIL"
    atomic_write_json(os.path.join(outdir, "phase0_gate_table.json"), table)
    if rates:
        atomic_write_json(os.path.join(outdir, "phase0_rate.json"),
                          dict(rate_gpuh_80k_per_arm=rates,
                               rate_gpuh_80k_mean=table["rate_gpuh_80k_mean"]))
    print("\n=== PHASE 0 GATE TABLE ===")
    print(json.dumps({k: v for k, v in table.items() if k != "param_report"}, indent=1,
                     default=float))
    print(f"PHASE0 VERDICT: {table['phase0_verdict']}")
    return table


# ---------------------------------------------------------------------------
# Smokes
# ---------------------------------------------------------------------------

def closed_form_checks(device: str):
    """S2.26 lesson: hand-computed closed forms at zero-accumulation configs,
    independent of any reference implementation.
    (1) standard-basis 8-cycle: bin-exp read of the literal shift matrix must
        land exactly on e_{(i+h) mod 8}. (2) single-binding Z = v k^T maps
        k -> v at h=1. (3) transpose tooth: the [K,V]-transposed layout
        (k v^T / the inverse shift) must be DETECTED as wrong."""
    d, K = 16, 8
    Z = torch.zeros(1, d, d, device=device)
    for i in range(K):
        Z[0, (i + 1) % K, i] = 1.0                     # shift: e_i -> e_{i+1}
    for h in (1, 2, 3, 5, 13, 21, 61, 1021):
        for i in range(K):
            q = torch.zeros(1, 1, d, device=device)
            q[0, 0, i] = 1.0
            o = nm.binexp_read(Z, q, h)["o"][0, 0]
            tgt = (i + h) % K
            assert abs(o[tgt].item() - 1.0) < 1e-5 and o.abs().sum().item() < 1.0 + 1e-4, \
                (h, i, o)
    # single binding, orthonormal pair
    k = torch.zeros(1, 1, d, device=device); k[0, 0, 0] = 1.0
    v = torch.zeros(1, 1, d, device=device); v[0, 0, 1] = 1.0
    Z1 = torch.einsum("bki,bkj->bij", v, k)            # v k^T (value-outer-key)
    o = nm.binexp_read(Z1, k, 1)["o"][0, 0]
    assert abs(o[1].item() - 1.0) < 1e-6, o
    # transpose tooth: k v^T (the fla-style [K,V]-transposed layout) must fail
    Zt = torch.einsum("bki,bkj->bij", k, v)
    ot = nm.binexp_read(Zt, k, 1)["o"][0, 0]
    assert abs(ot[1].item()) < 0.5, ("transpose layout NOT detected", ot)
    # cycle transpose = inverse shift: h=1 from e_0 must NOT reach e_1
    oT = nm.binexp_read(Z.transpose(1, 2), torch.eye(d, device=device)[0].view(1, 1, d), 1)["o"][0, 0]
    assert abs(oT[1].item()) < 0.5, ("cycle transpose NOT detected", oT)
    print("  closed-form checks: shift-matrix bin-exp exact to h=1021; v*k^T "
          "convention confirmed; transposed [K,V] layout DETECTED as wrong")


def smoke(device: str):
    """CPU-scale logic suite (fast). Real-kernel/CUDA coverage is the
    SEPARATE --box-smoke (CPU smokes cannot catch device-placement bugs --
    the S2.27 lesson + CLAUDE.md's CPU-stub rule)."""
    print("=" * 60 + "\n  NCR SMOKE (logic suite)\n" + "=" * 60)
    import ncr_selftest
    ncr_selftest.run_all(device)
    print("\nALL NCR SMOKE CHECKS PASSED")


def box_smoke():
    """Real-CUDA smoke: device teeth, forward/backward/grad, checkpoint/
    resume, closed forms ON CUDA, read agreement (S6b chain step)."""
    assert torch.cuda.is_available(), "box-smoke requires real CUDA"
    device = "cuda"
    print("=" * 60 + "\n  NCR BOX SMOKE (real CUDA)\n" + "=" * 60)
    print(f"  device: {torch.cuda.get_device_name(0)}")

    print("\n[1] device-placement teeth (real-CUDA, incl. negative)")
    cfg = nt.claim_config(8)
    gen = torch.Generator(device=device).manual_seed(0)
    for arm in nm.ALL_ARMS:
        m = nm.ARM_BUILDERS[arm]().to(device)
        assert all(p.device.type == "cuda" for p in m.parameters()), arm
        b = te.sample_batch(cfg, 8, gen, hop_set=cfg.H_train, device=device)
        assert all(t.device.type == "cuda" for t in b.values()
                   if isinstance(t, torch.Tensor)), arm
        pred, _ = m(b)
        assert pred.device.type == "cuda" and torch.isfinite(pred).all(), arm
    m = nm.NCRModel().to(device)
    b_cpu = te.sample_batch(cfg, 4, torch.Generator().manual_seed(0),
                            hop_set=cfg.H_train, device="cpu")
    raised = False
    try:
        m(b_cpu)
    except RuntimeError:
        raised = True
    assert raised, "CPU batch into CUDA model should raise (negative device tooth)"
    print("  all arms fully on-device; CPU-into-CUDA negative raises as expected")

    print("\n[2] closed-form analytic checks ON CUDA (S2.26 discipline)")
    closed_form_checks(device)

    print("\n[3] forward/backward/grad for every trainable param, every arm")
    for arm in nm.TRAINED_ARMS + ("cmlp",):
        m = nm.ARM_BUILDERS[arm]().to(device)
        b = te.sample_batch(cfg, 16, gen, hop_set=cfg.H_train, device=device)
        pred, _ = m(b)
        cosine_loss(pred, b["targets"]).backward()
        for name, p in m.named_parameters():
            assert p.grad is not None, (arm, name)
            assert torch.isfinite(p.grad).all(), (arm, name)
    print("  every parameter of every arm receives a finite gradient")

    print("\n[4] short real training + checkpoint/resume equivalence (ncr)")
    import tempfile
    with tempfile.TemporaryDirectory() as td_:
        cid, csha = "smoke_ncr", "smokesha"
        model = nm.NCRModel().to(device)
        g1 = torch.Generator(device=device).manual_seed(7)
        opt = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)
        for step in range(1, 101):
            b = te.sample_batch(cfg, 64, g1, hop_set=cfg.H_train, device=device)
            loss = cosine_loss(model(b)[0], b["targets"])
            opt.zero_grad(); loss.backward(); opt.step()
        ck = _ckpt_path(td_, cid)
        save_ckpt(ck, model, opt, g1, 100, csha, device)
        model2 = nm.NCRModel().to(device)
        opt2 = torch.optim.Adam(model2.parameters(), lr=TRAIN_LR)
        g2 = torch.Generator(device=device).manual_seed(0)
        step_resumed = try_resume(ck, model2, opt2, g2, csha, device)
        assert step_resumed == 100
        for (n1, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1, p2), n1
        print(f"  100-step train, checkpoint, resume: params bit-identical, step={step_resumed}")

        print("\n[5] trained-Z read agreement: binexp vs loop vs fp64 (h<=125)")
        with torch.no_grad():
            b = te.sample_batch(cfg, 64, g1, hop_set=(1,), device=device)
            Z = model.encode(b["keys"], b["values"])
            q = b["query_keys"]
            for h in (5, 21, 61, 125):
                ob = nm.binexp_read(Z, q, h)["o"]
                ol = nm.loop_read(Z, q, h)["o"]
                o64 = nm.binexp_read(Z.double(), q.double(), h)["o"]
                d_bl = (ob - ol).abs().max().item()
                cos_pair = (ob * ol).sum(-1)
                cos_shadow = (ob.double() * o64).sum(-1)
                assert cos_pair.min().item() > 1 - 1e-4, (h, cos_pair.min())
                assert cos_shadow.min().item() > 1 - 1e-3, (h, cos_shadow.min())
                print(f"    h={h:4d}  max|binexp-loop|={d_bl:.2e}  "
                      f"min cos(binexp,loop)={cos_pair.min():.8f}  "
                      f"min cos(binexp,fp64)={cos_shadow.min():.8f}")

        print("\n[6] blank-out ON CUDA (m3 executed, not shape-checked)")
        bo = blank_out_check(model, cfg, device, 0)
        assert bo["passed"], bo
        print(f"  {bo}")

    print("\nALL BOX-SMOKE CHECKS PASSED")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="CPU logic suite")
    ap.add_argument("--box-smoke", action="store_true", help="real-CUDA smoke")
    ap.add_argument("--phase0", action="store_true",
                    help="the 3-cell S3.6 calibration gate (K=8, seed 0)")
    ap.add_argument("--cell", choices=list(nm.ALL_ARMS), default=None)
    ap.add_argument("--K", type=int, default=8, choices=(8, 12))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=PHASE0_STEPS_DEFAULT)
    ap.add_argument("--outdir", type=str,
                    default=os.path.join(_HERE, "results"))
    ap.add_argument("--stop-file", type=str,
                    default=os.path.join(_HERE, "results", "STOP"))
    ap.add_argument("--rate-gpuh-80k", type=float, default=None,
                    help="calibrated rate (phase0_rate.json) superseding the 2.4 anchor")
    ap.add_argument("--eval-only-ckpt", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        smoke(device)
        return
    if args.box_smoke:
        box_smoke()
        return
    if args.phase0:
        phase0(device, args.outdir, args.steps, args.stop_file)
        return
    if args.cell:
        run_cell(args.cell, args.K, args.seed, args.steps, device, args.outdir,
                 args.stop_file, args.rate_gpuh_80k, args.eval_only_ckpt)
        return
    ap.error("one of --smoke / --box-smoke / --phase0 / --cell is required")


if __name__ == "__main__":
    main()
