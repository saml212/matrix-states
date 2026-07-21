"""READ-ONLY mechanistic probe -- sec G3-B23. NO training, NO writes to
results/. Characterizes the encoder-written operator Z from the COMPLETED
mob_g3b20_s0 checkpoint (ortho-reg push, non-TF, aux 3.0 + ortho 0.1) to
test whether it is a near-orthogonal, order-24 cycle operator -- independent
structural evidence, not the aux loss's own recovery-metric target.

Loads via the runner's OWN load_checkpoint/restore_arms_and_opts (no
reimplementation of the checkpoint format). Builds one fixed-seed batch of
grammar_rd Task-1 documents via the runner's own build_task1_document.
Computes Z = ncr_head.encode(keys_v, values_v) for BOTH arms (full_graft =
trained; backbone_only = frozen-at-random-init, since its o_injected is
zeroed with no autograd edge back to Z's own graph -- see
NCR_REAL_LM_DESIGN.md sec G3-B20 item 6). backbone_only therefore doubles as
the "random-init encoder" contrast the probe brief asked for -- no third
model needed.

Writes ONE small JSON to THIS directory (NOT results/) and prints numbers.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, "/home/nvidia/ncr_g3b12_fix")

import torch
import torch.nn.functional as F

import ncr_lm_wave1_runner as runner  # noqa: E402 -- triggers graft._setup_paths() internally

DEVICE = "cuda"
CKPT_PATH = "/home/nvidia/ncr_g3b12_fix/results/mob_g3b20_s0_ckpts/mob_g3b20_s0.ckpt.pt"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "z_ortho_probe_results.json")
PROBE_SEED = 20260721       # fixed, independent of the training data_gen stream
BATCH_SIZE = 64
HOP_VALUE = 1                # Z itself does not depend on hop_value (extract_kv is hop-independent);
                              # any TRAIN_HOPS value is fine for building the document.


def isfinite_frac(t: torch.Tensor) -> float:
    return torch.isfinite(t).float().mean().item()


@torch.no_grad()
def diagnose(Z: torch.Tensor, keys_v: torch.Tensor, values_v: torch.Tensor, label: str) -> dict:
    B, d, d2 = Z.shape
    assert d == d2
    Zf = Z.float()
    eye = torch.eye(d, device=Zf.device, dtype=Zf.dtype).expand(B, d, d)

    # (a) orthogonality via SVD
    sv = torch.linalg.svdvals(Zf)  # (B,d)
    ztz = torch.matmul(Zf.transpose(-1, -2), Zf)
    resid = ztz - eye
    resid_fro = resid.norm(dim=(-2, -1))  # (B,)

    # (b) eigenvalue magnitudes
    eigvals = torch.linalg.eigvals(Zf)  # (B,d) complex
    eig_mag = eigvals.abs()

    # (c) cyclicity -- Z^24, Z^12, Z^25, and the h=61-vs-13 (61 mod 24 == 13) periodicity check
    def mpow(k):
        Zp = torch.linalg.matrix_power(Zf, k)
        finite = isfinite_frac(Zp)
        return Zp, finite

    Z24, fin24 = mpow(24)
    Z12, fin12 = mpow(12)
    Z25, fin25 = mpow(25)
    Z61, fin61 = mpow(61)
    Z13, fin13 = mpow(13)

    d24 = (Z24 - eye).norm(dim=(-2, -1))
    d12 = (Z12 - eye).norm(dim=(-2, -1))
    d25mz = (Z25 - Zf).norm(dim=(-2, -1))
    d_period = (Z61 - Z13).norm(dim=(-2, -1))

    # (d) permutation-of-entities check: Z @ keys_v[i] vs values_v[i], the K
    # bind-clause pairs THIS batch's own succ K-cycle defines (entity_i ->
    # entity_succ(i)) -- keys_v/values_v are already entity_adapter(embed(.))
    # in Z's own space (sec G3-B12 single-space fix), so this is the direct
    # "does Z map entity_i -> entity_{succ(i)}" check, computed for all K
    # pairs in the batch at once (no separate construction needed).
    pred = torch.einsum("bij,bkj->bki", Z, keys_v)  # (B,K,d)
    cos_perm = F.cosine_similarity(pred, values_v, dim=-1)  # (B,K)

    def stat(t):
        tf = t[torch.isfinite(t)]
        if tf.numel() == 0:
            return {"min": None, "mean": None, "max": None, "finite_frac": 0.0}
        return {"min": tf.min().item(), "mean": tf.mean().item(), "max": tf.max().item(),
                "finite_frac": torch.isfinite(t).float().mean().item()}

    return {
        "label": label,
        "batch_size": B, "d": d,
        "singular_values": stat(sv),
        "ZtZ_minus_I_fro": stat(resid_fro),
        "eig_magnitude": stat(eig_mag),
        "Z24_minus_I_fro": stat(d24), "Z24_finite_frac": fin24,
        "Z12_minus_I_fro": stat(d12), "Z12_finite_frac": fin12,
        "Z25_minus_Z_fro": stat(d25mz), "Z25_finite_frac": fin25,
        "Z61_minus_Z13_fro_periodicity_check": stat(d_period),
        "Z61_finite_frac": fin61, "Z13_finite_frac": fin13,
        "permutation_cos_Zkey_vs_value": stat(cos_perm.reshape(-1)),
    }


def main():
    t0 = time.time()
    print(f"[probe] loading checkpoint {CKPT_PATH} ...", flush=True)
    ckpt = runner.load_checkpoint(CKPT_PATH, DEVICE)
    assert ckpt is not None, "checkpoint failed to load/validate -- STOP, do not force it"
    print(f"[probe] checkpoint step={ckpt['step']} runner_tag={ckpt['runner_tag']} "
          f"cell_id={ckpt['cell_id']}", flush=True)

    pools, cfg, pool_report = runner.build_grammar_pools_and_cfg(seed=0)  # matches launch --seed 0
    vocab_size_total = pool_report["vocab_size_total"]
    pools = pools.to(DEVICE)

    arms, opts, data_gen = runner.restore_arms_and_opts(ckpt, vocab_size_total, 3e-4, DEVICE)
    del opts, data_gen  # inference only, no optimizer/data-stream state needed
    for arm in arms.values():
        arm["backbone"].eval(); arm["ncr"].eval(); arm["integ"].eval()
    print(f"[probe] arms restored: {list(arms.keys())}", flush=True)

    gen = torch.Generator(device=DEVICE).manual_seed(PROBE_SEED)
    batch = runner.build_task1_document(cfg, pools, gen, BATCH_SIZE, HOP_VALUE, DEVICE)
    input_ids = batch["doc"][:, :-1]
    print(f"[probe] built batch: doc {tuple(batch['doc'].shape)}, K={cfg.K}", flush=True)

    results = {}
    for arm_name in ("full_graft", "backbone_only"):
        arm = arms[arm_name]
        with torch.no_grad():
            keys_v, values_v = arm["integ"].extract_kv(
                input_ids, batch["key_pos"], batch["val_pos"], arm["backbone"].embed)
            Z = arm["ncr"].encode(keys_v, values_v)
        print(f"[probe] {arm_name}: Z shape {tuple(Z.shape)}, keys_v {tuple(keys_v.shape)}", flush=True)
        results[arm_name] = diagnose(Z, keys_v, values_v, arm_name)

    # peak GPU memory this process used, for the "did not disturb production" record
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None

    out = {
        "probe": "sec_G3_B23_z_orthogonality_probe",
        "ckpt_path": CKPT_PATH, "ckpt_step": ckpt["step"], "ckpt_cell_id": ckpt["cell_id"],
        "probe_seed": PROBE_SEED, "batch_size": BATCH_SIZE, "hop_value": HOP_VALUE,
        "K": cfg.K, "d_ncr": arms["full_graft"]["ncr"].d if hasattr(arms["full_graft"]["ncr"], "d") else None,
        "results": results,
        "peak_gpu_mem_gb_this_process": peak_mem_gb,
        "elapsed_s": time.time() - t0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[probe] wrote {OUT_PATH}", flush=True)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
