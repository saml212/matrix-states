"""Wave -1 — zero-GPU items. STAGE_G_DESIGN.md section 4 item 1, section 7's
Wave -1 row, and section 12 build requirement (vi).

Four checks, all runnable on a laptop (no GPU, no cluster needed):

  1. Instrumented FLOP corroboration: analytic (section 5.1 formulas,
     flops.py) vs FlopCounterMode-instrumented count, at BOTH the original
     Regime-1 config (corroborating V1's 230.6M/15.45M/14.9x numbers) and
     Stage G's own Regime-2 standard-cell config.
  2. Step-0 residual-stream entry-std for BOTH Regime-1 models (H_j's free
     datum, design section 4 item 1: "F6 predicts ~1.0 (matrix) vs
     ~0.02-scale (LoopFormer)") -- using the VERBATIM common.py classes
     (round2_matrix_script.MatrixThinker-equivalent embedding logic /
     loopformer_96K_script.LoopFormer, unmodified), not the Regime-2
     harness.
  3. Data-availability check: local text.bin presence/size vs what the
     training box has at /home/nvidia -- reports size and shipping need,
     does NOT ship (per task instruction).
  4. Checkpoint-recoverability check (N1's local half already established
     by prior investigation; this script re-verifies it plus checks
     whether the retired pod is reachable via the Brev CLI).

Usage: python wave_neg1.py [--box-host youthful-indigo-turkey] [--ssd-path ...]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

import torch

from common import MatrixRMSNorm, ThinkingBlock, LoopFormer
from flops import analytic_matrix_flops, analytic_loopformer_flops, instrumented_flops_per_token
from models import MatrixThinkerG, MATRIX_ALL_BASELINE, VectorReferenceModel, VECTOR_ALL_BASELINE


# ═══════════════════════════════════════════════════════════════════════════
# 1. Instrumented FLOP corroboration
# ═══════════════════════════════════════════════════════════════════════════

def check_flops():
    print("\n" + "=" * 70 + "\n  [1] INSTRUMENTED FLOP CORROBORATION\n" + "=" * 70)
    report = {}

    # -- Regime 1 config (d=32, vocab=50257, L=512, n_layers=8, n_iter=8) --
    print("\n-- Regime 1 (original) config: d=32, vocab=50257, L=512, T=8 --")
    d, L, n_layers, n_iter, vocab, K = 32, 512, 8, 8, 50257, 32
    a_nc, bd_nc = analytic_matrix_flops(d, L, n_layers, n_iter, vocab, K, causal_exact=False)
    a_c, bd_c = analytic_matrix_flops(d, L, n_layers, n_iter, vocab, K, causal_exact=True)
    print(f"  analytic (non-causal): {a_nc/1e6:.2f}M/token  (V1-verified: 230.6M)")
    print(f"  analytic (causal-exact): {a_c/1e6:.2f}M/token")
    a_lf_nc, _ = analytic_loopformer_flops(96, 4, 2, 8, 240, L, vocab, causal_exact=False)
    a_lf_c, _ = analytic_loopformer_flops(96, 4, 2, 8, 240, L, vocab, causal_exact=True)
    print(f"  loopformer analytic (non-causal): {a_lf_nc/1e6:.2f}M/token  (V1-verified: 15.45M)")
    print(f"  RATIO non-causal: {a_nc/a_lf_nc:.2f}x (V1: 14.9x)   causal-exact: {a_c/a_lf_c:.2f}x (V1: 11.8x)")

    # Instrumented count needs an actual forward pass -- use a small-but-
    # structurally-identical config (full d=32/vocab=50257/L=512 is CPU-
    # feasible but slow; this corroborates the FORMULA/COUNTER machinery,
    # not the exact absolute number, which the analytic double-derivation
    # (this doc + V1) already covers within <1%). A moderate config keeps
    # this a genuine live measurement rather than a token gesture.
    d_i, L_i, n_layers_i, n_iter_i, vocab_i, K_i = 32, 64, 8, 8, 2000, 32
    torch.manual_seed(0)
    m = MatrixThinkerG(mat_dim=d_i, n_heads=8, n_iterations=n_iter_i, max_len=128,
                        vocab_size=vocab_i, n_layers=n_layers_i, **{k: v for k, v in
                        MATRIX_ALL_BASELINE.items() if k not in ("n_layers",)})
    ids = torch.randint(0, vocab_i, (1, L_i))
    inst_per_tok, inst_total, _ = instrumented_flops_per_token(m, ids, forward_kwargs={"n_iterations": n_iter_i})
    a_i_nc, _ = analytic_matrix_flops(d_i, L_i, n_layers_i, n_iter_i, vocab_i, K_i, causal_exact=False)
    a_i_c, _ = analytic_matrix_flops(d_i, L_i, n_layers_i, n_iter_i, vocab_i, K_i, causal_exact=True)
    print(f"\n-- Instrumented corroboration config: d={d_i}, vocab={vocab_i}, L={L_i}, T={n_iter_i} --")
    print(f"  instrumented (FlopCounterMode): {inst_per_tok:.0f}/token")
    print(f"  analytic non-causal: {a_i_nc:.0f}   analytic causal-exact: {a_i_c:.0f}")
    print(f"  ratio instrumented/non-causal: {inst_per_tok/a_i_nc:.3f}   "
          f"instrumented/causal-exact: {inst_per_tok/a_i_c:.3f}")
    print("  (expected: between the two conventions -- SDPA's causal masking is captured by "
          "FlopCounterMode's own formula, landing between the non-causal and causal-exact "
          "hand-derived bounds; see flops.py's docstring. NOT expected to exact-match either.)")

    report["regime1_analytic"] = {"non_causal_M": a_nc / 1e6, "causal_exact_M": a_c / 1e6,
                                   "loopformer_non_causal_M": a_lf_nc / 1e6,
                                   "ratio_non_causal": a_nc / a_lf_nc, "ratio_causal_exact": a_c / a_lf_c}
    report["instrumented_corroboration"] = {"config": {"d": d_i, "vocab": vocab_i, "L": L_i, "n_iter": n_iter_i},
                                             "instrumented_per_token": inst_per_tok,
                                             "analytic_non_causal": a_i_nc, "analytic_causal_exact": a_i_c}

    # -- Regime 2 standard-cell config (this build's own sizing) --
    print("\n-- Regime 2 standard-cell config: d=32, vocab=256, L=512, T=8 (matrix), "
          "n_embd=80 (vector, MEASURED param match -- audit MAJOR-2) --")
    d2, L2, n_layers2, n_iter2, vocab2, K2 = 32, 512, 8, 8, 256, 32
    a2_nc, bd2 = analytic_matrix_flops(d2, L2, n_layers2, n_iter2, vocab2, K2, causal_exact=False)
    a2_c, _ = analytic_matrix_flops(d2, L2, n_layers2, n_iter2, vocab2, K2, causal_exact=True)
    a2_lf_nc, _ = analytic_loopformer_flops(80, 4, 2, 8, 128, L2, vocab2, causal_exact=False)
    a2_lf_c, _ = analytic_loopformer_flops(80, 4, 2, 8, 128, L2, vocab2, causal_exact=True)
    print(f"  matrix analytic (non-causal): {a2_nc/1e6:.3f}M/token  breakdown={bd2}")
    print(f"  vector analytic (non-causal): {a2_lf_nc/1e6:.4f}M/token")
    print(f"  RATIO (Regime-2 standard cell) non-causal: {a2_nc/a2_lf_nc:.2f}x   "
          f"causal-exact: {a2_c/a2_lf_c:.2f}x")
    print("  NOTE: this ratio differs from Regime 1's 14.9x/11.8x because vocab collapses from "
          "50257->256 (design section 2.2/5.4: byte vocab shrinks the head-GEMM term that "
          "dominated Regime 1's total on BOTH sides) -- backbone-only ratio is the more "
          "comparable number across regimes; reported separately below.")
    print(f"  backbone-only ratio: {bd2['backbone']/analytic_loopformer_flops(80,4,2,8,128,L2,vocab2)[1]['backbone']:.2f}x")

    report["regime2_standard_cell"] = {
        "matrix_non_causal_M": a2_nc / 1e6, "matrix_causal_exact_M": a2_c / 1e6,
        "vector_non_causal_M": a2_lf_nc / 1e6, "vector_causal_exact_M": a2_lf_c / 1e6,
        "ratio_non_causal": a2_nc / a2_lf_nc, "ratio_causal_exact": a2_c / a2_lf_c,
        "matrix_breakdown": bd2,
    }
    return report


# ═══════════════════════════════════════════════════════════════════════════
# 2. Step-0 entry-std for BOTH Regime-1 models (H_j free datum)
# ═══════════════════════════════════════════════════════════════════════════

def check_step0_std(n_seeds=5):
    print("\n" + "=" * 70 + "\n  [2] STEP-0 RESIDUAL-STREAM ENTRY-STD (H_j, Regime-1 models)\n" + "=" * 70)
    print("  Using the VERBATIM common.py classes at the Run 12/13 config "
          "(d=32/n_embd=96, vocab=50257, default inits -- NO Stage-G intervention applied).")
    d, vocab, n_embd = 32, 50257, 96
    matrix_stds, vector_stds = [], []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        embed_u = torch.nn.Embedding(vocab, d)   # round2_matrix_script.py's DEFAULT init (std=1.0) -- unmodified
        embed_v = torch.nn.Embedding(vocab, d)
        ids = torch.randint(0, vocab, (4096,))
        with torch.no_grad():
            u, v = embed_u(ids), embed_v(ids)
            M = torch.einsum('...i,...j->...ij', u, v)
        matrix_stds.append(M.float().std().item())

        torch.manual_seed(seed)
        lf = LoopFormer(n_embd=n_embd, n_head=4, n_layer=2, n_loops=8, intermediate_dim=240,
                         max_len=64, vocab_size=vocab, dropout=0.0)
        ids2 = torch.randint(0, vocab, (1, 64))
        with torch.no_grad():
            x = lf.wte(ids2)
        vector_stds.append(x.float().std().item())

    m_mean = sum(matrix_stds) / n_seeds
    v_mean = sum(vector_stds) / n_seeds
    print(f"\n  Matrix (round2 default init, outer product u(x)v):  entry-std = {m_mean:.4f} "
          f"(over {n_seeds} seeds: {[f'{s:.4f}' for s in matrix_stds]})")
    print(f"  LoopFormer (explicit std=0.02 init, wte lookup):     entry-std = {v_mean:.4f} "
          f"(over {n_seeds} seeds: {[f'{s:.4f}' for s in vector_stds]})")
    print(f"\n  RATIO: {m_mean / v_mean:.1f}x   (design F6 predicts ~50x: matrix ~1.0 vs LoopFormer ~0.02)")
    return {"matrix_entry_std_mean": m_mean, "matrix_entry_std_per_seed": matrix_stds,
            "loopformer_entry_std_mean": v_mean, "loopformer_entry_std_per_seed": vector_stds,
            "ratio": m_mean / v_mean}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Data-availability check
# ═══════════════════════════════════════════════════════════════════════════

def check_data_availability(ssd_path, box_host):
    print("\n" + "=" * 70 + "\n  [3] DATA-AVAILABILITY CHECK (byte data, section 12)\n" + "=" * 70)
    report = {"ssd_path": ssd_path}
    if os.path.exists(ssd_path):
        size_mb = os.path.getsize(ssd_path) / 1e6
        print(f"  LOCAL: {ssd_path} exists, {size_mb:.1f} MB")
        report["local_exists"] = True
        report["local_size_mb"] = size_mb
    else:
        print(f"  LOCAL: {ssd_path} NOT FOUND")
        report["local_exists"] = False

    if box_host is None:
        print("  BOX: --box-host not given, skipping remote check")
        report["box_checked"] = False
        return report

    print(f"\n  Checking {box_host}:/home/nvidia/ for existing byte data or a matrix-thinking checkout ...")
    try:
        cmd = (
            "find /home/nvidia -maxdepth 4 \\( -iname '*text.bin*' -o -iname '*data_bytes*' "
            "-o -iname 'stageg' \\) 2>/dev/null; "
            "echo '---STAGEG_DIR---'; ls -la /home/nvidia/stageg 2>&1 | head -5; "
            "echo '---DISK---'; df -h /home/nvidia 2>&1 | tail -1"
        )
        out = subprocess.run(["ssh", box_host, cmd], capture_output=True, text=True, timeout=30)
        print(out.stdout)
        if out.stderr.strip():
            print("  stderr:", out.stderr.strip()[:500])
        report["box_checked"] = True
        report["box_raw_output"] = out.stdout
        report["box_has_stageg_dir"] = "No such file" not in out.stdout.split("---STAGEG_DIR---")[-1]
    except Exception as e:
        print(f"  SSH check failed (non-fatal): {e!r}")
        report["box_checked"] = False
        report["box_error"] = repr(e)

    if report.get("local_exists") and not report.get("box_has_stageg_dir", True):
        size_mb = report.get("local_size_mb", 0)
        print(f"\n  VERDICT: box has NO byte data staged yet. {size_mb:.1f} MB would need to be "
              f"scp'd (text.bin) plus ~1-5 MB of stageg/*.py. NOT shipping now (per task "
              f"instruction) -- ship with:\n"
              f"    scp {ssd_path} {box_host}:/home/nvidia/stageg_data/text.bin\n"
              f"    rsync -av {os.path.dirname(os.path.abspath(__file__))}/ "
              f"{box_host}:/home/nvidia/stageg/")
    return report


# ═══════════════════════════════════════════════════════════════════════════
# 4. Checkpoint recoverability (N1)
# ═══════════════════════════════════════════════════════════════════════════

def check_checkpoint_recovery():
    print("\n" + "=" * 70 + "\n  [4] RUN-12 CHECKPOINT RECOVERABILITY (N1)\n" + "=" * 70)
    report = {}

    local_ckpt_dir = os.path.join(REPO_ROOT, "experiment-runs", "8xh100-session1")
    found = subprocess.run(["find", local_ckpt_dir, "-iname", "*.pt"],
                            capture_output=True, text=True).stdout.strip()
    print(f"  Local .pt search under {local_ckpt_dir}: {found or '(none found)'}")
    report["local_pt_files"] = found.splitlines() if found else []

    ssd_ckpt = "/Volumes/1TB_SSD/learned-representations/checkpoints"
    if os.path.exists(ssd_ckpt):
        listing = subprocess.run(["find", ssd_ckpt, "-iname", "best.pt"],
                                  capture_output=True, text=True).stdout.strip()
        print(f"  SSD best.pt search: {listing or '(none found)'}")
        report["ssd_best_pt_files"] = listing.splitlines() if listing else []
        for f in report["ssd_best_pt_files"]:
            try:
                sd = torch.load(f, map_location="cpu", weights_only=True)
                keys = list(sd.keys())[:6]
                is_matrixthinker = any(k.startswith(("embed_u", "embed_v", "layers.", "output_head.")) for k in sd)
                print(f"    {f}: sample keys={keys} looks_like_MatrixThinker={is_matrixthinker}")
                report.setdefault("ssd_checkpoint_inspection", {})[f] = {
                    "sample_keys": keys, "looks_like_matrixthinker": is_matrixthinker}
            except Exception as e:
                print(f"    {f}: could not inspect ({e!r})")

    print("\n  Retired-pod (/toy_story_slam) reachability via Brev CLI:")
    try:
        out = subprocess.run(["brev", "ls"], capture_output=True, text=True, timeout=20)
        print(f"  {out.stdout.strip()}")
        report["brev_ls_output"] = out.stdout.strip()
        report["retired_pod_listed"] = "toy_story_slam" in out.stdout
    except Exception as e:
        print(f"  brev ls failed (non-fatal): {e!r}")
        report["brev_ls_error"] = repr(e)

    recoverable = bool(report.get("local_pt_files")) or report.get("retired_pod_listed", False)
    print(f"\n  VERDICT: Run-12 checkpoint recoverable = {recoverable} "
          f"({'MAINLINE fallback (N1) confirmed' if not recoverable else 'RECOVERY PATH FOUND -- re-check N1!'})")
    report["recoverable"] = recoverable
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-host", type=str, default="youthful-indigo-turkey")
    ap.add_argument("--ssd-path", type=str,
                     default="/Volumes/1TB_SSD/learned-representations/data/text.bin")
    ap.add_argument("--skip-box", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("  STAGE G -- WAVE -1 (ZERO-GPU ITEMS)")
    print("=" * 70)

    report = {}
    report["flops"] = check_flops()
    report["step0_std"] = check_step0_std()
    report["data_availability"] = check_data_availability(
        args.ssd_path, None if args.skip_box else args.box_host)
    report["checkpoint_recovery"] = check_checkpoint_recovery()

    print("\n" + "=" * 70)
    print("  WAVE -1 COMPLETE")
    print("=" * 70)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
