#!/usr/bin/env python3
"""R0 v2 MEASUREMENT DRIVER -- loops the FROZEN, VALIDATED v2 recall-gap probe
over the 8 R0 checkpoints (4 rungs x 2 corpora), one DiD cell JSON per
(rung, corpus) into a QUARANTINED output dir.

This driver LOOPS the instrument; it does NOT reimplement it. Each cell is a
fresh `lm_recall_gap_probe_v2_rd.py --run` SUBPROCESS -- process isolation gives
clean per-cell GPU teardown and guarantees byte-for-byte reuse of the frozen
probe (md5 652b479e...). The probe REFUSES `--compute-verdict` by design; this
driver honours that and emits NO verdict. The verdict is computed later, by the
BLIND `param_axis_r0_betafit.py`, from the quarantined cells.

Provenance / gates this driver respects:
  * PARAM_AXIS_SCALING_DESIGN.md sec 32 -- INSTRUMENT_VALID = TRUE (precondition).
  * sec 9.6 item 3 -- checkpoints must be QUIESCED (probe's own load_checkpoint
    quiescence guard is left ON) AND their training job terminated (the probe's
    REQUIRED `--attest-job-terminated` operator attestation; runs 033/034 etc.
    are in queue/completed, sec 33.4).
  * sec 11.7 -- N_rows is fixed by a MODEL-FREE pre-pass (`--nrows-prepass`,
    below), rung-invariant, and floored (>=1,500 contributing rows AND >=8,000
    resolved candidates per (rung, corpus)). The pinned/ratified value is 2048.
  * sec 33 (record-first) -- GATE-A FAILS on {14M,98M,392M} (d_state 2 != 1) and
    GATE-1 strikes FLAT at n_seeds=1. Those are the FIT layer's concern
    (param_axis_r0_betafit.py); this driver only produces the measurement cells.

USAGE (Phase 2, on the box, a DRAINED single GPU):
  # (optional but sec-11.7-mandated) fix N_rows model-free, once, over both corpora:
  python3 param_axis_r0_v2_driver.py --nrows-prepass --gpu 0
  # price ONE cell's wall-clock first (14M / openr1) to size the timeout:
  python3 param_axis_r0_v2_driver.py --smoke --gpu 0
  # then the full 8-cell loop (resume-safe; re-run to fill any missing cell):
  python3 param_axis_r0_v2_driver.py --gpu 0

This build session NEVER runs any of the above against a real checkpoint (that
is Phase 2, a separate runner). `--dry-run` prints the exact probe commands.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(SCRIPT_DIR, "lm_recall_gap_probe_v2_rd.py")

# The frozen, VALIDATED instrument (sec 32). The driver REFUSES to run against a
# modified probe -- looping a changed instrument would silently break the pin.
PROBE_MD5_PINNED = "652b479ee0cb4d9fd6e302a65d4a949f"

CKPT_ROOT = "/data/fixscale_ckpts/train"
CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")   # sec 9.6 item 6: BOTH, always.
CKPT_STEP = 67547                                   # sec 22.3 common slice T=1.10669B tokens.

# sec 11.7 THE PIN: N_rows ratified at 2048 (R0's own disclosed D1 deviation);
# the model-free pre-pass may only RAISE it, never lower it. C_max is per-row,
# rung-invariant (kills F-4). Both are passed EXPLICITLY so a future change to the
# probe's argparse defaults cannot silently move the sampling under us.
N_ROWS_PINNED = 2048
C_MAX_PINNED = 8

DEFAULT_QUARANTINE_DIR = os.path.join(SCRIPT_DIR, "results", "param_axis_r0_cells")

# The rung table. params are the design's nominal counts (DSTATE_CONFOUND_PREREG.md
# sec 1, filename-confirmed: lmC_<corpus>_dm<dm>_ds<ds>_L<L>_s<seed>_step<step>.pt).
# `seed_label` differs at 98M (s0 vs s3) -- n_seeds=1 per rung regardless; disclosed
# by the fit layer, carried here only so the ckpt dir resolves.
RUNGS = [
    # rung  d_state  nominal_params  dir_template                                                   seed_label
    ("14M",   64,   14_048_896,  "fixscale_fulltoken_arm_per_token_14m_{corpus}_s3",           "s3"),
    ("98M",   64,   97_618_176,  "fixscale_train_arm_per_token_98m_{corpus}_s0",               "s0"),
    ("392M",  128,  391_869_440, "fixscale_fulltoken_arm_per_token_392m_{corpus}_s3",          "s3"),
    ("Y",     64,   385_577_984, "fixscale_fulltoken_arm_per_token_392mY_ds64_{corpus}_s3",    "s3"),
]


def build_cells():
    """One cell spec per (rung, corpus) -- 4 rungs x 2 corpora = 8 cells."""
    cells = []
    for rung, d_state, n_params, dir_tmpl, seed_label in RUNGS:
        for corpus in CORPORA:
            ckpt_dir = os.path.join(CKPT_ROOT, dir_tmpl.format(corpus=corpus))
            cells.append({
                "rung": rung, "corpus": corpus, "d_state": d_state,
                "nominal_params": n_params, "seed_label": seed_label,
                "ckpt_dir": ckpt_dir,
                "ckpt_glob": os.path.join(ckpt_dir, f"*_step{CKPT_STEP}.pt"),
                "cell_name": f"{rung}_{corpus}_step{CKPT_STEP}",
            })
    return cells


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_probe_md5():
    """Refuse to loop a modified instrument."""
    got = md5_of(PROBE)
    if got != PROBE_MD5_PINNED:
        raise SystemExit(
            f"REFUSING TO RUN: probe md5 {got} != pinned {PROBE_MD5_PINNED}.\n"
            f"  {PROBE}\n"
            f"The R0 measurement must loop the FROZEN, VALIDATED instrument (sec 32). "
            f"A changed probe silently breaks the pin -- restore it or re-validate before looping."
        )
    return got


def resolve_ckpt(cell):
    """Resolve the EXACT step-67547 checkpoint file; assert exactly one match."""
    matches = sorted(glob.glob(cell["ckpt_glob"]))
    if len(matches) == 0:
        raise SystemExit(f"{cell['cell_name']}: NO checkpoint matches {cell['ckpt_glob']}")
    if len(matches) > 1:
        raise SystemExit(f"{cell['cell_name']}: AMBIGUOUS -- {len(matches)} match {cell['ckpt_glob']}: {matches}")
    return matches[0]


def cell_output_valid(path):
    """Resume-safety: a cell is DONE only if its output is a STRUCTURALLY valid,
    verdict-grade DiD cell at the right step -- not merely present (CLAUDE.md:
    'skip already-completed work by checking output validity, not just existence').
    NOTE: validity here is structural ONLY -- it reads sample-size / provenance
    fields, never `did` / `did_ci` / `t1a_*` (no outcome value is inspected)."""
    if not os.path.exists(path):
        return False, "missing"
    try:
        with open(path) as f:
            cell = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"unparseable ({e})"
    if not cell.get("_verdict_grade"):
        return False, "not verdict-grade"
    if cell.get("did") is None:                      # presence check, value NOT read
        return False, "did is None (VerdictGradeError cell)"
    if cell.get("ckpt_step") != CKPT_STEP:
        return False, f"ckpt_step {cell.get('ckpt_step')} != {CKPT_STEP}"
    if not cell.get("checkpoint_md5"):
        return False, "no checkpoint_md5 pin (sec 9.6 item 3)"
    if cell.get("n_candidates_resolved") is None:
        return False, "no n_candidates_resolved"
    return True, "valid"


def probe_env(args):
    env = dict(os.environ)
    # Fragmentation-tolerant allocator; models are <=392M (<=~1.75GB) so this is
    # belt-and-braces on a drained 80GB card.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)   # single-GPU pin.
    return env


def build_probe_cmd(cell, ckpt_path, out_path, n_rows, args):
    """The probe's own `--run` contract (argparse-verified against
    lm_recall_gap_probe_v2_rd.py main()). NEVER pass --no-quiesce-check (we WANT
    sec 9.6 item 3's live-writer guard), --compute-verdict (refused), or
    --normalization (refused)."""
    cmd = [
        sys.executable, PROBE, "--run",
        "--checkpoint", ckpt_path,
        "--corpus", cell["corpus"],
        "--rung", cell["rung"],
        "--n-windows", str(n_rows),
        "--c-max", str(C_MAX_PINNED),
        "--attest-job-terminated",           # REQUIRED by mode_run (sec 9.6 item 3, 2nd clause).
        "--out", out_path,
    ]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.eval_micro_batch is not None:
        cmd += ["--eval-micro-batch", str(args.eval_micro_batch)]
    if args.device is not None:
        cmd += ["--device", args.device]
    return cmd


def resolved_n_rows(out_dir):
    """Use the model-free pre-pass's N_rows if it was run (sec 11.7 authoritative),
    else fall back to the ratified floor 2048 with a loud note. The pre-pass may
    only RAISE N_rows, never lower it -- enforced here."""
    pp = os.path.join(out_dir, "nrows_prepass.json")
    if os.path.exists(pp):
        with open(pp) as f:
            d = json.load(f)
        if d.get("void"):
            raise SystemExit(f"sec 11.7 pre-pass VOID: {d.get('void_reason')}")
        n = int(d["n_rows"])
        if n < N_ROWS_PINNED:
            raise SystemExit(f"pre-pass N_rows {n} < ratified floor {N_ROWS_PINNED} -- impossible; investigate.")
        return n, "prepass"
    return N_ROWS_PINNED, "ratified-default"


def run_nrows_prepass(args):
    """sec 11.7: fix N_rows model-free (NO checkpoint loaded), over BOTH corpora,
    BEFORE any cell runs. Imports the frozen probe LAZILY (its top-level import
    calls ensure_fla_stub()) so this driver stays import-light for local syntax
    checks. Writes nrows_prepass.json into the quarantine dir; VOID aborts."""
    sys.path.insert(0, SCRIPT_DIR)
    import lm_recall_gap_probe_v2_rd as probe  # noqa: E402 -- lazy, box-only path
    from lm_pretrain_rd import load_corpus, DEFAULT_DATA_DIR  # noqa: E402

    device = args.device or "cuda"
    val_by_corpus, mode_by_corpus = {}, {}
    for corpus in CORPORA:
        _, val_tokens, _, _, _ = load_corpus(DEFAULT_DATA_DIR, corpus, device)
        # mode table is model-free (train modal bigram); build via the probe's own fn.
        train_tokens, _, _, _, _ = load_corpus(DEFAULT_DATA_DIR, corpus, device)
        val_by_corpus[corpus] = val_tokens
        mode_by_corpus[corpus] = probe.build_bigram_mode_table(train_tokens, probe.VOCAB_SIZE, device)
    res = probe.resolve_n_rows_pre_pass(val_by_corpus, mode_by_corpus, args.seq_len, device,
                                        c_max=C_MAX_PINNED)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "nrows_prepass.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    if res.get("void"):
        raise SystemExit(f"sec 11.7 pre-pass VOID (written to {out}): {res.get('void_reason')}")
    print(f"sec 11.7 pre-pass: N_rows = {res['n_rows']} (>= ratified floor {N_ROWS_PINNED}). Wrote {out}")
    return 0


def run_cell(cell, out_dir, n_rows, args):
    """Run ONE probe --run subprocess. Captures the probe's stdout/stderr (which
    would echo `did`) into a QUARANTINED per-cell log so it never lands in this
    driver's own stdout -- the coordinator stays blind. Returns a status dict of
    SAMPLE-SIZE / provenance fields only (no outcome value)."""
    out_path = os.path.join(out_dir, cell["cell_name"] + ".json")
    log_path = os.path.join(out_dir, cell["cell_name"] + ".probe.log")

    if args.dry_run:
        # Preview the exact probe command. Tolerate a missing checkpoint (e.g. run
        # off-box) by falling back to the glob pattern so the command is still shown.
        try:
            ckpt_path = resolve_ckpt(cell)
        except SystemExit:
            ckpt_path = cell["ckpt_glob"] + "  (UNRESOLVED off-box)"
        cmd = build_probe_cmd(cell, ckpt_path, out_path, n_rows, args)
        print(f"[DRY-RUN] {cell['cell_name']}:\n  {' '.join(cmd)}")
        return {"cell": cell["cell_name"], "dry_run": True, "cmd": cmd}

    ckpt_path = resolve_ckpt(cell)
    cmd = build_probe_cmd(cell, ckpt_path, out_path, n_rows, args)

    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=probe_env(args), stdout=logf,
                              stderr=subprocess.STDOUT, timeout=args.timeout_s)
    elapsed = time.time() - t0

    status = {"cell": cell["cell_name"], "exit_code": proc.returncode,
              "elapsed_s": round(elapsed, 1), "out": out_path, "log": log_path}
    ok, why = cell_output_valid(out_path)
    status["output_valid"] = ok
    status["validity"] = why
    if ok:
        # sample-size / provenance ONLY -- these are NOT recall outcomes.
        with open(out_path) as f:
            c = json.load(f)
        status["n_candidates_resolved"] = c.get("n_candidates_resolved")
        status["contributing_rows"] = len(c.get("per_row_did") or {})
        status["checkpoint_md5"] = c.get("checkpoint_md5")
        status["ckpt_step"] = c.get("ckpt_step")
        status["cell_void_placebo_match"] = c.get("cell_void_placebo_match")
        status["cell_void_missing_s2_fields"] = c.get("cell_void_missing_s2_fields")
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=DEFAULT_QUARANTINE_DIR,
                    help="QUARANTINED cell dir the coordinator does not read.")
    ap.add_argument("--gpu", type=int, default=None, help="CUDA_VISIBLE_DEVICES pin (single GPU).")
    ap.add_argument("--only", default=None, help="Run only this rung label (e.g. 14M).")
    ap.add_argument("--corpus", default=None, choices=list(CORPORA), help="Run only this corpus.")
    ap.add_argument("--smoke", action="store_true",
                    help="Single-cell price check: run ONE cell (default 14M/openr1) and print its wall-clock.")
    ap.add_argument("--nrows-prepass", action="store_true",
                    help="sec 11.7 model-free N_rows pre-pass over BOTH corpora (NO checkpoint). Run FIRST.")
    ap.add_argument("--dry-run", action="store_true", help="Print the exact probe commands; run nothing.")
    ap.add_argument("--force", action="store_true", help="Re-run cells even if a valid output exists.")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=None, help="Override probe default (16).")
    ap.add_argument("--eval-micro-batch", type=int, default=None, help="Override probe default (64).")
    ap.add_argument("--device", default=None, help="Override probe default (cuda).")
    ap.add_argument("--timeout-s", type=int, default=3600,
                    help="Per-cell subprocess timeout. Price the 14M cell (--smoke) first to size it.")
    args = ap.parse_args()

    verify_probe_md5()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.nrows_prepass:
        return run_nrows_prepass(args)

    n_rows, n_rows_src = (N_ROWS_PINNED, "ratified-default") if args.dry_run else resolved_n_rows(args.out_dir)
    if n_rows_src == "ratified-default" and not args.dry_run:
        print(f"NOTE: no nrows_prepass.json in {args.out_dir}; using ratified N_rows={N_ROWS_PINNED} "
              f"(sec 11.7). Run `--nrows-prepass` first to VERIFY the floors model-free.")

    cells = build_cells()
    if args.only:
        cells = [c for c in cells if c["rung"] == args.only]
    if args.corpus:
        cells = [c for c in cells if c["corpus"] == args.corpus]
    if args.smoke:
        cells = [c for c in cells if c["rung"] == "14M" and c["corpus"] == "openr1-mix-ext"] or cells[:1]

    print(f"probe md5 OK ({PROBE_MD5_PINNED}); N_rows={n_rows} ({n_rows_src}); "
          f"{len(cells)} cell(s) -> {args.out_dir}")

    results = []
    for cell in cells:
        out_path = os.path.join(args.out_dir, cell["cell_name"] + ".json")
        if not args.force and not args.dry_run:
            ok, why = cell_output_valid(out_path)
            if ok:
                print(f"  SKIP {cell['cell_name']} (valid output exists)")
                results.append({"cell": cell["cell_name"], "skipped": True})
                continue
        print(f"  RUN  {cell['cell_name']} ...", flush=True)
        status = run_cell(cell, args.out_dir, n_rows, args)
        # Print SAMPLE-SIZE / provenance status ONLY (never `did`).
        print("    " + json.dumps({k: v for k, v in status.items() if k != "cmd"}), flush=True)
        results.append(status)

    if args.smoke:
        done = [r for r in results if r.get("elapsed_s") is not None]
        if done:
            print(f"\nSMOKE price: {done[0]['cell']} took {done[0]['elapsed_s']} s. "
                  f"Size the full-loop timeout from this (392M ~10x slower).")
    print("\nNOTE: NO verdict emitted (the probe refuses --compute-verdict). "
          "Run param_axis_r0_betafit.py (fresh BLIND agent) on the quarantined cells.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
