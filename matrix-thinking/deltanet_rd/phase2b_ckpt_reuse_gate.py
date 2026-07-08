"""phase2b_ckpt_reuse_gate.py -- REASONING_LINK_DESIGN.md sec 16.16.8's
registered sha256 reuse gate for Phase-2b's 30 reused OFF-arm checkpoints
(2 corpora x 3 seeds x 5 trajectory checkpoints), closing the exact
"silently scored a corrupted or wrong-version reused checkpoint" failure
mode the K=69 precedent (`run_deltanet_rd_exactness_sweep.
keyanchor_scaling_wide_k69_sha256_gate`) exists to prevent -- mirrored
here, IN PLACE (no copy step: unlike the K=69 precedent, these 30 files
already live at their production path, `results/phase2/ckpts/`, and are
never copied elsewhere before being read).

Independently RECOMPUTES sha256 of every file the pinned manifest names
and diffs against the PINNED digests (generated ONCE against the ORIGINAL
box archive, committed at
`experiment-runs/2026-07-08_phase2_familiarization/gates/
phase2b_off_ckpts_reuse_manifest.sha256` -- never regenerated from the
files being checked, which would make the check tautological). Fails
closed on ANY mismatch or missing file.

Run standalone:
    python phase2b_ckpt_reuse_gate.py --ckpt-dir results/phase2/ckpts \\
        --manifest experiment-runs/2026-07-08_phase2_familiarization/gates/phase2b_off_ckpts_reuse_manifest.sha256
    python phase2b_ckpt_reuse_gate.py --selftest   # negative test, run to completion
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Default path mirrors the K=69 precedent's own convention EXACTLY
# (run_deltanet_rd_exactness_sweep.KEYANCHOR_SCALING_WIDE_K69_PINNED_SHA256_PATH: `HERE/results/
# <name>.sha256`, a BOX-LOCAL path under this script's own directory) -- NOT `experiment-runs/`,
# which is a repo-root path never scp'd to the box (H100_SETUP.md: only `matrix-thinking/chapter2/`
# / `deltanet_rd/` itself is scp'd to `/home/nvidia/chapter2/`). The repo's own committed copy at
# `experiment-runs/2026-07-08_phase2_familiarization/gates/phase2b_off_ckpts_reuse_manifest.sha256`
# is the SOURCE OF TRUTH (git-tracked, crash-proof); deploying to a real box requires copying that
# SAME file content to this box-local path first (a deploy-step obligation, out of scope for this
# LOCAL BUILD -- disclosed here rather than silently assumed already in place).
DEFAULT_MANIFEST = os.path.join(HERE, "results", "phase2", "gates",
                                 "phase2b_off_ckpts_reuse_manifest.sha256")


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest(path: str) -> dict:
    pinned = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, name = line.split(None, 1)
            pinned[name.strip()] = digest.strip()
    return pinned


def sha256_reuse_gate(ckpt_dir: str, manifest_path: str) -> dict:
    """Returns {"ok": bool, "reason": str, "n_checked": int}. Fails closed on a missing manifest,
    missing file, or ANY digest mismatch -- never silently proceeds on a partial match."""
    if not os.path.exists(manifest_path):
        return {"ok": False, "reason": f"pinned sha256 manifest not found at {manifest_path!r}", "n_checked": 0}
    pinned = _parse_manifest(manifest_path)
    if not pinned:
        return {"ok": False, "reason": f"pinned manifest {manifest_path!r} contains zero entries", "n_checked": 0}
    for name, expected_digest in sorted(pinned.items()):
        fp = os.path.join(ckpt_dir, name)
        if not os.path.exists(fp):
            return {"ok": False, "reason": f"expected reused OFF checkpoint missing: {fp!r}",
                     "n_checked": 0}
        actual = _sha256_of_file(fp)
        if actual != expected_digest:
            return {"ok": False,
                     "reason": f"sha256 MISMATCH for {name!r}: expected {expected_digest} got "
                               f"{actual} -- the checkpoint diverged from the pinned box archive",
                     "n_checked": 0}
    return {"ok": True, "reason": f"all {len(pinned)} reused OFF checkpoints match the pinned "
                                    f"sec 16.16.8 sha256 manifest", "n_checked": len(pinned)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return 0 if _run_selftest() else 1

    if not args.ckpt_dir:
        print("usage: phase2b_ckpt_reuse_gate.py --ckpt-dir <dir> [--manifest <path>] | --selftest",
              file=sys.stderr)
        return 2

    result = sha256_reuse_gate(args.ckpt_dir, args.manifest)
    print(f"[phase2b-ckpt-reuse-gate] ckpt_dir={args.ckpt_dir}: {result['reason']}")
    if not result["ok"]:
        print("REFUSE: Phase-2b's 30-checkpoint sha256 reuse gate FAILED -- per sec 16.16.8's own "
              "belt-and-suspenders discipline, refusing to read any reused OFF checkpoint through a "
              "failed gate. Halting mechanically, not narrated.", file=sys.stderr)
        return 1
    print(f"PASS: Phase-2b's 30-checkpoint sha256 reuse gate PASSED ({result['n_checked']} files).")
    return 0


# ---------------------------------------------------------------------------
# Negative-test fixture suite -- run to completion (CLAUDE.md's "run the negative test to
# completion, don't just write it"): corrupt one byte of one reused .pt file's copy in a throwaway
# scratch dir, confirm sha256_reuse_gate (Python-level) AND a subprocess-level `sha256sum -c`
# invocation both fail (MINOR-1, sec 16.16.8's own "run it, don't just write it" discipline,
# mirroring the K=69 precedent exactly).
# ---------------------------------------------------------------------------

def _run_selftest() -> bool:
    import shutil
    import subprocess
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as td:
        real_dir = os.path.join(td, "real")
        os.makedirs(real_dir)
        fixture_names = [f"fake_ckpt_{i}.pt" for i in range(3)]
        for i, name in enumerate(fixture_names):
            with open(os.path.join(real_dir, name), "wb") as f:
                f.write((f"fixture-content-{i}" * 100).encode())

        manifest_path = os.path.join(td, "fixture_manifest.sha256")
        with open(manifest_path, "w") as f:
            f.write("# fixture manifest for phase2b_ckpt_reuse_gate --selftest\n")
            for name in fixture_names:
                digest = _sha256_of_file(os.path.join(real_dir, name))
                f.write(f"{digest}  {name}\n")

        # POSITIVE: untampered dir passes.
        r_pos = sha256_reuse_gate(real_dir, manifest_path)
        if not r_pos["ok"]:
            print(f"SELFTEST FAIL: positive fixture did not pass: {r_pos}")
            ok = False

        # NEGATIVE: corrupt one byte of ONE file in a scratch COPY, confirm both the Python gate
        # AND a real `sha256sum -c` subprocess reject it.
        copy_dir = os.path.join(td, "copy")
        shutil.copytree(real_dir, copy_dir)
        target = os.path.join(copy_dir, fixture_names[0])
        with open(target, "r+b") as f:
            f.seek(0)
            b = f.read(1)
            f.seek(0)
            f.write(bytes([b[0] ^ 0xFF]))  # flip one byte

        r_neg = sha256_reuse_gate(copy_dir, manifest_path)
        if r_neg["ok"]:
            print(f"SELFTEST FAIL: corrupted fixture INCORRECTLY passed: {r_neg}")
            ok = False
        elif "MISMATCH" not in r_neg["reason"]:
            print(f"SELFTEST FAIL: corrupted fixture failed for the WRONG reason: {r_neg['reason']!r}")
            ok = False

        # Missing-file negative: delete one file entirely.
        os.remove(os.path.join(copy_dir, fixture_names[1]))
        r_missing = sha256_reuse_gate(copy_dir, manifest_path)
        if r_missing["ok"]:
            print(f"SELFTEST FAIL: dir with a missing file INCORRECTLY passed: {r_missing}")
            ok = False

        # subprocess-level teeth: this module's own CLI, invoked for real, on the corrupted copy dir
        # (with the missing file restored so the mismatch is isolated to the corruption, not the
        # deletion) -- proves the enforcement has teeth at the process-boundary level, mirroring
        # phase2_gate_enforce._run_selftest's own subprocess-exit-code proof.
        shutil.copy2(os.path.join(real_dir, fixture_names[1]), os.path.join(copy_dir, fixture_names[1]))
        this_file = os.path.abspath(__file__)
        r_sub = subprocess.run([sys.executable, this_file, "--ckpt-dir", copy_dir,
                                 "--manifest", manifest_path], capture_output=True, text=True)
        if r_sub.returncode == 0:
            print(f"SELFTEST FAIL: subprocess on corrupted dir exited 0 (expected nonzero) -- the "
                  f"gate has NO TEETH. stdout={r_sub.stdout!r}")
            ok = False
        elif "REFUSE" not in r_sub.stderr:
            print(f"SELFTEST FAIL: corrupted-dir subprocess exited nonzero but did not print the "
                  f"REFUSE message: stderr={r_sub.stderr!r}")
            ok = False

        # ALSO exercise real `sha256sum -c` directly (the bash-level belt half of the
        # belt-and-suspenders pair sec 16.16.8 registers) -- skipped gracefully if unavailable
        # (e.g. some minimal CI images), never silently counted as a pass.
        if shutil.which("sha256sum"):
            check_file = os.path.join(td, "check.sha256")
            with open(manifest_path) as fin, open(check_file, "w") as fout:
                for line in fin:
                    if line.strip() and not line.startswith("#"):
                        fout.write(line)
            r_shasum = subprocess.run(["sha256sum", "-c", os.path.abspath(check_file)],
                                       cwd=copy_dir, capture_output=True, text=True)
            if r_shasum.returncode == 0:
                print(f"SELFTEST FAIL: bash-level `sha256sum -c` on corrupted dir exited 0 "
                      f"(expected nonzero). stdout={r_shasum.stdout!r}")
                ok = False

    if ok:
        print("phase2b_ckpt_reuse_gate --selftest: ALL CHECKS PASSED (positive fixture PASSES; "
              "one-byte-corrupted-copy fixture FAILS at the Python level, the subprocess exit-code "
              "level, AND the bash-level `sha256sum -c` level; missing-file fixture FAILS)")
    return ok


if __name__ == "__main__":
    sys.exit(main())
