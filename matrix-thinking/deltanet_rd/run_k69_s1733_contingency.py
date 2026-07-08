"""run_k69_s1733_contingency.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.20.4
MAJOR-2 disclosure + sec 15.22 next-step 2: launches EXACTLY the one
registered, reserved contingency cell K=69/seed=1733/d_state=96 (the
already-registered KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K[96][69] first
seed -- the SECOND, seed=1734, stays unfired/reserved per this design's own
"+2 contingency, fire only as needed" convention, and this RUN task
registers only seed 1733).

WHY A STANDALONE WRAPPER, NOT A NEW --wave/--scaling-wide-leg FLAG:
run_deltanet_rd_exactness_sweep.py's --wave keyanchor-scaling-wide only
exposes two legs (--scaling-wide-leg d96-wide: K in {72,78,84,90} x 3 seeds
fresh cells; d80-escalation: K in {48,53}'s own reserved seeds) -- neither
leg's manifest function includes a K=69/d=96 contingency cell anywhere
(keyanchor_scaling_wide_d96_manifest() explicitly excludes K=69, reusing the
already-harvested seeds 1730-1732 via copy+sha256 instead, per its own
docstring: "The ORIGINAL K=69 cells... are NOT re-launched here"). The
--wave keyanchor-scaling wave (the ORIGINAL, non-wide wave) also has no
seed-subset dispatch -- its own manifest is the fixed 3-seeds-per-K
keyanchor_scaling_manifest(96). No env var, flag, or leg mechanism in
run_deltanet_rd_exactness_sweep.py's CLI (grepped exhaustively this
session) can launch this one specific cell. Per the RUN task's own explicit
instruction, this wrapper is the minimal fallback: it calls
_keyanchor_scaling_spec()/build_cmd()/is_done()/default_timeout() --
the EXACT SAME functions the audited orchestrator itself uses for every
other K=69/d=96 cell -- directly, never hand-typing a single CLI flag.

VERIFICATION BEFORE LAUNCH (this script's own field-diff, run automatically
below, not a manual step): the spec/cmd this script builds for seed=1733 is
diffed token-by-token against the SAME functions' own output for seed=1730
(one of the three already-archived, audited K=69 cells) with every
seed-derived token normalized away. This script REFUSES to launch unless
that diff shows ONLY the seed-derived tokens differ (--seed value, --out
filename, --ckpt-dir path) -- i.e. unless the generated command is
byte-identical, module code, to what actually produced the archived
seed=1730/1731/1732 cells except for the seed itself.

GATES CHECKED (belt, mirroring keyanchor_scaling_wide_stage_gate's own
(a1)/(a2)/(b)/(c) -- this wrapper does NOT re-run the full smoke gauntlet,
per this RUN task's own explicit instruction that the wave's smoke gates
already passed on this box for this exact config):
  (a1) KEYANCHOR_SCALING_PI_SIGNOFF=1
  (a2) KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1
  (b)  ORIGINAL kernel-safety gate (T in {128,224,448}) -- already-committed
       artifact, re-checked not re-run.
  (c)  wide kernel-safety gate (T in {448,504,546,588,630}) -- already-
       committed artifact, re-checked not re-run.
  (t483) THIS RUN session's own NEW finding: T_bind(K=69)=483 sits strictly
       between gate (b)'s ceiling (448) and gate (c)'s floor (504) --
       covered by NEITHER committed artifact. Requires a PASSING
       results/smoke_dstate_kernel_t483_probe_result.json (run
       smoke_dstate_kernel_t483_probe.py FIRST, separately, before this
       script) -- this wrapper refuses to launch without it on file.

Usage (inside tmux, GPU 3 ONLY per this RUN task's hard constraint):
  CUDA_VISIBLE_DEVICES is set internally to "3" (not read from environment)
  so this script cannot be accidentally pointed at a different GPU.
  KEYANCHOR_SCALING_PI_SIGNOFF=1 KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1 \\
      /home/nvidia/tdenv/bin/python run_k69_s1733_contingency.py
"""
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_deltanet_rd_exactness_sweep as rdx  # noqa: E402

GPU = "3"  # HARD CONSTRAINT (RUN task): GPU 3 ONLY. Not read from environment on purpose.
K = 69
SEED = 1733
D_STATE = 96
T483_PROBE_ARTIFACT = os.path.join(HERE, "results", "smoke_dstate_kernel_t483_probe_result.json")


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    # --- Gates (a1)/(a2): PI signoff tokens, same as keyanchor_scaling_wide_stage_gate ---
    if os.environ.get("KEYANCHOR_SCALING_PI_SIGNOFF", "0") != "1":
        refuse("KEYANCHOR_SCALING_PI_SIGNOFF=1 not set (gate a1).")
    if os.environ.get("KEYANCHOR_SCALING_EXT_PI_SIGNOFF", "0") != "1":
        refuse("KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1 not set (gate a2, distinct from a1).")
    print("gates (a1)/(a2) PI signoff: OK", flush=True)

    # --- Gate (b): ORIGINAL kernel-safety gate, T in {128,224,448} ---
    kernel_gate = rdx.keyanchor_scaling_kernel_gate_check()
    if not kernel_gate["ok"]:
        refuse(f"ORIGINAL kernel-safety gate (b) FAILED: {kernel_gate['reason']}")
    print(f"gate (b) ORIGINAL kernel-safety: OK -- {kernel_gate['reason']}", flush=True)

    # --- Gate (c): wide kernel-safety gate, T in {448,504,546,588,630} ---
    wide_kernel_gate = rdx.keyanchor_scaling_wide_kernel_gate_check()
    if not wide_kernel_gate["ok"]:
        refuse(f"wide kernel-safety gate (c) FAILED: {wide_kernel_gate['reason']}")
    print(f"gate (c) wide kernel-safety: OK -- {wide_kernel_gate['reason']}", flush=True)

    # --- Gate (t483): THIS session's new finding -- T_bind(K=69)=483 is
    # covered by NEITHER (b) [ceiling 448] NOR (c) [floor 504]. ---
    t_bind = 7 * K
    assert t_bind == 483, f"T_bind formula drifted -- expected 483, got {t_bind}"
    covered_by_b = t_bind in set(rdx.KEYANCHOR_SCALING_KERNEL_GATE_T_SWEEP)
    covered_by_c = t_bind in set(rdx.KEYANCHOR_SCALING_WIDE_KERNEL_GATE_T_SWEEP) \
        if hasattr(rdx, "KEYANCHOR_SCALING_WIDE_KERNEL_GATE_T_SWEEP") else False
    print(f"T_bind(K={K})={t_bind} -- covered by gate (b) t_sweep: {covered_by_b}; "
          f"covered by gate (c) t_sweep: {covered_by_c}", flush=True)
    if not os.path.exists(T483_PROBE_ARTIFACT):
        refuse(f"gate (t483) artifact missing at {T483_PROBE_ARTIFACT!r} -- run "
                f"smoke_dstate_kernel_t483_probe.py on GPU 3 FIRST (T=483 is untested by both "
                f"existing kernel gates; sub-minute, no measurable GPU-h).")
    with open(T483_PROBE_ARTIFACT) as f:
        t483_doc = json.load(f)
    if t483_doc.get("exit_code") != 0 or "KERNEL-SAFE" not in t483_doc.get("verdict", ""):
        refuse(f"gate (t483) FAILED: {t483_doc.get('verdict')!r} (exit_code="
                f"{t483_doc.get('exit_code')!r}) -- contingency seed 1733 must NOT launch.")
    print(f"gate (t483) T=483 probe: OK -- {t483_doc['verdict']}", flush=True)

    # --- Build the spec via the SAME builder every other K=69/d=96 cell used ---
    out_dir = os.path.join(rdx.DEFAULT_OUT_DIR, "wavekeyanchor-scaling-wide")
    ckpt_base_dir = "/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling-wide"
    spec = rdx._keyanchor_scaling_spec(K, SEED, D_STATE)
    print(f"spec: {spec}", flush=True)

    if rdx.is_done(out_dir, spec):
        print(f"ALREADY DONE: {rdx.out_path(out_dir, spec)!r} validity-checks as complete -- "
              f"nothing to launch (idempotent no-op).", flush=True)
        return 0

    timeout = rdx.default_timeout(spec["K"], spec["steps"])
    cmd = rdx.build_cmd(spec, out_dir, timeout, unblind_override_at=None, ckpt_base_dir=ckpt_base_dir)

    # --- Field-diff verification against an archived seed (1730), seed field excepted ---
    ref_spec = rdx._keyanchor_scaling_spec(K, 1730, D_STATE)
    ref_cmd = rdx.build_cmd(ref_spec, out_dir, rdx.default_timeout(ref_spec["K"], ref_spec["steps"]),
                             unblind_override_at=None, ckpt_base_dir=ckpt_base_dir)

    def normalize(c, seed):
        return [tok.replace(str(seed), "SEED") for tok in c]

    n_new = normalize(cmd, SEED)
    n_ref = normalize(ref_cmd, 1730)
    if n_new != n_ref:
        print("FIELD-DIFF MISMATCH (beyond the seed field):", file=sys.stderr)
        for a, b in zip(n_new, n_ref):
            if a != b:
                print(f"  new={a!r} ref={b!r}", file=sys.stderr)
        refuse("generated command diverges from the archived-seed reference command in a field "
                "OTHER than seed -- refusing to launch. Do not override without human review.")
    print("field-diff vs. archived seed=1730 (same builder, seed field excepted): MATCH", flush=True)
    print(f"sha256(cmd)={hashlib.sha256(' '.join(cmd).encode()).hexdigest()}", flush=True)
    print(f"CMD: {' '.join(cmd)}", flush=True)
    print(f"timeout={timeout}s (~{timeout/60:.1f} min); GPU={GPU} (CUDA_VISIBLE_DEVICES, hard-pinned)",
          flush=True)

    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{spec['name']}.log")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU}
    start = time.time()
    print(f"LAUNCHING at {time.strftime('%Y-%m-%d %H:%M:%S')} -- log: {log_path}", flush=True)
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    wall_s_wrapper = time.time() - start
    print(f"FINISHED rc={rc} wrapper_wall_s={wall_s_wrapper:.1f} "
          f"({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)

    ok = rc == 0 and rdx.is_done(out_dir, spec)
    if not ok:
        print(f"FAILED: rc={rc} is_done={rdx.is_done(out_dir, spec)} -- see {log_path}",
              file=sys.stderr)
        return 1

    result_path = rdx.out_path(out_dir, spec)
    with open(result_path) as f:
        result = json.load(f)
    wall_s = result.get("wall_s")
    is_anchor = spec["K"] == rdx.KEYANCHOR_SCALING_ANCHOR_K_BY_D[D_STATE]
    trigger = rdx.KEYANCHOR_SCALING_ABORT_WALL_S[(D_STATE, is_anchor)]
    print(f"SUCCESS: {result_path}", flush=True)
    print(f"  wall_s={wall_s} (sec 15.14 abort trigger for this (d_state, is_anchor)={trigger:.1f}s; "
          f"{'OVER TRIGGER -- would have aborted a real chain' if wall_s and wall_s >= trigger else 'within bracket'})",
          flush=True)
    ga = result.get("geo3_admission", {})
    print(f"  geo3_admission: ns_converged_no_fallback={ga.get('ns_converged_no_fallback')} "
          f"value_salvage_tier_pass={ga.get('value_salvage_tier_pass')} "
          f"task_performance_floor_pass={ga.get('task_performance_floor_pass')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
