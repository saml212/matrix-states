"""run_poolmargin_k84s1943_k90s2043.py -- KEY_ANCHORING_SCALING_DRAFT.md
sec 15.26 (Rev 2, c66b3f6, DESIGN-CLEARED-FOR-BUILD at the round-3 VERIFY
pass, fd8c509/16f3543): launches the K90 POOL-MARGIN CONTROL DIAGNOSTIC's
own EXACTLY 2 cells -- K=84/seed=1943 (overlap-equalized, restricted eval)
and K=90/seed=2043 (natural-margin comparator) -- both d_state=96,
REDIRECTED, disclosed reuse of 2 of the killed 10-cell scatter-resolution
grid's own now-idle reserved contingency seeds (sec 15.26.2.3).

WHY A STANDALONE WRAPPER, NOT A NEW --wave/--scaling-wide-leg FLAG (sec
15.26.3.1, round-1 MAJOR-2 fix): a small, two-cell, non-manifest launch
does not fit the `--wave keyanchor-scaling-wide` manifest abstraction
cleanly -- the EXACT SAME reasoning run_k69_s1733_contingency.py's own
module docstring already used for its own single contingency cell. This
wrapper mirrors that precedent's own shape line-for-line: it calls
_keyanchor_scaling_spec()/build_cmd()/is_done()/default_timeout() -- the
SAME functions the audited orchestrator itself uses for every other
keyanchor-scaling cell -- directly, never hand-typing a CLI flag, and
RE-IMPLEMENTS both PI-signoff checks itself (mirroring the precedent's own
main() lines 85-88) rather than deferring to run_deltanet_rd_exactness_
sweep.py's central --wave dispatcher gate (disclosed, sec 15.26.3.1).

WHAT'S NEW RELATIVE TO THE PRECEDENT (sec 15.26.3.1's own round-2 MAJOR-3
fix): the precedent's own field-diff/token-diff check (refuse unless the
generated command matches a sibling-seed reference command with only
seed-derived tokens differing) cannot pass VERBATIM for K=84's own cell,
whose real command carries an extra `--m3-pool-restrict-n 100` token
build_cmd() does not know how to generate. Fixed here by stripping an
explicitly-enumerated, PER-K whitelist (NEW_FLAG_WHITELIST_BY_K, build-audit
MAJOR-3 fix -- keyed by K, not global, so an errant flag on K=90's own cell
is never silently absorbed through K=84's own whitelist entry) from the
generated command BEFORE the precedent's own equality-diff runs
(strip_whitelisted() below, build-audit MAJOR-2 fix: refuses loudly on a
DUPLICATED whitelisted flag rather than stripping every occurrence), so the
check keeps its teeth against every OTHER kind of drift. K=90's own
whitelist entry is EMPTY -- its own whitelist strip is a documented no-op,
and any extra token on that cell's cmd, including this same flag, refuses.

n_iter=28 (sec 15.26.3, REUSED verbatim from the killed grid's own
registered bump, additive-only): applied as a POST-CONSTRUCTION override
on the spec _keyanchor_scaling_spec() returns -- rdx.KEYANCHOR_SCALING_
GATE2_N_ITER_BY_D_K (the dict that function reads n_iter FROM) is NEVER
touched (sec 15.26.0 item 4's own build-scope fence).

GATES CHECKED (sec 15.26.6's own table; letters match that table, not the
precedent's own (a1)/(b)/(c)/(t483) lettering):
  (a1) kernel-safety, T in {588,630} -- rdx.keyanchor_scaling_wide_kernel_
       gate_check(), REUSED, no new probe (T_bind(84)=588, T_bind(90)=630,
       both already inside the WIDE T-sweep {448,504,546,588,630}; the
       ORIGINAL narrow gate's own {128,224,448} sweep does NOT cover
       either value, so only the wide check applies here).
  (b)  n_iter-sufficiency (static anchor table) -- REUSED by disclosed
       reasoning (sec 15.26.3), not re-run at n_iter=28; no live check.
  (c)  sha256 reused-JSON manifest -- N/A, both cells are fresh launches.
  (d)  PI signoff, primary + extension -- KEYANCHOR_SCALING_PI_SIGNOFF=1
       AND KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1, both required (sec
       15.26.3.1's own disclosed reasoning: the 2x pessimistic bracket
       marginally exceeds the ORIGINAL 21 GPU-h ceiling, sec 15.26.4).
  (e)  n_iter-bump negative test -- REUSED, already validated twice on
       this box (sec 15.24.2/15.25.2); no live check here.
  (f)  admission-recalibration gate -- post-hoc `geo3_admission.
       admissible is True` check on both landed cells (check_admission()
       below), DISCLOSED (not hard-aborting) on failure per sec 15.26.5's
       own degenerate-cell trigger row.
  (g)  seed-space collision check -- grep across matrix-thinking/deltanet_
       rd/*.py + experiment-runs/ for '_s1943_'/'_s2043_', zero hits
       expected in any archived result JSON (mechanical, re-run this
       session, sec 15.26.2.3).
  (h)  calibration-first launch -- K=84/seed=1943 alone first, abort
       trigger REUSED verbatim from the killed grid's own derivation
       (SAME 0.427 base rate, sec 15.26.4 item 1): 1.5*0.427*2*3600 =
       4611.6s. If it fires: halt, do not launch K=90.
  (i)  launch-mechanism negative test (PI-signoff refusal, all 3 missing-
       token combinations) -- `--self-test` mode below, run to completion
       before either real cell launches.
  (j)  field-diff whitelist negative test (one extra, non-whitelisted
       token must still refuse) -- `--self-test` mode below.
  (k)  noise-floor measurement precondition -- mechanical, enforced by
       harvest_poolmargin_k84s1943_k90s2043.py, not this wrapper.

Usage (inside tmux, GPU 2 ONLY per this diagnostic's hard constraint):
  CUDA_VISIBLE_DEVICES is set internally to "2" (not read from
  environment) so this script cannot be accidentally pointed at a
  different GPU.
  KEYANCHOR_SCALING_PI_SIGNOFF=1 KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1 \\
      .venv/bin/python run_poolmargin_k84s1943_k90s2043.py

  Local, GPU-free negative-test/self-test mode (sec 15.26.3.1's own
  registered build-phase negative tests, run to completion, no launch):
  DRY_RUN_BYPASS=1 .venv/bin/python run_poolmargin_k84s1943_k90s2043.py --self-test
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_deltanet_rd_exactness_sweep as rdx  # noqa: E402

GPU = "2"  # HARD CONSTRAINT: GPU 2 ONLY. Not read from environment on purpose.
D_STATE = 96
K_A, SEED_A, REF_SEED_A = 84, 1943, 1944   # overlap-equalized cell + its sibling-seed reference
K_B, SEED_B, REF_SEED_B = 90, 2043, 2044   # natural-margin comparator cell + its sibling-seed reference
M3_POOL_RESTRICT_N = 100                    # sec 15.26.2.2 Rev 2 MAJOR-2 fix: 84/100=84.00% vs 90/107=84.11%

# sec 15.26.2.3's own seed table, cross-checked against the central registration this session
# (never hand-typed independently of it) -- catches any future drift in either constant table.
assert rdx.KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K[D_STATE][K_A] == (SEED_A, REF_SEED_A), \
    "K=84/d=96 contingency seed pair drifted from the central registration"
assert rdx.KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K[D_STATE][K_B] == (SEED_B, REF_SEED_B), \
    "K=90/d=96 contingency seed pair drifted from the central registration"

# sec 15.26.3: the killed 10-cell grid's own registered n_iter override, REUSED unmodified,
# additive-only, scope narrowed to the 2 K's this diagnostic actually launches. NEVER touches
# rdx.KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K (sec 15.26.0 item 4's own build-scope fence) --
# applied as a POST-CONSTRUCTION override on the spec _keyanchor_scaling_spec() returns instead.
POOLMARGIN_N_ITER_OVERRIDE = {96: {84: 28, 90: 28}}

# sec 15.26.4 item 1: REUSED verbatim from the killed grid's own derivation (SAME 0.427 GPU-h/
# cell REALIZED base rate, sec 15.22's own K=69/seed=1733 wall_s=1535.2s -- a DIFFERENT, tighter
# number than rdx.KEYANCHOR_SCALING_GPUH_PER_CELL_MAIN[96]=0.4313, the wider, NOT-yet-calibration-
# verified a-priori cost-model estimate that dict's own KEYANCHOR_SCALING_ABORT_WALL_S entry is
# built from -- this diagnostic's own design doc explicitly overrides with the tighter, realized-
# measurement-based figure rather than reusing that dict's own (different-purpose) entry).
POOLMARGIN_ABORT_WALL_S = 1.5 * 0.427 * 2 * 3600.0
assert abs(POOLMARGIN_ABORT_WALL_S - 4611.6) < 1e-6, \
    f"abort trigger arithmetic drifted from sec 15.26.4's own pinned 4611.6s: {POOLMARGIN_ABORT_WALL_S}"

# sec 15.26.3.1's Rev 2 MAJOR-3 fix: the enumerated whitelist of new flag tokens this wave's own
# cmd carries that build_cmd() itself does not know how to generate. Kept exhaustive by
# construction (see the reserved second-generator-flag entry's own comment, sec 15.26.3.1).
#
# build-audit MAJOR-3 fix (2026-07-08): keyed PER-K, not global. A single shared whitelist would
# let an ERRANT --m3-pool-restrict-n on the K=90 side (which must NEVER carry it -- K=90 is the
# unmodified natural-margin comparator, sec 15.26.2.2) be silently stripped and no-op through the
# field-diff check, exactly like a legitimate K=84 token -- masking a real bug instead of catching
# it. K=90's own entry is deliberately EMPTY: any extra token on that cell's cmd, including this
# flag, must still refuse. Negative test: `field-diff-negative[K90-errant-whitelisted-flag-...]`
# below injects the flag into K=90's cmd and confirms refusal.
NEW_FLAG_WHITELIST_BY_K = {
    84: {
        "--m3-pool-restrict-n": 1,   # takes exactly 1 positional value (the int)
        # second-generator flag: N/A this design -- the noise-floor repeat passes (sec 15.26.2.2,
        # Rev 2 MAJOR-1 fix + round-3 adopted second draw) are auto-gated by the SAME
        # `m3_pool_restrict_n is not None` condition inside train(), not their own separate CLI
        # token; this entry is reserved in case a future revision splits one out, so the whitelist
        # stays exhaustive by construction rather than by omission.
    },
    90: {},   # K=90 carries NO new flag ever -- empty by design (build-audit MAJOR-3 fix).
}

# build-audit MAJOR-1 fix (2026-07-08): OUT_DIR is DELIBERATELY its own
# directory, NOT the shared wavekeyanchor-scaling-wide/ dir the rest of this
# wave's cells (K in {24,51,57,63,69,72,78,84,90} at n_iter=20) live in.
# `fit_cliff_curve.load_k_mean_h4(cell_dir, K, arm="d")` globs
# `cell_dir/*_K{K}_armd_*.json` with NO n_iter/seed filter -- if this
# diagnostic's own n_iter=28 K=84/seed=1943 and K=90/seed=2043 cells landed
# in the SAME directory as the FROZEN sec 15.26.1 per-K table's own n_iter=20
# cells (K=84/90 are both members of KEYANCHOR_SCALING_D96_WIDE_KS=(72,78,
# 84,90)), a future regeneration of that table via the SAME loader would
# silently pull these mismatched-n_iter cells into the "mean h4"/"sample sd"
# columns, contaminating a table this wave explicitly registers as frozen
# (sec 15.26.1). Routing to a NEW, disjoint directory closes this by
# construction -- mirrors the LOCAL_D96_ARCHIVE_DIR isolation precedent
# (smoke_keyanchor_scaling_wide.py) of never letting a test/diagnostic's own
# reads or writes share a path with the production archive they must not
# perturb. poolmargin_stage_minus1.py's own new Item 5 asserts this
# disjointness by construction (path-prefix check), not just by comment.
FROZEN_WIDE_DIR = os.path.join(rdx.DEFAULT_OUT_DIR, "wavekeyanchor-scaling-wide")
OUT_DIR = os.path.join(rdx.DEFAULT_OUT_DIR, "wavekeyanchor-scaling-poolmargin")
assert OUT_DIR != FROZEN_WIDE_DIR and not OUT_DIR.startswith(FROZEN_WIDE_DIR + os.sep) and \
       not FROZEN_WIDE_DIR.startswith(OUT_DIR + os.sep), (
    "build-audit MAJOR-1: OUT_DIR must be disjoint from the frozen wavekeyanchor-scaling-wide/ "
    f"dir -- got OUT_DIR={OUT_DIR!r}, FROZEN_WIDE_DIR={FROZEN_WIDE_DIR!r}")
CKPT_BASE_DIR = "/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-scaling-poolmargin"
LOG_DIR = os.path.join(HERE, "logs")
SENTINEL_DIR = OUT_DIR   # sentinels live alongside this wave's own result JSONs


def refuse(msg: str) -> None:
    print(f"REFUSED: {msg}", file=sys.stderr)
    sys.exit(1)


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")


# ---------------------------------------------------------------------------
# Gate (d): PI signoff (two independent tokens).
# ---------------------------------------------------------------------------

def check_pi_signoff() -> None:
    if os.environ.get("KEYANCHOR_SCALING_PI_SIGNOFF", "0") != "1":
        refuse("KEYANCHOR_SCALING_PI_SIGNOFF=1 not set (gate d, primary).")
    if os.environ.get("KEYANCHOR_SCALING_EXT_PI_SIGNOFF", "0") != "1":
        refuse("KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1 not set (gate d, extension -- required as a "
                "conservative safety net even though the 1x ledger does not need it drawn on, "
                "sec 15.26.3.1/15.26.4).")


# ---------------------------------------------------------------------------
# Gate (a1): kernel-safety, reused, no new probe.
# ---------------------------------------------------------------------------

def check_kernel_safety() -> None:
    gate = rdx.keyanchor_scaling_wide_kernel_gate_check()
    if not gate["ok"]:
        refuse(f"gate (a1) wide kernel-safety FAILED: {gate['reason']}")
    print(f"gate (a1) wide kernel-safety (T in {{588,630}}): OK -- {gate['reason']}", flush=True)


# ---------------------------------------------------------------------------
# Gate (g): seed-space collision check, mechanical, re-run this session.
# ---------------------------------------------------------------------------

def check_seed_collision() -> None:
    pattern = f"_s{SEED_A}_|_s{SEED_B}_"
    try:
        out = subprocess.run(
            ["grep", "-rn", "-E", pattern,
             os.path.join(HERE), os.path.join(os.path.dirname(HERE), "..", "experiment-runs")],
            capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        refuse("gate (g) seed-collision check could not run: `grep` not found on PATH.")
        return
    hits = [ln for ln in out.stdout.splitlines() if ln.strip()]
    # Registration-table / fallback-note hits are EXPECTED (the seeds are correctly registered
    # constants); the check refuses only on a hit inside an ARCHIVED RESULT JSON (i.e. a path
    # containing '.json' under experiment-runs/, meaning the seed already fired for something
    # else) -- mirrors sec 15.26.2.3's own disclosed collision-check discipline.
    bad = [ln for ln in hits if ".json" in ln and "experiment-runs" in ln]
    if bad:
        refuse(f"gate (g) seed-collision check FOUND a prior archived result JSON already using "
                f"seed {SEED_A} or {SEED_B}: {bad[:5]}")
    print(f"gate (g) seed-collision check: OK -- {len(hits)} registration-table/fallback-note "
          f"hits, ZERO archived-result-JSON hits", flush=True)


# ---------------------------------------------------------------------------
# Spec construction: _keyanchor_scaling_spec + the n_iter override, additive-only.
# ---------------------------------------------------------------------------

def _pool_margin_spec(K: int, seed: int) -> dict:
    """Builds ONE cell's spec via rdx._keyanchor_scaling_spec (never hand-typed CLI flags),
    then applies sec 15.26.3's registered geo3_n_iter 20->28 bump as a POST-CONSTRUCTION,
    additive override -- rdx.KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K (the dict that function
    reads n_iter FROM) is NEVER modified. Also fixes the spec's own `name` bit
    (`geo3n{n_iter}`, baked in at construction time from the dict-read n_iter) to match the
    overridden value, so out_path()/is_done()/the ckpt-dir all consistently reflect n_iter=28
    rather than a misleading geo3n20 filename for a cell that actually trains at n_iter=28.

    build-audit MINOR-3 fix -- framing, not mechanism: this rename is a NON-MISLEADING-FILENAME
    fix only, never a collision-avoidance one. `_spec()`'s own bit order (rdx.py) already puts
    `s{seed}` BEFORE the `geo3n{n_iter}` bit, and seed=1943/2043 are unique to this diagnostic's
    2 cells (sec 15.26.2.3's own collision-check, gate (g) above) -- every filename in this
    directory is therefore already distinct by seed alone, with or without this rename. Renaming
    `geo3n20`->`geo3n28` changes nothing about whether a collision could occur; it exists solely
    so a human/tool reading the filename sees the n_iter this cell actually trained at."""
    spec = rdx._keyanchor_scaling_spec(K, seed, D_STATE)
    n_iter = POOLMARGIN_N_ITER_OVERRIDE[D_STATE][K]
    old_bit, new_bit = f"geo3n{spec['geo3_n_iter']}", f"geo3n{n_iter}"
    assert old_bit in spec["name"], f"expected name bit {old_bit!r} not found in {spec['name']!r}"
    spec["geo3_n_iter"] = n_iter
    spec["name"] = spec["name"].replace(old_bit, new_bit)
    return spec


def build_cell_cmd(K: int, seed: int, m3_pool_restrict_n: int | None) -> tuple[dict, list]:
    """Builds ONE cell's spec + cmd via _pool_margin_spec/build_cmd (never hand-typed), then
    appends --m3-pool-restrict-n ONLY when m3_pool_restrict_n is given -- build_cmd() itself
    does not know about this diagnostic's own wave-specific flag (sec 15.26.2.2)."""
    spec = _pool_margin_spec(K, seed)
    timeout = rdx.default_timeout(spec["K"], spec["steps"])
    cmd = rdx.build_cmd(spec, OUT_DIR, timeout, unblind_override_at=None, ckpt_base_dir=CKPT_BASE_DIR)
    if m3_pool_restrict_n is not None:
        cmd = cmd + ["--m3-pool-restrict-n", str(m3_pool_restrict_n)]
    return spec, cmd


# ---------------------------------------------------------------------------
# Gate (j): whitelist-adapted field-diff (Rev 2 MAJOR-3 fix).
# ---------------------------------------------------------------------------

def strip_whitelisted(cmd_tokens: list, K: int) -> list:
    """build-audit MAJOR-3 fix: looks up K's OWN whitelist (NEW_FLAG_WHITELIST_BY_K[K]) rather
    than a single global one -- K=90's own entry is empty, so an errant whitelisted-elsewhere flag
    on K=90's cmd is never silently stripped.

    build-audit MAJOR-2 fix: a duplicated whitelisted flag has no registered meaning (the launch
    mechanism never emits one twice) -- silently stripping ALL occurrences would mask that
    corruption instead of surfacing it. Assert at most one occurrence of each whitelisted flag
    BEFORE stripping; refuse loudly (AssertionError) on a duplicate rather than stripping blindly."""
    whitelist = NEW_FLAG_WHITELIST_BY_K[K]
    for flag in whitelist:
        count = cmd_tokens.count(flag)
        assert count <= 1, (
            f"build-audit MAJOR-2: whitelisted flag {flag!r} appears {count} times in K={K}'s own "
            f"generated command -- refusing to strip a duplicated flag silently (cmd={cmd_tokens!r})")
    out, i = [], 0
    while i < len(cmd_tokens):
        tok = cmd_tokens[i]
        if tok in whitelist:
            i += 1 + whitelist[tok]   # skip the flag + its N values
            continue
        out.append(tok)
        i += 1
    return out


def normalize(cmd: list, seed: int) -> list:
    return [tok.replace(str(seed), "SEED") for tok in cmd]


def check_field_diff(K: int, seed: int, ref_seed: int, cmd: list, m3_pool_restrict_n: int | None) -> None:
    """sec 15.26.3.1's Rev 2 MAJOR-3 fix: strips K's own whitelist (NEW_FLAG_WHITELIST_BY_K[K],
    build-audit MAJOR-3 fix: per-K, not global) from `cmd` BEFORE the precedent's own equality-diff
    runs against a sibling-seed reference command (built the SAME way, at ref_seed, WITHOUT the new
    flag -- an in-process, never-launched reference, sec 15.26.2.3's own disclosed cross-reference),
    so the check keeps its teeth against any OTHER kind of drift."""
    _, ref_cmd = build_cell_cmd(K, ref_seed, m3_pool_restrict_n=None)
    cmd_for_diff = strip_whitelisted(cmd, K)
    n_new = normalize(cmd_for_diff, seed)
    n_ref = normalize(ref_cmd, ref_seed)
    if n_new != n_ref:
        print(f"FIELD-DIFF MISMATCH (K={K}, seed={seed} vs ref_seed={ref_seed}):", file=sys.stderr)
        for a, b in zip(n_new, n_ref):
            if a != b:
                print(f"  new={a!r} ref={b!r}", file=sys.stderr)
        if len(n_new) != len(n_ref):
            print(f"  length mismatch: new={len(n_new)} ref={len(n_ref)}", file=sys.stderr)
        refuse(f"K={K}/seed={seed}: generated command diverges from the sibling-seed reference "
                f"command (seed={ref_seed}) in a field OTHER than seed or a whitelisted new flag "
                f"-- refusing to launch. Do not override without human review.")
    print(f"field-diff vs. sibling seed={ref_seed} (whitelist-stripped, same builder, seed field "
          f"excepted): MATCH (K={K}, seed={seed})", flush=True)


# ---------------------------------------------------------------------------
# Gate (f): post-hoc admission check, disclosed (not hard-aborting) on failure.
# ---------------------------------------------------------------------------

def check_admission(result: dict, label: str) -> bool:
    """sec 15.26.3 gate item 2 (REUSED, scope narrowed to 2 cells): mechanical post-hoc
    `geo3_admission.admissible is True` check. Returns True/False, never raises -- an
    inadmissible cell is a disclosed, escalatable FINDING (sec 15.26.5's own degenerate-cell
    trigger row: routed to the SAME sec 15.23 C17-exclusive-signature adjudication Rev 0
    already registered), not silently treated as a crash."""
    ga = result.get("geo3_admission") or {}
    admissible = ga.get("admissible")
    ok = admissible is True
    status = "PASS" if ok else "FAIL"
    print(f"[admission-check:{label}] {status} -- admissible={admissible!r} "
          f"checkpoint_fallback_seen={ga.get('checkpoint_fallback_seen')!r} "
          f"ns_converged_no_fallback={ga.get('ns_converged_no_fallback')!r}", flush=True)
    if not ok:
        print(f"[admission-check:{label}] DISCLOSED FINDING: geo3_admission.admissible is not "
              f"True at n_iter=28 -- per sec 15.26.3 item 2, escalate against sec 15.23's own "
              f"C17-exclusive-signature adjudication before trusting this cell's h4 for the "
              f"harvest.", file=sys.stderr, flush=True)
    return ok


# ---------------------------------------------------------------------------
# One cell: build, diff-check, launch (or idempotent no-op resume).
# ---------------------------------------------------------------------------

def launch_cell(K: int, seed: int, ref_seed: int, m3_pool_restrict_n: int | None, label: str) -> dict:
    spec, cmd = build_cell_cmd(K, seed, m3_pool_restrict_n)
    print(f"[{label}] spec: {spec}", flush=True)
    check_field_diff(K, seed, ref_seed, cmd, m3_pool_restrict_n)

    if rdx.is_done(OUT_DIR, spec):
        print(f"[{label}] ALREADY DONE: {rdx.out_path(OUT_DIR, spec)!r} validity-checks as "
              f"complete -- nothing to launch (idempotent no-op).", flush=True)
        with open(rdx.out_path(OUT_DIR, spec)) as f:
            return json.load(f)

    timeout = rdx.default_timeout(spec["K"], spec["steps"])
    print(f"sha256(cmd)={hashlib.sha256(' '.join(cmd).encode()).hexdigest()}", flush=True)
    print(f"[{label}] CMD: {' '.join(cmd)}", flush=True)
    print(f"[{label}] timeout={timeout}s (~{timeout/60:.1f} min); GPU={GPU} "
          f"(CUDA_VISIBLE_DEVICES, hard-pinned)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{spec['name']}.log")

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU}
    start = time.time()
    print(f"[{label}] LAUNCHING at {time.strftime('%Y-%m-%d %H:%M:%S')} -- log: {log_path}",
          flush=True)
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
    wall_s_wrapper = time.time() - start
    print(f"[{label}] FINISHED rc={rc} wrapper_wall_s={wall_s_wrapper:.1f} "
          f"({time.strftime('%Y-%m-%d %H:%M:%S')})", flush=True)

    ok = rc == 0 and rdx.is_done(OUT_DIR, spec)
    if not ok:
        print(f"[{label}] FAILED: rc={rc} is_done={rdx.is_done(OUT_DIR, spec)} -- see {log_path}",
              file=sys.stderr)
        _touch(os.path.join(SENTINEL_DIR, f"POOLMARGIN_{label}_FAILED"))
        raise RuntimeError(f"[{label}] cell launch failed, see {log_path}")

    with open(rdx.out_path(OUT_DIR, spec)) as f:
        result = json.load(f)
    _touch(os.path.join(SENTINEL_DIR, f"POOLMARGIN_{label}_DONE"))
    return result


# ---------------------------------------------------------------------------
# Self-test mode: sec 15.26.3.1's own registered negative tests, run to
# completion, no GPU/CUDA required.
# ---------------------------------------------------------------------------

def _self_test() -> int:
    failures = []

    def check(name, cond):
        status = "PASS" if cond else "FAIL"
        print(f"[self-test:{name}] {status}", flush=True)
        if not cond:
            failures.append(name)

    # --- gate (i): PI-signoff negative test, all 3 missing-token combinations ---
    py = sys.executable
    script = os.path.join(HERE, __file__ if os.path.isabs(__file__) else os.path.join(HERE, __file__))
    script = os.path.abspath(__file__)
    base_env = {k: v for k, v in os.environ.items()
                if k not in ("KEYANCHOR_SCALING_PI_SIGNOFF", "KEYANCHOR_SCALING_EXT_PI_SIGNOFF")}
    combos = [
        ("neither", {}),
        ("only_primary", {"KEYANCHOR_SCALING_PI_SIGNOFF": "1"}),
        ("only_extension", {"KEYANCHOR_SCALING_EXT_PI_SIGNOFF": "1"}),
    ]
    for combo_name, extra in combos:
        env = {**base_env, **extra, "DRY_RUN_BYPASS": "1"}
        proc = subprocess.run([py, script, "--gate-check-only"], env=env,
                               capture_output=True, text=True, timeout=60)
        refused = proc.returncode == 1 and "REFUSED" in proc.stderr
        check(f"pi-signoff-refusal[{combo_name}]", refused)

    # positive control: both tokens set must NOT refuse at the gate-check-only stage
    env = {**base_env, "KEYANCHOR_SCALING_PI_SIGNOFF": "1", "KEYANCHOR_SCALING_EXT_PI_SIGNOFF": "1",
           "DRY_RUN_BYPASS": "1"}
    proc = subprocess.run([py, script, "--gate-check-only"], env=env,
                           capture_output=True, text=True, timeout=60)
    check("pi-signoff-positive-control[both-set]", "REFUSED" not in proc.stderr)

    # --- gate (j): whitelist field-diff, positive + negative + no-op cases ---
    spec_a, cmd_a = build_cell_cmd(K_A, SEED_A, M3_POOL_RESTRICT_N)
    check("whitelist-strip-removes-exactly-the-new-flag",
          cmd_a.count("--m3-pool-restrict-n") == 1 and
          strip_whitelisted(cmd_a, K_A).count("--m3-pool-restrict-n") == 0 and
          len(strip_whitelisted(cmd_a, K_A)) == len(cmd_a) - 2)
    try:
        check_field_diff(K_A, SEED_A, REF_SEED_A, cmd_a, M3_POOL_RESTRICT_N)
        check("field-diff-positive[K84-whitelisted]", True)
    except SystemExit:
        check("field-diff-positive[K84-whitelisted]", False)

    spec_b, cmd_b = build_cell_cmd(K_B, SEED_B, None)
    check("whitelist-strip-is-noop-for-K90", strip_whitelisted(cmd_b, K_B) == cmd_b)
    try:
        check_field_diff(K_B, SEED_B, REF_SEED_B, cmd_b, None)
        check("field-diff-positive[K90-no-new-flag]", True)
    except SystemExit:
        check("field-diff-positive[K90-no-new-flag]", False)

    # negative test: one extra, non-whitelisted token must still refuse
    bogus_cmd = cmd_a + ["--bogus-extra-flag", "1"]
    negative_refused = False
    try:
        check_field_diff(K_A, SEED_A, REF_SEED_A, bogus_cmd, M3_POOL_RESTRICT_N)
    except SystemExit as e:
        negative_refused = (e.code == 1)
    check("field-diff-negative[one-extra-non-whitelisted-token]", negative_refused)

    # build-audit MAJOR-2 negative test: a DUPLICATED whitelisted flag must refuse (loudly,
    # AssertionError), never be silently stripped in full.
    dup_cmd = cmd_a + ["--m3-pool-restrict-n", "100"]   # cmd_a already carries it once
    assert dup_cmd.count("--m3-pool-restrict-n") == 2, "test fixture itself must carry 2 copies"
    dup_refused = False
    try:
        strip_whitelisted(dup_cmd, K_A)
    except AssertionError:
        dup_refused = True
    check("whitelist-strip-refuses-duplicated-flag[MAJOR-2]", dup_refused)

    # build-audit MAJOR-3 negative test: K=90's own whitelist is EMPTY -- an errant
    # --m3-pool-restrict-n token on K=90's cmd must NOT silently no-op through K=84's own
    # whitelist entry (a global whitelist would mask exactly this bug).
    bogus_cmd_b = cmd_b + ["--m3-pool-restrict-n", "100"]
    negative_refused_b = False
    try:
        check_field_diff(K_B, SEED_B, REF_SEED_B, bogus_cmd_b, None)
    except SystemExit as e:
        negative_refused_b = (e.code == 1)
    check("field-diff-negative[K90-errant-whitelisted-elsewhere-flag][MAJOR-3]", negative_refused_b)
    check("whitelist-by-k-K90-entry-is-empty[MAJOR-3]", NEW_FLAG_WHITELIST_BY_K[K_B] == {})

    # --- n_iter override determinism ---
    check("n-iter-override-K84", spec_a["geo3_n_iter"] == 28 and "geo3n28" in spec_a["name"])
    check("n-iter-override-K90", spec_b["geo3_n_iter"] == 28 and "geo3n28" in spec_b["name"])
    check("n-iter-override-does-not-touch-central-dict",
          rdx.KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96][84] == 20 and
          rdx.KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96][90] == 20)

    # --- gate (a1): kernel-safety check runs cleanly (structural, not asserting a real PASS
    # locally -- the artifact lives at a box-only default path, sec 15.26.6's own disclosed
    # environment split) ---
    gate = rdx.keyanchor_scaling_wide_kernel_gate_check()
    check("kernel-gate-check-returns-well-formed-dict",
          isinstance(gate, dict) and "ok" in gate and "reason" in gate)

    # --- gate (g): seed-collision check runs cleanly and finds no archived-JSON hits ---
    try:
        check_seed_collision()
        check("seed-collision-check-runs-clean", True)
    except SystemExit:
        check("seed-collision-check-runs-clean", False)

    # --- gate (f): admission-check negative fixture ---
    fake_bad = {"geo3_admission": {"admissible": False, "checkpoint_fallback_seen": True,
                                    "ns_converged_no_fallback": False}}
    fake_good = {"geo3_admission": {"admissible": True, "checkpoint_fallback_seen": False,
                                     "ns_converged_no_fallback": True}}
    check("admission-check-negative-fixture", check_admission(fake_bad, "synthetic-bad") is False)
    check("admission-check-positive-fixture", check_admission(fake_good, "synthetic-good") is True)

    # --- abort-trigger arithmetic ---
    check("abort-trigger-pinned-value", abs(POOLMARGIN_ABORT_WALL_S - 4611.6) < 1e-6)

    print(f"\nSELF-TEST SUMMARY: {len(failures)} failure(s) out of self-test items run.", flush=True)
    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    print("ALL SELF-TESTS PASSED.", flush=True)
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()

    check_pi_signoff()
    print("gate (d) PI signoff (primary + extension): OK", flush=True)
    if "--gate-check-only" in sys.argv:
        # sec 15.26.3.1's own negative-test harness invokes this mode to exercise ONLY the
        # gates above without building specs/launching anything -- keeps the self-test's own
        # subprocess calls fast and CUDA-free.
        print("--gate-check-only: gates passed, stopping before spec construction/launch "
              "(this mode never launches).", flush=True)
        return 0

    check_kernel_safety()
    check_seed_collision()

    os.makedirs(SENTINEL_DIR, exist_ok=True)

    print("=" * 70)
    print("STAGE 0: K=84/seed=1943 (overlap-equalized, calibration-first)")
    print("=" * 70, flush=True)
    result_a = launch_cell(K_A, SEED_A, REF_SEED_A, M3_POOL_RESTRICT_N, "K84_s1943")
    wall_s_a = result_a.get("wall_s")
    print(f"[K84_s1943] wall_s={wall_s_a} (sec 15.26.4 item 1 abort trigger="
          f"{POOLMARGIN_ABORT_WALL_S:.1f}s; "
          f"{'OVER TRIGGER -- HALTING, do NOT launch K=90' if wall_s_a and wall_s_a >= POOLMARGIN_ABORT_WALL_S else 'within bracket'})",
          flush=True)
    check_admission(result_a, "K84/seed=1943")
    if wall_s_a is not None and wall_s_a >= POOLMARGIN_ABORT_WALL_S:
        _touch(os.path.join(SENTINEL_DIR, "POOLMARGIN_ABORT_K84_OVER_TRIGGER"))
        refuse(f"K=84/seed=1943 realized wall_s={wall_s_a:.1f}s >= abort trigger "
                f"{POOLMARGIN_ABORT_WALL_S:.1f}s (sec 15.26.4 item 1). Halting before K=90 -- "
                f"diagnose (nvidia-smi contention first), re-price before resuming.")

    print("=" * 70)
    print("STAGE 1: K=90/seed=2043 (natural-margin comparator)")
    print("=" * 70, flush=True)
    result_b = launch_cell(K_B, SEED_B, REF_SEED_B, None, "K90_s2043")
    check_admission(result_b, "K90/seed=2043")

    _touch(os.path.join(SENTINEL_DIR, "POOLMARGIN_CHAIN_DONE"))
    print("\nCHAIN COMPLETE. Both cells landed -- run harvest_poolmargin_k84s1943_k90s2043.py "
          "next.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
