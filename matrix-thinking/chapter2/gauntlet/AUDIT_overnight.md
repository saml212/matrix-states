# AUDIT — run_overnight.py (unattended 8-GPU, 421-run overnight orchestrator)

Scope: orchestration wrapper ONLY. `run_task_d.py` internals are out of scope
(already 3x-audited); read only to confirm the CLI flag / output-JSON contract
the orchestrator depends on. Static review — cannot execute (no torch/GPU
here). Manifest counts below were verified by actually running
`python3 run_overnight.py --dry-run` and a small programmatic check of
`build_manifest()` (no torch import required for these paths).

Verified: 421 total runs, tiers {1: 195, 2: 184, 3: 42}, all 421 `spec["name"]`
values are unique (zero collisions), no spec has `force_rank_k > d`, no spec
has `orthogonal=True` with `K > d`.

---

## FATAL

**F1. Launch loop has no exception isolation — a single transient OS error kills the entire overnight run.**
`run_overnight.py:181-188` (and `:212`, and the module has no top-level
`try/except` around `main()`, `:264-265`).

```python
lf = open(os.path.join(log_dir, f"{spec['name']}.log"), "w")      # can raise (ENOSPC, EMFILE, permissions)
env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
proc = subprocess.Popen(build_cmd(spec, out_dir), env=env, ...)   # can raise OSError (EAGAIN from fork()
                                                                    #   under load, ENOENT, etc.)
running[gpu] = (proc, spec, lf, time.time())
```

None of this is wrapped in `try/except`. Any `OSError`/`FileNotFoundError`
(fork() returning EAGAIN while launching dozens of subprocesses across 8 GPUs
over hundreds of runs, a transient NFS/disk hiccup on the log directory,
`ENOSPC`, a permissions blip) propagates straight out of `main()` and kills
the whole Python process. At that point all `pending` work and all currently
`running` work is abandoned — the process is gone, nothing gets freed,
nothing gets marked failed, no sentinel file is written. This directly
contradicts the script's own stated design goal (docstring, `:12-14`):
*"a crash/OOM cannot corrupt sibling runs or the orchestrator."* It's true for
the experiment subprocess crashing; it is **not** true for the orchestrator's
own launch calls. This is exactly the failure class that lost 24/27 runs in
`rank_aware_v1` — a single unhandled exception ending the whole night early,
discovered only the next morning.

`aggregate(out_dir, manifest, failed)` (`:212`) is likewise called with no
guard — if it raises (see F-adjacent MAJOR M5 below), the run ends without
`SUMMARY.txt`/`AGGREGATE.json` or the final `DONE.` message, even if all 421
runs individually succeeded.

**Fix:** wrap the per-launch body (`open` + `Popen`) in `try/except Exception`;
on failure, write the exception to a `launch-errors.log`, record the spec as
failed (`failed.append((spec["name"], f"LAUNCH_ERROR: {e}"))`), return the
`gpu` to `free` (or drop it and log — never lose track of it silently), and
`continue` the loop rather than propagate. Additionally wrap `main()`'s core
loop (and the `aggregate()` call) in a top-level `try/except` that writes a
`CRASHED.txt` with the traceback before exiting, so a still-uncaught failure
is instantly diagnosable in the morning instead of requiring log archaeology.

---

## MAJOR

**M2. Timeout-killed processes are never verified dead before their GPU is reassigned.**
`run_overnight.py:192-199`.
```python
if time.time() - start > args.run_timeout:
    proc.kill()
    lf.close()
    del running[gpu]
    free.append(gpu)
    failed.append((spec["name"], "TIMEOUT"))
```
`proc.kill()` sends SIGKILL but the code never calls `proc.wait()` to confirm
the process actually exited before returning `gpu` to the free pool and
letting the outer loop immediately launch a new subprocess on that same
physical GPU via `CUDA_VISIBLE_DEVICES`. SIGKILL is not deliverable to a
process stuck in an uninterruptible kernel wait (a driver hang / Xid error —
a real H100 failure mode, and precisely the kind of hardware-level flakiness
an 8-hour unattended run is exposed to). If that happens, the GPU is
optimistically marked free while the old process may still hold VRAM/CUDA
context, and the next run scheduled onto that slot can OOM or silently
misbehave — corrupting a run rather than cleanly failing it. There is also no
process-group kill (`start_new_session=True` + `os.killpg`) as defense in
depth; not exploitable today since `run_task_d.py` spawns no child processes,
but fragile if that ever changes (e.g., a future `DataLoader(num_workers>0)`).

**Fix:** after `proc.kill()`, do
`try: proc.wait(timeout=30); except subprocess.TimeoutExpired:` log a
CRITICAL warning and **do not** return that GPU to `free` for the remainder of
the run (quarantine it) rather than assuming success.

**M3. A single global `--run-timeout` (3600s default) is applied uniformly across tiers whose cost varies ~2x+ in steps alone.**
`run_overnight.py:47` (`STEPS = {8: 8000, 16: 8000, 32: 12000, 64: 12000, 128: 16000}`)
vs. `:143-145` (one `--run-timeout` for the whole manifest). Tier 3 (`d=128`,
16000 steps, 42 runs — confirmed via dry-run) is both 2x the step count of
Tier 1 *and* has a strictly more expensive per-step matrix op than `d=16`. If
real per-step wall-clock at `d=128` exceeds ~2.25x that of `d=16` (very
plausible), **every one of the 42 Tier-3 runs will be killed as a false
timeout**, silently burning 8-GPU-hours for zero output, with no signal
besides a wall of `TIMEOUT` entries in `PROGRESS.txt`/`SUMMARY.txt` that looks
identical to a real hang.

**Fix:** scale `--run-timeout` per spec (e.g. proportional to `steps`, or a
per-tier override), or at minimum time one real `d=128`/16000-step run before
the overnight launch and set a global timeout with generous headroom above it.

**M4. Resume/aggregate trust file *existence*, not file *validity* — a mid-write kill can silently and permanently lose a run.**
`run_overnight.py:172-173` (resume split) and `:224-227` (aggregate load).
`run_task_d.py` writes `--out` only once, at the very end
(`run_task_d.py:283-287`, `with open(args.out, "w") as f: json.dump(...)`),
which correctly protects against the orchestrator's own `--run-timeout` kill
(no partial write is possible there, since `run_task_d.py` hasn't reached that
line yet when killed for exceeding the *orchestrator's* timeout). But an
external kill of the **worker** — node preemption, `OOM-killer` targeting the
training subprocess, disk full mid-write — can land exactly inside the
`json.dump` call, leaving a file that **exists** but is truncated/invalid.
`aggregate()`'s loader silently drops it (`except Exception: pass`, `:226-227`)
— the run vanishes from both `done` and the aggregate with no warning — and on
a **restart** of `run_overnight.py`, the resume check (`os.path.exists`,
`:172`) treats the corrupt file as "already done" and **never retries it**.
This is exactly the preemption scenario the whole resumable design exists to
survive, and it fails silently rather than loudly.

**Fix:** when building `pending`/`done`, don't just check existence — attempt
a parse and check for an expected key (e.g. `"K" in data`); treat
invalid/unparseable files as **not done** (optionally rename them to
`*.json.corrupt` and log), so they get retried on the next run.

**M5. `aggregate()` can crash on a malformed-but-parseable record, discarding the whole night's summary even though the underlying data is safe.**
`run_overnight.py:236-237, 244-245` index `r["d"]` / `r["K"]` directly (no
`.get()`, unlike `force_rank_k` which is defensively `.get()`'d at `:236,
244`), and `aggregate()` itself is called unguarded at `:212`. Under the
current `run_task_d.py` contract every JSON always has `"d"`/`"K"`
(`run_task_d.py:98`), so this is unlikely to trigger today — but if it ever
does (schema drift, a hand-inspected file re-saved without those keys, a
`run_task_d.py` change), the `KeyError` happens at the very end of a 6-8 hour
run, before `SUMMARY.txt`/`AGGREGATE.json` are written and before the final
`"DONE. X/Y succeeded"` message prints. No data is lost (the 421 individual
run JSONs are untouched), but the researcher wakes up to an unhandled
traceback and no summary of an otherwise-successful night.

**Fix:** wrap the per-record aggregation in `try/except` (skip + count
malformed records rather than crash), and wrap the `aggregate()` call itself
in `try/except` so a summary-generation failure can never mask a successful
run.

**M6. Tier-1 manifest ordering starves the PRIMARY causal test (M3) if the night is cut short mid-tier.**
Verified programmatically: Tier 1's 195 specs are strictly FIFO-ordered with
all 80 M1 (unconstrained K-sweep) runs at indices 0-79, followed by all 115
M3 (`force_rank_k`) runs at indices 80-194 (`build_manifest()`,
`run_overnight.py:66-72`). Dispatch pops `pending[0]` (`:183`) with only 8
concurrent GPU slots, so the first M3 run cannot launch until nearly all 80
M1 runs have completed. `TASK_D_PREREGISTRATION.md` §6 explicitly labels M3
*"PRIMARY causal test"* and the overall gate requires *"M1 CONFIRM and M3
CONFIRM"* (§6, "Overall gate"). If the run is interrupted partway through
Tier 1 — a crash, a preemption, hitting the 6-8h wall-clock budget before
Tier 1 finishes — the likely outcome is "M1 fully done, M3 barely started,"
which is the *opposite* of a "coherent, useful prefix" for the decisive
question (the script's own stated goal, docstring `:19-20`): you'd have
correlational evidence but none of the causal evidence the gate is actually
decided on.

**Fix:** interleave M1 and M3 tier-1 specs (round-robin merge, or an explicit
sub-priority key) so a partial Tier-1 completion has some coverage of both.

---

## MINOR

**m7. Smoke gate has no timeout.** `run_overnight.py:112-122`,
`subprocess.call([...])` with no `timeout=`. If GPU 0 (or the CUDA driver) is
already wedged before the run starts, the smoke gate hangs forever — the
*entire* unattended window is silently consumed with zero progress and no
`ABORTED.txt`, since the code never returns from `subprocess.call`. Fix: pass
a timeout, catch `subprocess.TimeoutExpired`, treat as smoke failure.

**m8. No validation of `--gpus`.** `run_overnight.py:139, 178, 180-210`. If
invoked with `--gpus 0` (bad scheduler value, typo), `free` starts empty, and
`while pending or running:` spins forever polling every `args.poll` seconds,
launching nothing, writing no `ABORTED.txt`, no diagnostic — a silent,
resource-cheap, totally unproductive loop for the whole night. Fix:
`assert args.gpus >= 1` at startup.

**m9. Unclosed file handle in `aggregate()`.** `run_overnight.py:225`,
`json.load(open(p))` relies on CPython refcounting to close the handle rather
than a `with` block. Not a practical fd-exhaustion risk at 421 files, but not
portable/robust. Fix: `with open(p) as f: runs.append(json.load(f))`.

**m10. `recovered_frac@0.99` key match is an untyped cross-file string
contract.** `run_overnight.py:228` hardcodes `"recovered_frac@0.99"`;
`run_task_d.py:62-63` builds the same string via
`f"{prefix}recovered_frac@{tau}"` with `tau` a Python float from
`TAUS = (0.9, 0.95, 0.99)`. Currently correct — `str(0.99)` formats to
`"0.99"` — and confirmed the `d`/`K`/`force_rank_k`/`effective_rank_mean`
keys `aggregate()` reads (`:236-237, 244-245`) match exactly what
`evaluate()` writes (`run_task_d.py:98-111`). But there's no shared constant
and no test tying the two files together; if `TAUS` is ever edited, this
fails *soft* (empty M3 summary, guarded by `if tau in r`, `:244`) rather than
loudly. Nice-to-have: export the tau-key format from `run_task_d.py` and
import it, or leave a comment cross-referencing the two files.

**m11. No daemonization / disconnect protection.** The script has no built-in
`setsid`/nohup-equivalent; unattended resilience against an SSH/terminal
disconnect depends entirely on the caller wrapping it in `nohup`/`tmux`/
`screen`. Not a code bug — a deployment prerequisite worth stating explicitly
given the "walk away for 6-8 hours" framing.

**m12. Tier 2 (4 seeds) and Tier 3 (3 seeds) fall short of the pre-registered
C3 "≥5 seeds" mandatory control.** `run_overnight.py:45-46`
(`SEEDS_T2 = (0,1,2,3)`, `SEEDS_T3 = (0,1,2)`) vs.
`TASK_D_PREREGISTRATION.md` §5 C3: *"≥5 seeds ... rank_aware_v1 saw 3x rank
spread across seeds at matched accuracy — rank is noisy."* Tier 1 (the
decisive gate) correctly uses 8 seeds (M1) and 5 seeds (M3), matching or
exceeding C3. Tiers 2/3 are lower-confidence "generality"/"scale stretch"
extensions, so under-shooting C3 there is plausibly an intentional compute
trade-off — but C3 is stated as a blanket control with no tier scoping, so
flagging it for an explicit sign-off rather than a silent deviation.

---

## Verified correct (no finding)

- **Smoke gate genuinely gates.** `run_smoke()` runs and is checked
  (`:166-169`) strictly before `pending`/`running` are even constructed
  (`:172-188`); on failure, `ABORTED.txt` is written and `sys.exit(1)` — no
  training subprocess is reachable if smoke `rc != 0`.
- **GPU bookkeeping has no double-assignment or leak** in the normal-launch
  and normal-completion paths: `running` is keyed by unique GPU index,
  `free`/`running` are complementary, and the completion loop iterates a
  `list(running.items())` snapshot so in-loop mutation is safe.
  `CUDA_VISIBLE_DEVICES` is set correctly per subprocess from the popped GPU
  index (only weakness is the unverified-kill race, M2 above).
- **Poll loop termination is correct**: `while pending or running:` exits iff
  both are empty; no other exit path; `time.sleep(args.poll)` prevents a tight
  busy-wait.
- **Resume correctness holds for the common case.** `run_task_d.py` writes
  `--out` only as its very last statement (`:283-287`), so a
  timeout-killed-by-the-orchestrator or crashed run genuinely leaves no JSON
  and is correctly retried on restart; a normally-succeeded run's JSON is
  written in one shot and correctly skipped. (Exception: external
  mid-write kills — see M4.)
- **Aggregation key contract matches byte-for-byte**: `effective_rank_mean`,
  `recovered_frac@0.99`, `force_rank_k`, `d`, `K` as read in `aggregate()`
  (`run_overnight.py:236-237, 244-245`) match exactly what `evaluate()` writes
  in `run_task_d.py:98-111`.
- **Manifest is well-formed**: 421 unique run names (verified
  programmatically — zero collisions), no `force_rank_k > d` anywhere, no
  `orthogonal=True` with `K > d` anywhere (the K>d lossy-regime tier
  explicitly passes `--gaussian`, never relying on the silent
  orthogonal-QR-fallback in `task_d.py:47,50`). `straddle(K, d)` correctly
  brackets K with low anchors {1,2} and clamps into `[1, d]`.
- **Cross-tier ordering is correct**: all of Tier 1 (`d=16` decisive gate)
  sorts before Tier 2 (generality) and Tier 3 (scale), matching the stated
  design goal (within-tier ordering is the M6 finding above).

---

## Verdict

**Not safe to launch unattended for 6-8 hours as-is.** F1 (no exception
isolation around subprocess launch) is disqualifying on its own: it is the
same failure class — one unhandled exception during subprocess creation
ending the whole run early with no retry and no signal until morning — that
lost 24/27 runs in `rank_aware_v1`, and it directly contradicts the script's
own "cannot corrupt the orchestrator" design claim. M2-M5 compound the risk
(GPU reuse racing a hung kill, a timeout too tight for the most expensive
tier, silent permanent loss of preempted runs, and a possible end-of-run crash
that hides an otherwise-successful night). M6 doesn't threaten data loss but
undermines the "useful partial prefix" goal for exactly the primary
causal-test tier.

**Minimum bar before launch:** fix F1 (try/except around the launch body,
never let a launch-time exception propagate) and M2 (verify the killed
process actually died before reusing its GPU). Strongly recommend also fixing
M3 (per-tier timeout) and M4 (validate JSON before treating as done) before
committing 8 H100s to a full unattended night — both are plausible, not just
theoretical, given the manifest's own step-count spread and H100 preemption
realities. M5/M6/minors are cheap, worth doing in the same pass but not
individually disqualifying.
