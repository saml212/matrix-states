# AUDIT r2 — run_overnight.py (rewrite verification)

Scope: confirm round-1 findings (`gauntlet/AUDIT_overnight.md`) are genuinely fixed
in the rewritten `run_overnight.py`, and hunt for new bugs introduced by the
rewrite. Static review, no torch/GPU here — but `run_overnight.py` itself has
zero torch dependency, so `build_manifest()`, `is_done()`, `aggregate()`, and
the TAU string contract were **actually executed** (not just read) against
synthetic inputs to get ground truth instead of manual arithmetic. Commands
and outputs are reproduced inline below each finding.

---

## Part A — round-1 findings re-verified

**F1 (FATAL) — launch loop exception isolation → PARTIAL.**
The cited site (`open()` + `Popen()`, `:191-200`) is now correctly wrapped:
```python
try:
    lf = open(...); proc = subprocess.Popen(...)
    running[gpu] = (proc, spec, lf, time.time())
except Exception as e:
    failed.append((spec["name"], "LAUNCH_ERR")); free.append(gpu)
```
`running[gpu]` is assigned only after `Popen()` succeeds, so a failure here
can never leave a partial `running` entry, and `free.append(gpu)` returns the
GPU exactly once. This closes the specific FATAL site.

However, round-1's fix also asked for a top-level `try/except` around
`main()`'s loop / the `aggregate()` call to emit `CRASHED.txt` — **not done**
(no top-level try/except anywhere; `if __name__ == "__main__": main()` is
bare). More importantly, the *same failure class* (uncaught OS error killing
the whole orchestrator) still lives one level down: `write_progress()`
(`:140-146`, called after **every** completion/timeout event — ~421+ times
over the night) and the `lf.close()` calls in the harvest loop (`:211,218`)
do unguarded file I/O with zero exception handling. A disk-full/NFS hiccup
there (plausible: 421 logs + 421 JSONs + repeated `PROGRESS.txt` writes on
one filesystem for 6-8h) still propagates out of `main()` uncaught and kills
the whole process — same symptom as the original F1, just relocated. See NB1.

**M2 (MAJOR) — timeout `.wait()`-reaps before GPU reuse → PARTIAL.**
```python
proc.kill()
try:
    proc.wait(timeout=30)
except Exception:
    pass
lf.close(); del running[gpu]; free.append(gpu)
```
`.wait()` was added, which fixes the common case (SIGKILL delivered, process
reaped cleanly). But the exact pathological case M2 was about — SIGKILL
undeliverable to a process wedged in an uninterruptible kernel wait (driver
hang / Xid, "a real H100 failure mode" per round-1) — hits
`except Exception: pass` (**no log line**, nothing) and the code then
unconditionally frees the GPU anyway. Round-1's recommended fix explicitly
said: on wait-timeout, log CRITICAL and **do not** return the GPU to `free`.
That branch was not implemented — the poisoned-GPU-reuse race still exists,
just now gated behind "kill takes >30s to reap" instead of firing on every
timeout.

**M3 (MAJOR) — per-tier/per-d timeout → FIXED.**
```python
TIMEOUT = {8: 1200, 16: 1200, 32: 2400, 64: 4800, 128: 7200}  # seconds
...
if time.time() - start > TIMEOUT.get(spec["d"], 3600):
```
Scaled by `d` with a safe fallback for any unlisted `d`. Headroom grows
faster than `STEPS` alone would suggest (d=128 gets 1.5x d=64's timeout for
equal step counts), consistent with per-step matrix-op cost scaling. No
in-script evidence of an empirical d=128 timing run to calibrate the exact
constants, but the structural bug (one global timeout for all tiers) is gone.

**M4 (MAJOR) — resume validity-checked → FIXED.** Verified by execution:
```
is_done on corrupt file:                    False
is_done missing effective_rank_mean:        False
is_done valid file:                         True
```
`is_done()` requires `json.load` to succeed **and** both `"mean_cos"` and
`"effective_rank_mean"` present. Cross-checked against `run_task_d.py`:
`mean_cos` comes from `_recovery_stats(cos)` called with the default empty
prefix (`run_task_d.py:100`), and `effective_rank_mean` is written whenever
`model_type == "matrix"` (`:103`) — which the orchestrator always requests
(`build_cmd` hardcodes `--model matrix`). Same `is_done()` function is used
for both the initial pending/done split (`:178-179`) and the post-completion
re-check (`:221`) — no drift between the two call sites.

**M5 (MAJOR) — aggregate guarded → FIXED (coarse-grained).** Verified by
execution: forced a record with `effective_rank_mean` present, `force_rank_k`
None, and `"d"` missing (the exact schema-drift scenario round-1 described):
```
SUMMARY.txt:
{
  "n_failed": 1,
  "failed": ["some_failed_run"],
  "n_runs": 1,
  "aggregate_error": "KeyError('d')"
}
```
`SUMMARY.txt`/`AGGREGATE.json` are still written, with `n_failed`/`failed`/
`n_runs` intact and the error surfaced. Caveat: the guard is one `try/except`
around the whole aggregation body, not per-record — the first bad record
aborts the `M1_effrank_vs_K_by_d`/`M3_..._vs_forcerank` breakdown for *all*
records, not just itself (per-record skip-and-count from the round-1
suggestion wasn't implemented). The actual round-1 complaint — a malformed
record silently costing the whole night's summary — is resolved.

**M6 (MAJOR) — Tier-1 M1/M3 interleaved → FIXED.** Verified by execution:
```
tier1 total 195  m1 80  m3 115
tier1 head: ['t1_d16_K8_fr1_s0', 't1_d16_K1_frN_s0', 't1_d16_K8_fr1_s1',
             't1_d16_K1_frN_s1', ...]
first 160 strictly alternate: True
```
`_interleave(t1_m3, t1_m1)` (M3 leads, per the code comment) alternates
strictly for the first 160 of 195 entries (80 M3 + 80 M1 paired 1:1), then
trails with the remaining 35 M3-only entries once M1 (80 total) is exhausted.
Any interruption before run 160 leaves equal M1/M3 coverage; interruption
after 160 only adds extra M3 (the "primary causal test"), never starves it.
Manifest totals (421 / tiers {1:195, 2:184, 3:42} / 421 unique names / 0
`force_rank_k > d` / 0 `orthogonal & K > d`) all reconfirmed identical to
round-1's verified counts — the rewrite did not perturb anything outside the
interleaving change.

---

## Part B — new / residual findings

**NB1 (MAJOR).** `write_progress()` and the harvest-loop `lf.close()` calls
are unguarded OS I/O inside the main poll loop, structurally identical to the
original F1 bug and not covered by its fix (see F1 discussion above). No
top-level `try/except` exists anywhere in `main()` or at module scope. A
transient disk/NFS error here still takes down the whole orchestrator
uncaught, mid-run, with no `CRASHED.txt` and no signal beyond a stalled
`PROGRESS.txt` — exactly the "discovered only the next morning" failure mode
the rewrite's docstring claims to have eliminated. **Fix:** wrap
`write_progress()` call sites and `lf.close()` in
`try/except Exception: print(...)` (never let it propagate), and add one
top-level `try/except` around `main()`'s `while` loop + the `aggregate()`
call that writes `CRASHED.txt` with the traceback before exiting — this was
explicitly requested by the original F1 fix and is still missing.

**NB2 (MINOR).** File-descriptor leak on a specific sub-case of the F1 fix:
if `open()` succeeds but the subsequent `Popen()` raises, the already-open
`lf` handle is never explicitly closed in the `except` block. In CPython
this is closed deterministically via refcounting when `lf` is reassigned on
the next loop iteration, so it's not a practical fd-exhaustion risk (same
class as round-1's `m9`), but add `try: lf.close(); except Exception: pass`
in the except branch for portability/defense-in-depth.

**NB3 (carried-over, still open, not part of the 6 re-verified items but
directly relevant to "strand the run overnight").** Confirmed by source
inspection — neither was touched by the rewrite:
- `m8`: no `--gpus` validation (`assert args.gpus` absent). `--gpus 0`
  still produces a silent, resource-cheap infinite poll loop all night
  (`pending` non-empty, `running` always empty, no `ABORTED.txt`).
- `m7`: `run_smoke()`'s `subprocess.call(...)` still has no `timeout=`. A
  wedged GPU 0 / driver before the run even starts hangs the smoke gate
  forever, consuming the whole unattended window with zero progress.
Both are one-line fixes (`assert args.gpus >= 1`; `timeout=600` on the smoke
call) and both independently satisfy "strands the run overnight" — worth
fixing in the same pass even though out of the original 6-item scope.

**Bonus (unrequested, already fixed):** round-1's `m9` (unclosed handle in
`aggregate()`) is resolved in the rewrite — now `with open(p) as f:`.

**Smoke-gate abort still blocks all training — confirmed unchanged.**
`run_smoke()` is checked and `sys.exit(1)` fires (after writing
`ABORTED.txt`) strictly before `pending`/`running` are constructed
(`:172-188`); no training subprocess is reachable on smoke failure. Correct,
undisturbed by the rewrite.

---

## Verdict

**Meaningfully safer than round 1, but not yet fully safe to walk away from
for 6-8 hours unattended.** The disqualifying FATAL (F1) is fixed at its
literal site, and M3/M4/M5/M6 are solidly fixed (M4/M5/M6 verified by actual
execution above, not just reading). But two gaps remain that map onto the
exact same "strand the run overnight" failure class the rewrite set out to
eliminate: NB1 (write_progress/lf.close still unguarded, same bug pattern as
F1, one disk hiccup away from killing the whole orchestrator with no
`CRASHED.txt`) and M2's incomplete fix (a genuinely wedged H100 process still
gets its GPU silently freed and reused after the 30s reap attempt fails,
poisoning a subsequent run with no log trace). Recommend a second small patch
pass — guard `write_progress`/`lf.close`, add a top-level `CRASHED.txt`
handler, and quarantine (don't free) a GPU whose kill-reap timed out — before
committing all 8 H100s to the full unattended run. `m7`/`m8` are cheap
one-liners worth bundling into the same pass.
