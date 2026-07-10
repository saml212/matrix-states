# MANIFEST — §3.8(c) negative-test archive

Gate: `matrix-thinking/NOVEL_ARCH_WATERFALL.md` §3.8(c) — the corrected trust
rule's TWO pinned negative tests (N1 amplifying near-normal D, N2 non-normal
nilpotent D), executed to completion, output archived, per the pin at §3.4
and the construction added at §3.9 MA1, verified clean by the §5 scoped
micro-attack (zero math-level defects; this run's numbers cross-check
against §5's recorded simulation values to stated precision).

Executed 2026-07-10. Repo commit at execution time:
`1a2fee94c8d1b6c3065b4e39e7c9ebf5eb415dbf`. Discharge recorded at
`matrix-thinking/NOVEL_ARCH_WATERFALL.md` §6.

## Files and MD5 checksums

| File | MD5 |
|---|---|
| `../test_trust_rule_negative.py` (the script, one directory up) | `3879e56de2158c028dc2768dbcc93fd0` |
| `results.json` | `c5c8f81847cbb8731090604f5d4b1045` |
| `run.log` | `37d0850cd30019da8c7b4acc4a756ed3` |

Recompute with `md5 <file>` (macOS) or `md5sum <file>` (Linux).

## Result summary

- **N1** (amplifying near-normal D): 11 checks, all PASS. T_lin(21) =
  0.2100 (pinned range [0.20, 0.22]; matches §5's recorded 0.2100 exactly);
  corrected T(21) = 99.7377 (pinned criterion > 10; matches §5's recorded
  99.74 to stated precision); measured junk/signal = 22.83 (pinned
  criterion > 1; §5/Rev-1's own instance was a different, illustrative,
  RNG-order-dependent draw — only ">1" is a pinned criterion, not an exact
  match, and is met with wide margin).
- **N2** (non-normal nilpotent D = 100·E01, fully deterministic, no RNG):
  18 checks, all PASS. e3 occupancy confirmed at steps 3, 11, AND 19 (nit
  n1); D²=0 confirmed exactly; the j=4 and j=12 injection terms confirmed
  annihilated, the j=20 term confirmed surviving; rho-based (OLD) T(21) =
  0.0100 (pinned criterion ≤ 0.011; matches §5's recorded 0.0100 — falsely
  ADMITS at tau=0.2); corrected (sigma) T(21) = 1.010101×10³⁸ (pinned
  criterion > 10³⁰; matches §5's recorded 1.010×10³⁸); junk/signal = 1.0
  exactly (pinned criterion in [0.9, 1.1]; matches §5's recorded 1.0 and
  the deterministic-construction expectation); cosine = 0.707107 (matches
  §5's recorded 0.7071 = 1/√2 exactly).

**Overall: 29/29 checks PASS, process exit code 0.**

## Teeth verification

A throwaway mutated copy (D_n2[1,0] = 5.0 added alongside the pinned
D_n2[0,1] = 100.0, breaking D²=0) was run outside this repo (scratchpad),
confirmed to FAIL (17/29 checks pass, exit code 1 — the D-nilpotency, D²=0,
both annihilation-term checks, the fast-vs-naive matrix_power cross-check,
the analytic-vs-direct recursion cross-check, both rho-based pass criteria,
and all junk/signal/cosine checks all correctly fail), then deleted. The
unmodified script was re-run twice afterward and diffed byte-for-byte
against the original clean run: identical in both cases. Full transcript in
`run.log`.

## Provenance note

The script rewrites `results.json` and `run.log` as a side effect of every
invocation (by design, so a fresh run always produces a self-consistent
pair). `run.log` in this archive was manually reassembled after all three
runs (canonical, mutated, restored) completed, from each run's captured
stdout, so the single archived file preserves the full record instead of
only the last invocation's bare output; `results.json` is the literal
output of the final canonical (clean) run, byte-for-byte identical in its
`checks` array to the two other clean runs.
