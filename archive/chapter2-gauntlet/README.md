# Chapter 2 (Task D / Task E) gauntlet + scratch — archived 2026-07-04

Process artifacts (design-gauntlet attack/audit/research rounds) and
superseded scratch from Task D and Task E, both now CLOSED — CONFIRMED (see
`matrix-thinking/chapter2/TASK_D_WRITEUP.md`,
`matrix-thinking/chapter2/TASK_E_FINDINGS.md`, and `STATE.md` "Chapter 2 —
STATUS"). Every finding below was verified, during this consolidation pass,
to have survived into a living document — grep phrases point to where.

- **DEPLOY.md** — Task D's deploy/overnight-run runbook. Superseded by
  `matrix-thinking/H100_SETUP.md` §"Perpetual/unattended sweep pattern",
  which distills the same pattern without DEPLOY.md's now-stale box
  specifics (wrong SSH user, pre-correction GPU-hour framing).
- **TASK_D_FINDINGS_DRAFT.md** — an earlier, 830-run snapshot of Task D
  results. Superseded by `matrix-thinking/chapter2/TASK_D_WRITEUP.md`, which
  carries everything substantive plus two later dated corrections this
  draft still gets wrong (the Stage-0 d≥32 reframing; the d=16,K=4
  budget-confounded "ceiling").
- **gauntlet/** (14 files) — attack/audit/research rounds for Task D and
  Task E. 11 of 14 archive with their finding fully preserved in a living
  doc (`TASK_D_PREREGISTRATION.md`, `TASK_D_WRITEUP.md`,
  `NEXT_EXPERIMENT_DESIGN.md`, `EXPERIMENT_LOG.md`, or the CLAUDE.md hard
  rules — e.g. `AUDIT_task_e_validity.md`'s cycle-periodicity confound is
  now a standing hard rule). 3 files (`AUDIT_overnight.md`,
  `AUDIT_overnight_r2.md`, `AUDIT_round3.md`) had residual
  infrastructure/instrumentation findings not fully captured elsewhere —
  folded into `matrix-thinking/H100_SETUP.md`'s sweep-pattern section (the
  `write_progress()`/`lf.close()` unguarded-write gap) and
  `matrix-thinking/chapter2/TASK_D_WRITEUP.md` (the `m1_trends_up` proxy
  caveat) before archiving.
