# Matrix-thinking workshop-paper era — archived 2026-07-04

Design, audit, and results documents from the bolt-on matrix-CODI /
ICML MI Workshop 2026 paper era (April-May 2026). The program these
documents cover is closed: the paper ("The Gradient Does Not See Rank")
was submitted and **accepted**. Superseded or fully absorbed into a living
document at archive time — see below for where each file's content landed.
Moved here, not deleted, per the project's "dead directions stay dead but
keep the record" policy (see `../README.md`).

- **BILINEAR_READOUT_PATCH_PLAN.md** — design for four nonlinear-in-Z
  readout variants testing the Jacobian-linearity thesis. All four were
  built and run; the result (flat rank-k curves regardless of readout
  nonlinearity) is preserved in `matrix-thinking/KILL_LIST.md` ("Lesson 5")
  and the accepted paper itself.
- **CHAPTER_2_DESIGN.md** — the original Chapter 2 plan (Task A,
  K-parallel-entity-tracking). Superseded by
  `matrix-thinking/chapter2/TASK_D_PREREGISTRATION.md`, which states
  explicitly that it supersedes this design (Task A was killed by the
  2026-07-01 design gauntlet — the position-decomposition escape).
- **PAPER_RESULTS_SUMMARY.md**, **PAPER_WRITER_BRIEF.md** — working results
  tables and a paper-writing brief, both superseded by the actual accepted
  paper at `matrix-thinking/submissions/icml-mi-workshop-2026/`.
- **PRE_SUBMISSION_OUTREACH.md** — a pre-submission outreach plan for the
  May 8, 2026 deadline. The deadline has passed and the paper is accepted;
  no evidence the outreach emails were sent (no log file exists anywhere in
  the repo).
- **RANK_AWARE_RESEARCH_COMPILATION.md** — an 8-agent research compilation
  ranking follow-up experiment ideas post-workshop-paper. Plan A
  (ProsQA-MULTI) was executed as `rank_aware_v1` (2026-04-29); the decisive
  negative result (matrix-CODI composes via position, not within-position
  rank) is preserved in `EXPERIMENT_LOG.md` and `STATE.md`'s Chapter-2-kickoff
  narrative.
- **CONTROL_A_RESEARCH_REPORT.md, CONTROL_A_EXECUTION_BRIEF.md,
  CONTROL_A_ATTACK_REPORT.md, CONTROL_A_IMPLEMENTATION_NOTES.md,
  CONTROL_A_AUDIT_REPORT.md, CONTROL_A_FIX_RECEIPT.md** — the six-document
  design/audit/attack lifecycle for Control A (the vanilla-GPT-2
  rank-1-solvability null baseline). Consolidated, with the actual
  2026-04-28 run result (which had never been logged anywhere), into
  `matrix-thinking/CONTROL_A_HISTORY.md`.
