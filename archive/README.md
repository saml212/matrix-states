# Archive

This folder holds dead ends, superseded designs, and historical material from the project. Nothing here is currently active. Read this folder if you're studying the project's history or want to understand why a direction was abandoned. If you want to know the current state of the project, read [STATE.md](../STATE.md) instead.

## Contents

### Top-level archived files

- **ARCHITECTURE.md** — The original "autoregressive matrix thought generation" architecture spec. The core mechanism (autoregressive thought appending with halt signal) was killed by the April 2 attack analysis. See `ARCHIVE_NOTE_ARCHITECTURE.md` for why.
- **ARCHIVE_NOTE_ARCHITECTURE.md** — Why ARCHITECTURE.md was archived and what replaced it.
- **ARCHIVE_NOTE_byte-agnostic.md** — Why the byte-agnostic project was archived and what survived.

### Subfolders

- **matrix-thinking-workshop-era/** (added 2026-07-04) — Design/audit/results
  docs from the bolt-on matrix-CODI / ICML MI Workshop 2026 paper era
  (accepted). Includes the six-document Control A lifecycle (consolidated
  into `matrix-thinking/CONTROL_A_HISTORY.md`), the superseded original
  Chapter 2 plan, and the pre-submission/paper-writing scratch. See its own
  README for the file-by-file supersession map.
- **chapter2-gauntlet/** (added 2026-07-04) — Task D/Task E design-gauntlet
  attack/audit/research rounds (14 files) plus Task D's superseded findings
  draft and deploy runbook. Both programs are CLOSED — CONFIRMED (see
  `matrix-thinking/chapter2/TASK_D_WRITEUP.md`,
  `matrix-thinking/chapter2/TASK_E_FINDINGS.md`). See its own README for
  which findings were folded into living docs before archiving.
- **team-output-2026-04-28/** (added 2026-07-04) — The planning gauntlet for
  `rank_aware_v1`, superseded by that experiment's 2026-04-29 execution
  (`experiment-runs/2026-04-29_rank_aware_v1/`). See its own README.
- **byte-agnostic/** — The original Phase 1-2 work: PHM layers + learned byte segmentation. 7 experiments. Best result 2.34 BPC on multi-modal data with fixed-stride. Killed because (a) PHM converges to nilpotent rather than learning meaningful algebra, (b) the matrix-thinking thread is a stronger version of the same intuition with a clearer observable. See `ARCHIVE_NOTE_byte-agnostic.md`.

- **algebra_ablation/** — PHM algebra ablation results. Tested whether quaternion-fixed structure beats learned PHM. Result: nilpotent (learned) wins on task performance even though it loses the algebraic structure.

- **algebra_reg/** — PHM algebraic regularization results. Tested whether a soft regularizer can preserve quaternion structure. Result: yes, regularization preserves the structure (dist=0.02 vs 0.67 without), but nilpotent still wins on task performance.

- **old-docs/** — 13 superseded documents from earlier project phases:
  - BUILD_PLAN.md, conceptual-framework.md, CONVERSATION_CONTEXT.md, EXPERIMENT_VARIABLES.md, experiment-plan.md, H100_ENVIRONMENT.md, H100_EXPERIMENT_QUEUE.md, HANDOFF.md, MATRIX_THINKING_ARCHITECTURE.md, matrix-thinking-README.md, PARAM_MATCHING.md, PROPOSAL.md, vision.md
  - Most of these were earlier versions of files that have since been consolidated into STATE.md, EXPERIMENT_LOG.md, references.md, or matrix-thinking/QUEUE.md.

### Top-level files

- **algebra_reg.py, phm.py, run_algebra_*.py** — PHM analysis scripts. Reproduce the negative results documented in `research/phm-nilpotent-convergence.md`. Won't run without the byte-agnostic codebase, which lives in `byte-agnostic/`.

## Why dead ends are kept

CLAUDE.md hard rule: "Dead directions stay dead. Don't revisit archived ideas unless the user asks." But the archive serves three purposes:

1. **Avoiding reinvention.** A future agent that reads STATE.md and gets enthusiastic about, say, PHM layers can come here and find out they were tested for 7 experiments and converge to nilpotent.
2. **Negative results for publication.** PHM-converges-to-nilpotent is a publishable empirical finding even though it's negative. The data is here.
3. **Learning from failures.** The attack-and-research workflow that killed these directions is itself valuable as a record of how the project narrows hypotheses over time.

## Don't add new things here without a reason

If a current experiment fails or a current direction is killed, archive it here with a `WHY_ARCHIVED.md` file explaining the kill reason. Don't archive things speculatively; the archive should grow slowly.
