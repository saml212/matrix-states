# team-output-2026-04-28 — archived 2026-07-04

The 2026-04-28 planning gauntlet (brainstorm/research/attack/audit/review)
for the `rank_aware_v1` experiment (ProsQA-MULTI, force-rank-1 causal test).
**Superseded, not lost:** the 2026-04-29 execution obsoleted this program —
forcing Z to rank-1 throughout training did *not* hurt accuracy, confirming
the position-decomposition escape this gauntlet's own `METH_VERDICT.md`
predicted. That prediction, and the executed result, are fully migrated into
`STATE.md`, `EXPERIMENT_LOG.md`, `matrix-thinking/chapter2/
TASK_D_PREREGISTRATION.md`, and `matrix-thinking/DELTANET_CAUSAL_RANK_DESIGN.md`
§4.2. All bug findings (dataset-builder option-order leak, SVD-backward NaN)
were fixed and re-verified before the run; the fixes live on in
`experiment-runs/2026-04-29_rank_aware_v1/`'s script.

Not a duplicate of `experiment-runs/2026-04-29_rank_aware_v1/SYNTHESIS.md`:
this directory's `SYNTHESIS.md` is the pre-run *planning* synthesis; the
`experiment-runs/` one is the post-hoc *results* synthesis and is the
authoritative, later document (kept in place — `experiment-runs/` is this
project's permanent-record convention for exact per-experiment scripts and
results).
