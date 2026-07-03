# Experiment Queue

**Last updated:** 2026-07-04 (banner refreshed; historical body split out — see below)

This document is the engineering queue for future experiments. Strategic
narrative and project state live in [STATE.md](../STATE.md).

---

> **STATUS AS OF 2026-07-04 — this file's original ~570-line body (matrix-CODI
> / workshop-paper era, Priority 0-7 items) is fully closed or superseded and
> has been split out to
> `archive/matrix-thinking-workshop-era/QUEUE_historical.md` to keep this
> file short.** Where each item's successor work landed:
> - Control A (Priority 0): **done**, `matrix-thinking/CONTROL_A_HISTORY.md`.
> - Priority 1 (matrix-CODI rank dynamics): **done** — negative result
>   (rank-blind), the accepted ICML MI Workshop 2026 paper's headline finding.
> - Chapter 2 (Task D, tensor-product key/value binding): **CLOSED,
>   CONFIRMED** at d=8,16. `matrix-thinking/chapter2/TASK_D_PREREGISTRATION.md`,
>   `matrix-thinking/chapter2/TASK_D_WRITEUP.md`.
> - Chapter 2.5 (Task E, compositional reasoning transfer): **CLOSED, gate
>   PASSED / CONFIRMED.** `matrix-thinking/chapter2/NEXT_EXPERIMENT_DESIGN.md`,
>   `matrix-thinking/chapter2/TASK_E_FINDINGS.md`.
> - Stage 0 (d≥32 trainability precursor): **CLOSED** — the wall was
>   substantially a step-budget artifact; the honest frontier is exactness,
>   not trainability. `matrix-thinking/chapter2/STAGE0_DESIGN.md`.
> - DeltaNet causal-rank + DeltaNet real-data (production-architecture
>   rank-K binding, synthetic then real tokenized text): both **CLOSED,
>   CONFIRMED**. `matrix-thinking/DELTANET_CAUSAL_RANK_DESIGN.md`,
>   `matrix-thinking/DELTANET_REALDATA_DESIGN.md`.
> - Stage G (matrix-vs-vector per-FLOP gap mechanism): **CLOSED** — named
>   mechanism (Kronecker-separable projection restriction).
>   `matrix-thinking/STAGE_G_DESIGN.md`.
> - **Now active:** the exactness mechanism study (why real-text composition
>   falls short of the synthetic razor cliff) — living, not closed.
>   `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md`. Byte-level input
>   remains a high-priority, explicitly-not-yet-started follow-on ablation
>   once matrix-native scales to real data (tokenization held fixed, not
>   bundled with the matrix change — see STATE.md's "Path Forward").
>
> See STATE.md's "Chapter 2 — STATUS" section for the full current
> narrative and the campaign ledger.
