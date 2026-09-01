# cancelled/ — job specs deliberately KILLED mid-flight (never deleted, never re-claimed)

Workers only ever look at `pending/` and at their own `claimed/*.g<N>.json`, so a
spec parked here is inert: it cannot be re-claimed. A cancel is reversible — `mv`
the file back into `pending/` (dropping the `.cancelled` suffix) to re-activate
it. This mirrors the existing `parked_*/` convention.

## 200_laneB_1p31b_arm_per_token_openr1_s0 — cancelled 2026-07-13

Killed at step ~86,000 / 183,105 (33.4h in) with
`tmux kill-session -t queue_worker_g1` — the QUEUE_README preemption contract: a
job runs as its worker's own synchronous child, so killing the worker's EXACT
tmux session name kills the in-flight job too. Never `pkill` (a pattern can match
the SSH command string invoking the kill and self-kill the shell). Worker g1 was
relaunched immediately afterwards with `bash ~/queue/launch_workers.sh 1`.

This spec was moved OUT of `claimed/` before g1 was relaunched, because
`queue_worker.sh` sweeps `claimed/*.g<N>.json` back into `pending/` on startup
(its resume-safety path) and would otherwise have re-queued the very job we
killed.

Cancelled on three independent grounds, each re-verified against the box:

1. **DUPLICATE.** `000_laneB_1p31b_arm_per_token_openr1_s0_pricefix` (running on
   GPU 4) has an identical training config: d-model 2560, d-state 128, n-layers
   22, seq-len 512, batch 16, steps 183105, seed 0, per_token / lambda 0.58,
   corpus openr1-mix-ext. Two H100s were computing the identical cell. The two
   jobs differ ONLY in `--internal-timeout` (160000 vs 340000) and in their
   ckpt/out paths — so killing 200 cannot touch 000's checkpoints.

2. **DOOMED.** 200's `--internal-timeout 160000` (44.4h) was priced off a wrong
   rate (0.7135 s/step, SCALE_TRANSFER_DESIGN.md S5.6). The MEASURED rate is
   ~1.3997 s/step — from this job's own log: step 86,000 at 120,371 s of wall
   clock. The full 183,105 steps need ~256,300 s = 71.2h, so 200 would have
   self-terminated at ~62% of budget with `complete=false`
   (lm_pretrain_rd.py:2160), been routed to `failed/` by its own validity check,
   and produced no admissible cell. `000` carries the corrected 340000 s.

3. **UNUSABLE.** PARAM_AXIS_SCALING_DESIGN.md §21 proved the 1.31B rung can never
   enter the param-axis primary fit: admitting it would need a common token slice
   T >= 1.311B, but 98M/392M cap at 1.107B; and §9.6 item 6 requires BOTH
   corpora, while no wikitext-mix-ext 1.31B cell exists or is queued.

The GPU this freed was handed to `031` / `032` — the 14M full-token extension,
which raises the common slice to 1.10669B and makes |A| = 3, so the trend verdict
becomes askable. See PARAM_AXIS_SCALING_DESIGN.md §22.
