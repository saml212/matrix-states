"""ON-BOX QUEUE JOB-SPEC GENERATOR. Produces the plain JSON job-spec files
that matrix-thinking/queue/queue_worker.sh consumes. Run locally (this
script does not launch anything or touch the box); the output directory
(jobs/pending/) is what gets scp'd to /home/nvidia/queue/pending/ at deploy
time.

Three lanes, each reusing an ALREADY-AUDITED, already-running-on-box script
(no new science, no unregistered protocol changes):

  LANE A (front, cheap, informative first): the NCR early-LN K-scaling
  ladder, extended to K in {20,32,48,64,96,128,192,256} via the additive
  GRIDS/GRID_SHAPES formula landed in ncr_task.py / ncr_earlyln_scale.py
  this same session (matrix-thinking/NOVEL_ARCH_WATERFALL.md S11). Reuses
  ncr_earlyln_scale.py's own --cell CLI verbatim, one process per cell.

  LANE B (the 1B-visibility lane): matrix-native fast-weight LM scale +
  seed-extension work on the fix-at-scale program (matrix-thinking/
  FROZEN_BIAS_LM_DESIGN.md S13), reusing lm_pretrain_rd.py's own CLI
  verbatim (the exact flags the CLOSED S13.22 wave already used).

  LANE C (evergreen tail): seed-replication of Lane A's own K-ladder cells
  (deepens statistical power on the SAME already-registered cells, not new
  science) -- the house fallback for filling remaining budget honestly.

Every job's cmd is a self-contained shell command; every job carries an
independent validity_check (house rule: check output validity, not mere
existence) and a one-line hypothesis (house rule: no experiment without a
stated hypothesis). GPU-h estimates and their confidence basis are recorded
in each spec's "notes" field.

Priority convention (filename prefix, ascending = claimed first):
  000-099  Lane A rate probes (500 steps, ~seconds each -- de-risks the
           formula-extrapolated GPU-h estimates before the main manifest)
  100-199  Lane A main manifest, K ascending (small/cheap/informative first,
           K>=192 last, matching the task's own "K>=96 behind them" intent
           generalized to the true cost tail)
  200-299  Lane B (long single-job runtimes; good for unattended overnight
           throughput)
  300-399  Lane C (evergreen tail, deepens Lane A's own cells)
  900-999  One-off deferred/low-priority diagnostic cells -- deliberately
           numbered above every currently-pending Lane A/B/C job so a
           worker only ever claims one after the sweep's own backlog is
           exhausted (PARAM_AXIS_SCALING_DESIGN.md sec 13.5's T2a-3
           deferral is the first tenant).

Usage: python3 generate_jobs.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared box paths (matrix-thinking/H100_SETUP.md convention)
# ---------------------------------------------------------------------------
PY = "/home/nvidia/tdenv/bin/python3"
NCR_DIR = "/home/nvidia/ncr"
DELTANET_RD_DIR = "/home/nvidia/chapter2/deltanet_rd"
DATA_DIR = "/data/deltanet_rd_data"
FIXSCALE_CKPT_ROOT = "/data/fixscale_ckpts/train"
FIXSCALE_RESULTS_ROOT = f"{DELTANET_RD_DIR}/results/fixscale/train"
TRACKC_CKPT_ROOT = "/data/queue_1p31b_ckpts"
TRACKC_RESULTS_ROOT = f"{DELTANET_RD_DIR}/results/queue_1p31b"

# ---------------------------------------------------------------------------
# LANE A -- NCR K-scaling extension (K20..K256), reusing ncr_earlyln_scale.py
# ---------------------------------------------------------------------------
# F(K,d,h) = 76*K*h^2 + 4*d*h^2 + 12*K^2*h + 4*K*d*h + 4*d^2*h  (S9.3-corrected
# cost formula, already audited/used in this exact program). At the
# Condition-A convention d=2K, h=64 this reduces to F = 344064*K + 2304*K^2.
# Calibration anchor: the LIVE opbank_earlyln_s7/s8 sessions (K=24, d=48,
# h=64, launched minutes before this queue was built) measured ~0.0228 s/step
# on an idle H100 -> 80,000 steps = 0.507 GPU-h/cell. Every other K's
# estimate below is F(K)/F(24) * 0.507 GPU-h -- a FORMULA EXTRAPOLATION from
# one measured point, not itself measured. Each cell's own --ceiling-gpuh
# breaker (set to ~2x the estimate, floor 1.0h) bounds worst-case cost
# regardless of extrapolation error -- this is the real safety rail, not the
# estimate's precision. A 500-step rate-probe cell (000-block) runs before
# the main manifest for every new K, exactly mirroring this program's own
# established Phase-0a discipline (S9.9), so the estimates below can be
# corrected from real data before the bulk of the budget commits.
F24 = 344064 * 24 + 2304 * 24 * 24


def f_cost(K: int) -> float:
    F = 344064 * K + 2304 * K * K
    return 0.507 * (F / F24)


LANE_A_KS_FULL = (20, 32, 48, 64, 96, 128)   # n=4 seeds (the task's own ask)
LANE_A_KS_DISCLOSED = (192, 256)             # n=2 seeds (disclosed-only, cost/risk tail)
STEPS_MAIN = 80_000       # S9.10/S11's own pinned per-cell budget, unchanged
STEPS_PROBE = 500         # S9.9's own Phase-0a rate-probe budget, unchanged

EARLYLN_OUTDIR = f"{NCR_DIR}/results_earlyln_scale"   # SAME outdir the live §11
                                                       # sweep (opbank_earlyln_s*)
                                                       # is already writing K=14/
                                                       # 15/16/24 into -- K=20..256
                                                       # land in the same harvest.
EARLYLN_PROBE_OUTDIR = f"{NCR_DIR}/results_earlyln_scale_probe"


def lane_a_jobs() -> list[dict]:
    jobs = []

    # -- rate probes (000-block), one per new K, 1 seed, 500 steps --
    for i, K in enumerate(LANE_A_KS_FULL + LANE_A_KS_DISCLOSED):
        est = max(0.005, f_cost(K) * (STEPS_PROBE / STEPS_MAIN))
        jid = f"{i:03d}_laneA_probe_K{K}_s0"
        cmd = (
            f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K {K} --seed 0 "
            f"--steps {STEPS_PROBE} --outdir {EARLYLN_PROBE_OUTDIR} "
            f"--ceiling-gpuh 0.5"
        )
        vcheck = (
            f"{PY} -c \""
            f"import json; d=json.load(open('{EARLYLN_PROBE_OUTDIR}/earlyln_K{K}_s0.json')); "
            f"assert d.get('status')=='COMPLETED'; "
            f"assert d.get('train',{{}}).get('step')=={STEPS_PROBE}; "
            f"assert 'eval' in d and 'deep_probe' in d\""
        )
        jobs.append(dict(
            id=jid, lane="A",
            hypothesis=(f"Rate probe only: measures real per-step wall time for the "
                        f"earlyln K={K} shape (d=2K={2*K}, h=64) BEFORE the 80,000-step "
                        f"main manifest commits its budget, mirroring S9.9's own "
                        f"Phase-0a discipline. Not itself a trainability readout."),
            cmd=cmd, gpu_h_estimate=round(est, 3),
            output_dir=EARLYLN_PROBE_OUTDIR, validity_check=vcheck,
            notes=(f"formula-extrapolated estimate (F(K)/F(24) scaling from the "
                   f"measured K=24 rate ~0.507 GPU-h/80k-step-cell); --ceiling-gpuh "
                   f"0.5 hard-bounds worst case regardless of extrapolation error."),
        ))

    # -- main manifest (100-block), K ascending, n=4 for K<=128, n=2 for K>=192 --
    seq = 100
    for K in LANE_A_KS_FULL:
        est = f_cost(K)
        ceiling = max(1.0, round(est * 2.0, 1))
        for seed in (0, 1, 2, 3):
            jid = f"{seq:03d}_laneA_main_K{K}_s{seed}"
            cmd = (
                f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K {K} --seed {seed} "
                f"--steps {STEPS_MAIN} --outdir {EARLYLN_OUTDIR} "
                f"--ceiling-gpuh {ceiling} --stop-file {EARLYLN_OUTDIR}/STOP"
            )
            vcheck = (
                f"{PY} -c \""
                f"import json; d=json.load(open('{EARLYLN_OUTDIR}/earlyln_K{K}_s{seed}.json')); "
                f"assert d.get('status')=='COMPLETED'; "
                f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
                f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True\""
            )
            jobs.append(dict(
                id=jid, lane="A",
                hypothesis=(f"Does the earlyln recipe (S8.9's parameter-free inter-hop "
                            f"LayerNorm, train-only, eval always pure-matmul-exact) let "
                            f"K={K} (d={2*K}, h=64) converge and compose exactly at far "
                            f"depth, extending S11's own K=14/15/16/24 result up the K "
                            f"axis, per the task's K-ladder charter?"),
                cmd=cmd, gpu_h_estimate=round(est, 3),
                output_dir=EARLYLN_OUTDIR, validity_check=vcheck,
                notes=("formula-extrapolated GPU-h estimate; --ceiling-gpuh is 2x the "
                       "estimate (floor 1.0h) as the real safety bound; resume-safe "
                       "(the script's own whole-cell skip-if-COMPLETED)."),
            ))
            seq += 1

    for K in LANE_A_KS_DISCLOSED:
        est = f_cost(K)
        ceiling = max(1.0, round(est * 2.0, 1))
        for seed in (0, 1):
            jid = f"{seq:03d}_laneA_main_K{K}_s{seed}"
            cmd = (
                f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K {K} --seed {seed} "
                f"--steps {STEPS_MAIN} --outdir {EARLYLN_OUTDIR} "
                f"--ceiling-gpuh {ceiling} --stop-file {EARLYLN_OUTDIR}/STOP"
            )
            vcheck = (
                f"{PY} -c \""
                f"import json; d=json.load(open('{EARLYLN_OUTDIR}/earlyln_K{K}_s{seed}.json')); "
                f"assert d.get('status')=='COMPLETED'; "
                f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
                f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True\""
            )
            jobs.append(dict(
                id=jid, lane="A",
                hypothesis=(f"Same K-ladder question as the n=4 cells above, at K={K} "
                            f"(d={2*K}, h=64) -- SUB-4-SEED, DISCLOSED-ONLY by design "
                            f"(cost/risk tail, the harness's own harvest() already scores "
                            f"a sub-4-seed rung as disclosed-only, never gate-eligible, "
                            f"never entering the pooled worst-of-K label)."),
                cmd=cmd, gpu_h_estimate=round(est, 3),
                output_dir=EARLYLN_OUTDIR, validity_check=vcheck,
                notes=("n=2 not n=4: cost/risk tail of a formula-extrapolated estimate; "
                       "Lane C deepens these to n=4 (seeds 2,3) once Lane A budget is "
                       "spent, matching the '>=4 seeds' ask without front-loading risk."),
            ))
            seq += 1

    return jobs


# ---------------------------------------------------------------------------
# LANE B -- fix-at-scale program extensions, reusing lm_pretrain_rd.py verbatim
# ---------------------------------------------------------------------------
ANCHOR_SEED = 20260703          # ANCHOR_INIT_SEED, matrix-thinking/FROZEN_BIAS_LM_DESIGN.md
LAMBDA = 0.58
CORPORA = ("openr1-mix-ext", "wikitext-mix-ext")


def _lmC_ckpt_name(corpus, dm, ds, L, seed, step):
    return f"lmC_{corpus}_dm{dm}_ds{ds}_L{L}_s{seed}_step{step}.pt"


def _fixscale_cell(name, corpus, dm, ds, L, steps, seed, arm, lam=LAMBDA,
                    batch=32, timeout=36000, extra_notes=""):
    ckpt_dir = f"{FIXSCALE_CKPT_ROOT}/{name}"
    out = f"{FIXSCALE_RESULTS_ROOT}/{name}.json"
    lam_flag = f" --frozen-bias-lambda {lam}" if arm != "off" else ""
    cmd = (
        f"mkdir -p {ckpt_dir} {FIXSCALE_RESULTS_ROOT} && "
        f"cd {DELTANET_RD_DIR} && {PY} lm_pretrain_rd.py "
        f"--corpus {corpus} --data-dir {DATA_DIR} "
        f"--d-model {dm} --d-state {ds} --n-layers {L} --seq-len 512 "
        f"--batch-size {batch} --steps {steps} --ckpt-every {max(1000, steps // 20)} "
        f"--seed {seed} --internal-timeout {timeout} "
        f"--frozen-bias-arm {arm}{lam_flag} "
        f"--ckpt-dir {ckpt_dir} --out {out}"
    )
    vcheck = (
        f"{PY} -c \""
        f"import json; d=json.load(open('{out}')); "
        f"assert d.get('complete') is True; "
        f"assert d.get('steps_completed', 0) >= {steps} - 1\""
    )
    return cmd, vcheck, ckpt_dir, out


def lane_b_jobs() -> list[dict]:
    jobs = []
    seq = 200

    # --- B1: fresh 1.31B arm_per_token training, rung-3 config, token-matched
    # to the USER-SIGNED-OFF Rev-2.2 budget (SCALE_TRANSFER_DESIGN.md S5.6:
    # 1.5B tokens/run @ batch=16/seq=512 = 183,105 steps).
    #
    # PRICING FIX (rescue op 2026-07-12, RESCUE_1P31B_TIMEOUT incident): the
    # prior per_step_s=0.7135 was measured on a 5/20-step SHORT two-point
    # calibration cell and was already disclosed as wrong once before
    # (SCALE_TRANSFER_DESIGN.md S5.11, harvest 2026-07-07: sustained rate
    # 1.416 s/step, a 1.985x miss, self-terminated the original rung-3 ladder
    # cells at 155k/183,105 steps) -- that lesson was never propagated into
    # this generator, so job 200 (generated 2026-07-12 05:27 UTC) repeated the
    # exact same failure with --internal-timeout=160000. Re-measured directly
    # off job 200's own checkpoint-file mtimes (three independent 10,000-step
    # segments: 1.3998 / 1.3969 / 1.4004 s/step), stable across both
    # near-idle and fully-8-GPU-loaded periods -- confirms this is an
    # INTRINSIC throughput characteristic of the 1.31B/batch-16 config, not
    # contention, not thermal/power throttling (nvidia-smi telemetry clean).
    # Priced at per_step_s=1.40 (rounds up from the max observed segment) ->
    # 183105*1.40/3600 = 71.21 GPU-h/run for the plain architecture; +2% for
    # the frozen-bias blend (S13's own <=1.2% measured overhead, rounded up,
    # unchanged methodology) -> 72.6 GPU-h. --internal-timeout carries a
    # >=30% margin over that 72.6h estimate (CLAUDE.md-mandated floor after a
    # pricing miss), not the old formula's ad hoc 1.6x-on-a-wrong-base.
    dm, ds, L = 2560, 128, 22
    steps_1p31b = 183_105
    per_step_s_1p31b = 1.40  # measured, see PRICING FIX note above
    timeout_1p31b = round(steps_1p31b * per_step_s_1p31b * 1.02 * 1.30)  # >=30% margin
    name = f"queue_1p31b_arm_per_token_openr1-mix-ext_s0"
    ckpt_dir = f"{TRACKC_CKPT_ROOT}/{name}"
    out = f"{TRACKC_RESULTS_ROOT}/{name}.json"
    cmd = (
        f"mkdir -p {ckpt_dir} {TRACKC_RESULTS_ROOT} && "
        f"cd {DELTANET_RD_DIR} && {PY} lm_pretrain_rd.py "
        f"--corpus openr1-mix-ext --data-dir {DATA_DIR} "
        f"--d-model {dm} --d-state {ds} --n-layers {L} --seq-len 512 "
        f"--batch-size 16 --steps {steps_1p31b} --ckpt-every 10000 "
        f"--seed 0 --internal-timeout {timeout_1p31b} "
        f"--frozen-bias-arm per_token --frozen-bias-lambda {LAMBDA} "
        f"--ckpt-dir {ckpt_dir} --out {out}"
    )
    vcheck = (
        f"{PY} -c \""
        f"import json; d=json.load(open('{out}')); "
        f"assert d.get('complete') is True; "
        f"assert d.get('steps_completed', 0) >= {steps_1p31b} - 1\""
    )
    jobs.append(dict(
        id=f"{seq:03d}_laneB_1p31b_arm_per_token_openr1_s0", lane="B",
        hypothesis=("Does the destabilizing per_token frozen-bias sign (persisting, "
                    "attenuating, at 14M/98M/392M per FROZEN_BIAS_LM_DESIGN.md S13.22) "
                    "continue to attenuate, hold, or reverse at 1.31B -- extending the "
                    "SAME measurement protocol one more point up the already-approved "
                    "Track-C scale ladder (SCALE_TRANSFER_DESIGN.md S5.6, USER-SIGNED-OFF "
                    "1.5B-token budget)? Long single-job runtime by design -- keeps a GPU "
                    "busy through the whole overnight window."),
        cmd=cmd, gpu_h_estimate=round(steps_1p31b * per_step_s_1p31b * 1.02 / 3600, 1),
        output_dir=TRACKC_RESULTS_ROOT, validity_check=vcheck,
        notes=(f"cost basis: MEASURED (not short-calibration-extrapolated) per_step_s="
               f"{per_step_s_1p31b} -- see PRICING FIX comment above the code that builds "
               f"this cell; supersedes the stale 0.7135 s/step constant that caused this "
               f"exact cell to self-terminate at ~62% budget on 2026-07-12 (job "
               f"200_laneB_1p31b_arm_per_token_openr1_s0) and caused the original rung-3 "
               f"ladder cells to self-terminate at 155k/183,105 (~84.7%) on 2026-07-07 "
               f"(SCALE_TRANSFER_DESIGN.md S5.11). "
               f"{steps_1p31b} steps x {per_step_s_1p31b}s / 3600 = "
               f"{steps_1p31b * per_step_s_1p31b / 3600:.2f} GPU-h, +2% for the frozen-bias "
               f"blend (S13's own <=1.2% measured overhead, rounded up) = "
               f"{steps_1p31b * per_step_s_1p31b * 1.02 / 3600:.1f}h. --internal-timeout="
               f"{timeout_1p31b}s ({timeout_1p31b/3600:.1f}h) is a >=30% margin over that "
               f"estimate (CLAUDE.md-mandated floor after a pricing miss). HIGH confidence "
               f"(measured rate, reproduced twice 5 days apart under different concurrent-"
               f"load conditions -- ruling out contention/throttling as the driver). "
               "Comparator: reuses "
               "the ALREADY-EXISTING Track-C 1.31B checkpoint (SCALE_TRANSFER_DESIGN.md "
               "S5.11, harvest run 2026-07-07) as the arm_off baseline -- no comparator "
               "cell needed. FOLLOW-ON NOT QUEUED (deliberately, to avoid a brittle "
               "checkpoint-path chain under time pressure): once this cell's final "
               "checkpoint lands, a fresh coordinator should run "
               f"`{PY} lm_attractor_probe_rd.py --checkpoints {ckpt_dir}/<final>.pt "
               "--data-dir " + DATA_DIR + " --out " + TRACKC_RESULTS_ROOT + "/probe.json` "
               "(~0.5 GPU-h, near-zero, mirrors the existing rung-3 probe cost) to close "
               "the attractor-robustness-at-1.31B reading Lane B's charter asked for. A "
               "full 1.31B qk-norm/gating 2x2 (matching the 14M attractor-robustness "
               "campaign) was NOT queued: those flags are only validated at the 14M "
               "architecture (lm_pretrain_rd.py:984) and a fresh 1.31B 2x2 would cost "
               "~4x this cell's budget on an unvalidated config -- out of scope for "
               "tonight without its own design/audit pass."),
    ))
    seq += 1

    # --- B2: 392M seed-extension (seeds 4-7, both arms, both corpora,
    # reduced-budget convention already used by the CLOSED S13.22 wave) ---
    dm, ds, L = 1536, 128, 16
    steps_392 = 20_000
    for seed in (4, 5, 6, 7):
        for arm in ("off", "per_token"):
            for corpus in CORPORA:
                name = f"fixscale_seedext_arm_{arm}_392m_{corpus}_s{seed}"
                cmd, vcheck, ckpt_dir, out = _fixscale_cell(
                    name, corpus, dm, ds, L, steps_392, seed, arm)
                jid = f"{seq:03d}_laneB_392m_seedext_{arm}_{corpus}_s{seed}"
                jobs.append(dict(
                    id=jid, lane="B",
                    hypothesis=(f"Tightens the n=3->n=7 CI on FROZEN_BIAS_LM_DESIGN.md "
                                f"S13.22's 392M-{corpus} reading (arm={arm}) -- the "
                                f"392M-openr1 cell was the wave's one NULL (CI includes "
                                f"zero, n=3); more seeds resolve whether that's a genuine "
                                f"null or an n=3 underpowered read, without re-opening the "
                                f"already-recorded verdict."),
                    cmd=cmd, gpu_h_estimate=4.671,
                    output_dir=FIXSCALE_RESULTS_ROOT, validity_check=vcheck,
                    notes=("cost basis: S13.22's own measured 392M rate (ref_per_step_s="
                           "0.836, per_cell_gpuh=4.671, fixscale_wave.py:117-118). HIGH "
                           "confidence (measured, not extrapolated). No compounding-resume "
                           "(matches the project's own disclosed fixscale-resume "
                           "precedent) -- a mid-cell crash restarts this cell from step 0."),
                ))
                seq += 1

    # --- B3: 98M seed-extension (seeds 3-6, both arms, both corpora,
    # full-budget convention already used by the CLOSED S13.22 wave) ---
    dm, ds, L = 768, 64, 12
    steps_98 = 67_547
    for seed in (3, 4, 5, 6):
        for arm in ("off", "per_token"):
            for corpus in CORPORA:
                name = f"fixscale_seedext_arm_{arm}_98m_{corpus}_s{seed}"
                cmd, vcheck, ckpt_dir, out = _fixscale_cell(
                    name, corpus, dm, ds, L, steps_98, seed, arm)
                jid = f"{seq:03d}_laneB_98m_seedext_{arm}_{corpus}_s{seed}"
                jobs.append(dict(
                    id=jid, lane="B",
                    hypothesis=(f"Tightens the n=3->n=7 CI on S13.22's 98M-{corpus} "
                                f"reading (arm={arm}, currently CI-excludes-zero, "
                                f"destabilizing) -- more seeds strengthen or weaken that "
                                f"reading without re-opening the recorded PARTIAL verdict."),
                    cmd=cmd, gpu_h_estimate=4.478,
                    output_dir=FIXSCALE_RESULTS_ROOT, validity_check=vcheck,
                    notes=("cost basis: S13.22's own measured 98M rate (ref_per_step_s="
                           "0.236, per_cell_gpuh=4.478, fixscale_wave.py:116). HIGH "
                           "confidence (measured, not extrapolated)."),
                ))
                seq += 1

    # --- B4: 392M FULL-TOKEN-MATCHED probe (seed 3 only, both arms, both
    # corpora) -- directly answers the disclosed open gap S13.11 item 8 flags
    # ("392M ran the reduced 20,000-step budget... token-confounded") by
    # running ONE seed at the SAME 67,547-step/1.108B-token budget 98M used.
    # At 392M's own measured rate (0.836 s/step) this is 67547*0.836/3600 =
    # 15.69 GPU-h, well above the _fixscale_cell default 36000s=10h timeout
    # -- explicitly overridden to 72000s=20.0h (audit fix, MAJOR M1; widened
    # from an initial 65000s=18.06h to a fuller ~4.3h/27% buffer against
    # contention-driven per-step slowdown on an unattended overnight run).
    for arm in ("off", "per_token"):
        for corpus in CORPORA:
            name = f"fixscale_fulltoken_arm_{arm}_392m_{corpus}_s3"
            cmd, vcheck, ckpt_dir, out = _fixscale_cell(
                name, corpus, dm := 1536, ds := 128, L := 16, steps_98, 3, arm,
                timeout=72000)
            jid = f"{seq:03d}_laneB_392m_fulltoken_{arm}_{corpus}_s3"
            jobs.append(dict(
                id=jid, lane="B",
                hypothesis=(f"S13.11 item 8's own disclosed gap: does the 392M-{corpus} "
                            f"reading (arm={arm}) change under a FULL, token-matched "
                            f"budget (67,547 steps, same 1.108B tokens as 98M's own full "
                            f"cells) instead of the reduced 20,000-step convention the "
                            f"CLOSED wave used? A single new seed, not a full n=3 -- "
                            f"purely descriptive/exploratory, does not reopen S13.22's "
                            f"verdict of record."),
                cmd=cmd, gpu_h_estimate=15.69,
                output_dir=FIXSCALE_RESULTS_ROOT, validity_check=vcheck,
                notes=("cost basis: 0.836 s/step (S13.22's own measured 392M rate) x "
                       "67,547 steps = 15.69 GPU-h. HIGH confidence on the rate; the "
                       "step count is a deliberate choice (98M's own full-budget step "
                       "count), not itself previously measured at 392M -- moderate "
                       "confidence overall, disclosed. --internal-timeout raised to "
                       "72000s (20.0h) above the 15.69h estimate (audit fix, MAJOR M1: "
                       "the _fixscale_cell default 36000s=10h guaranteed a timeout)."),
            ))
            seq += 1

    return jobs


# ---------------------------------------------------------------------------
# LANE C -- evergreen tail: deepen Lane A's own K-ladder cells (more seeds,
# same cells, same interface -- the house fallback for filling remaining
# budget honestly rather than inventing new science).
# ---------------------------------------------------------------------------
def lane_c_jobs() -> list[dict]:
    jobs = []
    seq = 300

    # Deepen K in {20,32,48,64,96,128} from n=4 (Lane A) to n=6 (seeds 4,5).
    for K in LANE_A_KS_FULL:
        est = f_cost(K)
        ceiling = max(1.0, round(est * 2.0, 1))
        for seed in (4, 5):
            jid = f"{seq:03d}_laneC_deepen_K{K}_s{seed}"
            cmd = (
                f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K {K} --seed {seed} "
                f"--steps {STEPS_MAIN} --outdir {EARLYLN_OUTDIR} "
                f"--ceiling-gpuh {ceiling} --stop-file {EARLYLN_OUTDIR}/STOP"
            )
            vcheck = (
                f"{PY} -c \""
                f"import json; d=json.load(open('{EARLYLN_OUTDIR}/earlyln_K{K}_s{seed}.json')); "
                f"assert d.get('status')=='COMPLETED'; "
                f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
                f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True\""
            )
            jobs.append(dict(
                id=jid, lane="C",
                hypothesis=(f"Seed-replication of Lane A's own K={K} cells (n=4->n=6): "
                            f"deepens the CONVERGED-ROBUST-rate / far-depth-median "
                            f"statistics on an already-registered cell, not new science "
                            f"-- the house fallback for filling remaining queue budget "
                            f"honestly."),
                cmd=cmd, gpu_h_estimate=round(est, 3),
                output_dir=EARLYLN_OUTDIR, validity_check=vcheck,
                notes="same formula-extrapolated estimate + ceiling convention as Lane A.",
            ))
            seq += 1

    # Bring K in {192,256} from n=2 (Lane A, disclosed-only) up to n=4 (adds
    # seeds 2,3) -- fully satisfies the task's '>=4 seeds' ask for these too,
    # just sequenced AFTER the cheaper cells rather than front-loaded.
    for K in LANE_A_KS_DISCLOSED:
        est = f_cost(K)
        ceiling = max(1.0, round(est * 2.0, 1))
        for seed in (2, 3):
            jid = f"{seq:03d}_laneC_deepen_K{K}_s{seed}"
            cmd = (
                f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K {K} --seed {seed} "
                f"--steps {STEPS_MAIN} --outdir {EARLYLN_OUTDIR} "
                f"--ceiling-gpuh {ceiling} --stop-file {EARLYLN_OUTDIR}/STOP"
            )
            vcheck = (
                f"{PY} -c \""
                f"import json; d=json.load(open('{EARLYLN_OUTDIR}/earlyln_K{K}_s{seed}.json')); "
                f"assert d.get('status')=='COMPLETED'; "
                f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
                f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True\""
            )
            jobs.append(dict(
                id=jid, lane="C",
                hypothesis=(f"Brings K={K} from n=2 (Lane A, disclosed-only) to n=4, "
                            f"fully satisfying the K-ladder's own '>=4 seeds' bar at the "
                            f"cost tail once cheaper cells have already run."),
                cmd=cmd, gpu_h_estimate=round(est, 3),
                output_dir=EARLYLN_OUTDIR, validity_check=vcheck,
                notes="same formula-extrapolated estimate + ceiling convention as Lane A.",
            ))
            seq += 1

    return jobs


# ---------------------------------------------------------------------------
# REGATE 2026-07-12 -- §11.2-driven re-gate of the K-scaling ladder. See
# matrix-thinking/queue/regate_2026-07-12.md for the full before/after
# ledger and audit record. Two purely-additive pieces (no existing job ID
# above is touched or renumbered):
#
#   (1) lane_a_budget2x_probe_jobs() -- the two probes §11.2 priced but did
#       NOT launch ("(i) K=16 budget/anneal probe ... 2x-steps/longer-anneal
#       variant at K=16 x 4 seeds ~=3.5 GPU-h ... (ii) [same] at K=24 ...
#       ~=4 GPU-h"). ln_alpha_at(step, total) anneals over `total // 2`
#       where `total` IS the --steps value passed on the CLI
#       (ncr_earlyln_scale.py:147-152) -- so a single --steps 160000 call
#       is simultaneously 2x budget AND 2x anneal length. No new CLI knob
#       needed; no model/training code touched.
#
#   (2) lane_b_seedext_increment_jobs() / lane_c_k20_increment_jobs() --
#       refill for the GPU-h removed by parking K>=24 Lane A/C cells
#       (§11.2: K=16 1/4 converged-partial, K=24 0/4 dead -- both
#       TRAINABILITY-STILL-LIMITED under the CURRENT flat-80K recipe, so
#       K>=24 MAIN cells at that recipe would burn GPU-h re-measuring a
#       now-known-dead result). Reuses _fixscale_cell() / the Lane C K=20
#       deepen loop verbatim, just the next seeds in sequence -- same
#       already-audited machinery, same validity checks, same cost model.
# ---------------------------------------------------------------------------
BUDGET2X_OUTDIR = f"{NCR_DIR}/results_earlyln_budget2x"   # separate from
        # EARLYLN_OUTDIR on purpose -- see the notes field below (resume-skip
        # collision if reused).
STEPS_BUDGET2X = 160_000   # 2x STEPS_MAIN=80_000; §11.2's own priced probe


def lane_a_budget2x_probe_jobs() -> list[dict]:
    jobs = []
    seq = 50
    # (K, measured per-cell GPU-h at 80K steps == §11.2's own per-K 4-seed
    # main-run total / 4: K16 1.705/4, K24 2.015/4 -- MEASURED, not
    # formula-extrapolated.)
    specs = [(16, 1.705 / 4), (24, 2.015 / 4)]
    for K, per_cell_80k in specs:
        est = round(per_cell_80k * 2.0, 4)                  # linear-in-steps
        ceiling = max(1.0, round(per_cell_80k * 4.0, 1))    # >=100% margin over est
        for seed in (0, 1, 2, 3):
            jid = f"{seq:03d}_laneA_budget2x_K{K}_s{seed}"
            cmd = (
                f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K {K} --seed {seed} "
                f"--steps {STEPS_BUDGET2X} --outdir {BUDGET2X_OUTDIR} "
                f"--ceiling-gpuh {ceiling} --stop-file {BUDGET2X_OUTDIR}/STOP"
            )
            vcheck = (
                f"{PY} -c \""
                f"import json; d=json.load(open('{BUDGET2X_OUTDIR}/earlyln_K{K}_s{seed}.json')); "
                f"assert d.get('status')=='COMPLETED'; "
                f"assert d.get('train',{{}}).get('step')=={STEPS_BUDGET2X}; "
                f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True\""
            )
            jobs.append(dict(
                id=jid, lane="A",
                hypothesis=(f"§11.2's priced next-rung question: is K={K}'s trainability "
                            f"wall (K16: 1/4 converged, recovery stuck at 0.60-0.97 with "
                            f"dirty writes; K24: 0/4, partial-formation profile, loss "
                            f"1.0->0.36, A_eff_rank->17.7/24) BUDGET-limited rather than a "
                            f"hard architecture ceiling? --steps {STEPS_BUDGET2X} doubles "
                            f"BOTH total budget AND anneal length in one call (anneal = "
                            f"first half of --steps) -- no new knob, no code change."),
                cmd=cmd, gpu_h_estimate=est,
                output_dir=BUDGET2X_OUTDIR, validity_check=vcheck,
                notes=(f"MEASURED cost basis (§11.2 K={K} 80K-step 4-seed total "
                       f"{per_cell_80k * 4:.3f} GPU-h / 4 = {per_cell_80k:.4f}/cell), "
                       f"linearly doubled for {STEPS_BUDGET2X} steps = {est} GPU-h/cell "
                       f"({est * 4:.2f} GPU-h/4 seeds, matches §11.2's own disclosed "
                       f"'~3.5'/'~4' GPU-h estimate). Separate outdir "
                       f"({BUDGET2X_OUTDIR}) is REQUIRED, not cosmetic: reusing "
                       f"EARLYLN_OUTDIR would hit the script's own whole-cell "
                       f"skip-if-COMPLETED check (ncr_earlyln_scale.py:217-222) against "
                       f"the ALREADY-COMPLETED 80K-step record at this exact (K,seed), "
                       f"returning it unrun -- the vcheck's step=={STEPS_BUDGET2X} "
                       f"assertion would then correctly route the job to failed/, "
                       f"silently burning the slot rather than measuring anything. "
                       f"Ceiling is 4x the 80K per-cell rate (>=100% margin over the "
                       f"2x-linear estimate)."),
            ))
            seq += 1
    return jobs


def lane_b_seedext_increment_jobs(start_seq: int = 400) -> list[dict]:
    jobs = []
    seq = start_seq
    dm392, ds392, L392 = 1536, 128, 16
    steps_392 = 20_000
    for seed in (8, 9, 10, 11):
        for arm in ("off", "per_token"):
            for corpus in CORPORA:
                name = f"fixscale_seedext_arm_{arm}_392m_{corpus}_s{seed}"
                cmd, vcheck, ckpt_dir, out = _fixscale_cell(
                    name, corpus, dm392, ds392, L392, steps_392, seed, arm)
                jid = f"{seq:03d}_laneB_392m_seedext_{arm}_{corpus}_s{seed}"
                jobs.append(dict(
                    id=jid, lane="B",
                    hypothesis=(f"Regate-2026-07-12 refill: further tightens the n=7->"
                                f"n=11 CI on FROZEN_BIAS_LM_DESIGN.md S13.22's "
                                f"392M-{corpus} reading (arm={arm}) -- identical cell/"
                                f"protocol to the original seedext block (seeds 4-7), "
                                f"next seeds in sequence, does not reopen the recorded "
                                f"verdict."),
                    cmd=cmd, gpu_h_estimate=4.671,
                    output_dir=FIXSCALE_RESULTS_ROOT, validity_check=vcheck,
                    notes=("identical cost basis to the original seedext block "
                           "(S13.22 ref_per_step_s=0.836, per_cell_gpuh=4.671)."),
                ))
                seq += 1

    dm98, ds98, L98 = 768, 64, 12
    steps_98 = 67_547
    for seed in (7, 8, 9, 10):
        for arm in ("off", "per_token"):
            for corpus in CORPORA:
                name = f"fixscale_seedext_arm_{arm}_98m_{corpus}_s{seed}"
                cmd, vcheck, ckpt_dir, out = _fixscale_cell(
                    name, corpus, dm98, ds98, L98, steps_98, seed, arm)
                jid = f"{seq:03d}_laneB_98m_seedext_{arm}_{corpus}_s{seed}"
                jobs.append(dict(
                    id=jid, lane="B",
                    hypothesis=(f"Regate-2026-07-12 refill: further tightens the n=7->"
                                f"n=11 CI on S13.22's 98M-{corpus} reading (arm={arm}) -- "
                                f"identical cell/protocol to the original seedext block "
                                f"(seeds 3-6), next seeds in sequence."),
                    cmd=cmd, gpu_h_estimate=4.478,
                    output_dir=FIXSCALE_RESULTS_ROOT, validity_check=vcheck,
                    notes=("identical cost basis to the original seedext block "
                           "(S13.22 ref_per_step_s=0.236, per_cell_gpuh=4.478)."),
                ))
                seq += 1
    return jobs


def lane_c_k20_increment_jobs(start_seq: int = 450) -> list[dict]:
    jobs = []
    seq = start_seq
    K = 20
    est = f_cost(K)
    ceiling = max(1.0, round(est * 2.0, 1))
    for seed in (6, 7):
        jid = f"{seq:03d}_laneC_deepen_K{K}_s{seed}"
        cmd = (
            f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K {K} --seed {seed} "
            f"--steps {STEPS_MAIN} --outdir {EARLYLN_OUTDIR} "
            f"--ceiling-gpuh {ceiling} --stop-file {EARLYLN_OUTDIR}/STOP"
        )
        vcheck = (
            f"{PY} -c \""
            f"import json; d=json.load(open('{EARLYLN_OUTDIR}/earlyln_K{K}_s{seed}.json')); "
            f"assert d.get('status')=='COMPLETED'; "
            f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
            f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True\""
        )
        jobs.append(dict(
            id=jid, lane="C",
            hypothesis=(f"Regate-2026-07-12 refill: K=20 is the one rung §11.2 leaves "
                        f"genuinely open (the K-scaling ladder was pinned at "
                        f"K in {{14,15,16,24}}; K=20 has never been measured under "
                        f"earlyln). Deepens n=6->n=8 on the SAME already-registered "
                        f"cell -- house fallback, not new science."),
            cmd=cmd, gpu_h_estimate=round(est, 3),
            output_dir=EARLYLN_OUTDIR, validity_check=vcheck,
            notes="same formula-extrapolated estimate + ceiling convention as Lane A/C.",
        ))
        seq += 1
    return jobs


def regate_20260712_jobs() -> list[dict]:
    return (lane_a_budget2x_probe_jobs() + lane_b_seedext_increment_jobs()
            + lane_c_k20_increment_jobs())


# ---------------------------------------------------------------------------
# NCR NEXT-LEVER WAVE, 2026-07-12b -- Q1 only (K=16 4x-budget cell). See
# matrix-thinking/NCR_NEXT_LEVER_DESIGN.md (commit a8e848d) S1.3/S1.5/S1.7
# for the full pre-registration; matrix-thinking/queue/regate_2026-07-12.md
# S6 for this build round's own record.
#
# Q2 Probe A (--d-override) and Probe B (--anneal-frac) were DELIBERATELY NOT
# built in THIS round: both required a new CLI flag that did not exist in
# ncr_earlyln_scale.py at the time, verified byte-identical between this repo
# mirror and the live box script (2026-07-12). The design's own S5 already
# labelled these "Build prerequisites... for the eventual builder -- not
# built here" -- this was the design's disclosed scope boundary firing as
# intended, not a defect. Both flags landed in a follow-on build round
# (additive-only, independent opus audit CLEARED) -- see
# ncr_next_lever_probe_ab_jobs() below for the jobs those flags unblocked,
# and regate_2026-07-12.md S7 for that build+audit record.
# ---------------------------------------------------------------------------
BUDGET4X_OUTDIR = f"{NCR_DIR}/results_earlyln_budget4x"   # separate outdir,
        # load-bearing not cosmetic -- identical reasoning to
        # lane_a_budget2x_probe_jobs() above: the script's own whole-cell
        # skip-if-COMPLETED check keys on (K, seed) only, not --steps, so
        # reusing results_earlyln_scale/ or results_earlyln_budget2x/ would
        # silently return the wrong-step-count COMPLETED record unrun.
STEPS_BUDGET4X = 320_000   # 4x STEPS_MAIN=80_000; NCR_NEXT_LEVER_DESIGN.md S1.3


def ncr_next_lever_q1_jobs() -> list[dict]:
    """Q1 -- K=16 4x-budget cell, n=4 seeds. NCR_NEXT_LEVER_DESIGN.md
    S1.3/S1.5/S1.7. Cost basis MEASURED (not extrapolated): S1.4's own two
    real rate points at this exact K=16/d=32 shape (1x 0.4263/80,000 =
    5.33e-6 GPU-h/step; 2x 0.8248/160,000 = 5.16e-6 GPU-h/step -- 3.4%
    agreement, i.e. cost is measured linear-in-steps here). Linear-in-steps
    extrapolation from the 2x rate: 0.8248*2 = 1.6496 GPU-h/cell, matching
    S1.5's own pinned "4 x 1.6496 ~= 6.60 GPU-h" for the 4-seed cohort.
    """
    jobs = []
    seq = 60
    per_cell_nominal = 1.6496   # S1.5, 2x measured rate x2 (linear-in-steps)
    ceiling = 3.5                # S1.5's own pinned per-cell breaker (~=2.1x nominal)
    for seed in (0, 1, 2, 3):
        jid = f"{seq:03d}_laneA_budget4x_K16_s{seed}"
        cmd = (
            f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K 16 --seed {seed} "
            f"--steps {STEPS_BUDGET4X} --outdir {BUDGET4X_OUTDIR} "
            f"--ceiling-gpuh {ceiling} --stop-file {BUDGET4X_OUTDIR}/STOP"
        )
        vcheck = (
            f"{PY} -c \""
            f"import json; d=json.load(open('{BUDGET4X_OUTDIR}/earlyln_K16_s{seed}.json')); "
            f"assert d.get('status')=='COMPLETED'; "
            f"assert d.get('train',{{}}).get('step')=={STEPS_BUDGET4X}; "
            f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True\""
        )
        jobs.append(dict(
            id=jid, lane="A",
            hypothesis=(
                "Does operator residual delta at K=16 continue the ~2.95x/doubling "
                "budget power law measured 1x->2x at 4x (320,000 steps)? PRIMARY "
                "readout is the 3-point log-log power-law fit (delta vs budget) and "
                "the failure-front consistency check (predicted to hold at front=29, "
                "not reach 61, per the arccos delta*(h)=0.451/h bound) -- NOT "
                "rec@h*=125, which the design's own depth-corrected crossing target "
                "(delta*=0.0036, not the h*=61-specific 0.0086) predicts as a "
                "foreordained 0.0 at this budget (NCR_NEXT_LEVER_DESIGN.md S1.1/S1.7)."
            ),
            cmd=cmd, gpu_h_estimate=per_cell_nominal,
            output_dir=BUDGET4X_OUTDIR, validity_check=vcheck,
            notes=(
                "MEASURED cost basis (NOT formula-extrapolated): S1.4's two real "
                "per-step rates at this exact K=16/d=32 shape (1x 0.4263/80,000 = "
                "5.33e-6 GPU-h/step; 2x 0.8248/160,000 = 5.16e-6 GPU-h/step, 3.4% "
                "agreement -- mildly sub-linear if anything). Linear-in-steps "
                "extrapolation from the 2x rate: 0.8248*2 = 1.6496 GPU-h/cell "
                "(matches S1.5's pinned '4 x 1.6496 ~= 6.60 GPU-h' for the 4-seed "
                "cohort). Ceiling 3.5 GPU-h/cell (~=2.1x nominal, S1.5's own pinned "
                "breaker, well above the >=25% margin floor). Separate outdir "
                f"({BUDGET4X_OUTDIR}) is REQUIRED: the script's whole-cell "
                "skip-if-COMPLETED check keys on (K,seed) only, not --steps -- reusing "
                "results_earlyln_scale/ or results_earlyln_budget2x/ would return the "
                "wrong-step-count COMPLETED record unrun (identical hazard to the "
                "2026-07-12 budget2x deploy, regate_2026-07-12.md). No new CLI knob "
                "needed (--steps/--outdir/--ceiling-gpuh/--stop-file all pre-existing, "
                "verified against the live argparse table on box 2026-07-12)."
            ),
        ))
        seq += 1
    return jobs


# ---------------------------------------------------------------------------
# NCR NEXT-LEVER WAVE, 2026-07-12c -- Q2 Probe A (--d-override) + Probe B
# (--anneal-frac). Both flags landed in ncr_earlyln_scale.py this build
# round (additive-only; independent opus audit CLEARED -- see
# matrix-thinking/queue/regate_2026-07-12.md S7 for the full build+audit
# record). NCR_NEXT_LEVER_DESIGN.md S2.1/S2.2/S5 is the pre-registration;
# job IDs follow S5's own proposed numbering band exactly: 066-073 Probe A
# (8 jobs), 074-077 Probe B-16 (4 jobs), 078-081 Probe B-24 (4 jobs). IDs
# 064-065 (the conditional Q1 8x recon) are DELIBERATELY skipped here --
# gated on Q1's own harvest + stopping rule (S1.7/S4), unrelated to this
# build.
# ---------------------------------------------------------------------------
DRATIO_OUTDIR = f"{NCR_DIR}/results_earlyln_dratio"            # Probe A --
        # NEW, separate from results_earlyln_scale/: the existing d=2K
        # record already occupies the identical earlyln_K{K}_s{seed}
        # filename there (same skip-if-COMPLETED collision hazard as every
        # separate-outdir probe this program has deployed).
ANNEALSHAPE_OUTDIR = f"{NCR_DIR}/results_earlyln_annealshape"  # Probe B --
        # NEW, separate outdir: same (K,seed,steps) as the frac=0.5
        # baseline cells in results_earlyln_scale/, identical hazard.


def ncr_next_lever_probe_a_jobs() -> list[dict]:
    """Q2 Probe A -- tight-spare d=K+1 at K=16 (d=17) and K=24 (d=25), n=4
    each, 80K steps (1x -- isolates d alone, never bundled with the budget
    axis). NCR_NEXT_LEVER_DESIGN.md S2.1/S5. A three-story-DISCRIMINATING
    binary existence test, not a confirmation run: does the K=15 tight-spare
    convention (SCALES 4/4, d-K=1), unchanged, also work at K=16/24? S1
    (registry S9.2's Mechanism-1 + dead-rate floor) predicts tight spare is
    WORSE (FAIL); S2 (the S0 convention-confound reading) predicts CONFIRM;
    the registry's own absolute-K-cliff reading predicts K=16 fails like
    K=16@d=32 did. Cost basis MEASURED at the SAME-K larger-d rate
    (conservative -- d enters the param count only via the 4dh term and d
    SHRINKS at the override, so true cost is <= this estimate): K16 0.4263
    GPU-h/cell (S11.2's own 1.705/4), K24 0.5038 GPU-h/cell (S11.2's own
    2.015/4). Per-cell breaker 1.0 GPU-h (S2.1's own pinned figure)."""
    jobs = []
    seq = 66
    ceiling = 1.0   # S2.1's own pinned per-cell breaker
    specs = [(16, 17, 1.705 / 4), (24, 25, 2.015 / 4)]  # (K, d_override, per_cell_nominal)
    for K, d_ov, per_cell in specs:
        for seed in (0, 1, 2, 3):
            jid = f"{seq:03d}_laneA_dratio_K{K}_d{d_ov}_s{seed}"
            cmd = (
                f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K {K} --seed {seed} "
                f"--steps {STEPS_MAIN} --outdir {DRATIO_OUTDIR} --d-override {d_ov} "
                f"--ceiling-gpuh {ceiling} --stop-file {DRATIO_OUTDIR}/STOP"
            )
            vcheck = (
                f"{PY} -c \""
                f"import json; d=json.load(open('{DRATIO_OUTDIR}/earlyln_K{K}_s{seed}.json')); "
                f"assert d.get('status')=='COMPLETED'; "
                f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
                f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True; "
                f"assert d.get('d')=={d_ov}\""
            )
            jobs.append(dict(
                id=jid, lane="A",
                hypothesis=(
                    f"Q2 Probe A (K={K}@d={d_ov}, the tight-spare d=K+1 convention K=15 "
                    f"already demonstrated working): does K={K} also converge/compose at "
                    f"this convention, discriminating three pinned, mutually-inconsistent "
                    f"stories about the K15->K16 wall -- S1 (registry S9.2's Mechanism-1 + "
                    f"dead-rate floor: tight spare is WORSE, predicts FAIL), S2 (the S0 "
                    f"convention-confound reading: predicts CONFIRM), and the registry's "
                    f"own absolute-K-cliff reading (predicts FAIL regardless of spare "
                    f"convention)? A single binary readout kills at least one story "
                    f"(NCR_NEXT_LEVER_DESIGN.md S2.1)."
                ),
                cmd=cmd, gpu_h_estimate=round(per_cell, 4),
                output_dir=DRATIO_OUTDIR, validity_check=vcheck,
                notes=(
                    f"MEASURED cost basis at the SAME-K larger-d rate (S11.2's own K={K} "
                    f"80K 4-seed total / 4 -- conservative, since d enters the param count "
                    f"only via the 4dh term and d SHRINKS at the override, "
                    "NCR_NEXT_LEVER_DESIGN.md S2.1). Ceiling 1.0 GPU-h/cell (S2.1's own "
                    f"pinned breaker). Separate outdir ({DRATIO_OUTDIR}) is REQUIRED: the "
                    f"existing d={2 * K} record already occupies the identical "
                    f"earlyln_K{K}_s{{seed}} filename in results_earlyln_scale/ -- same "
                    "skip-if-COMPLETED collision reasoning as every prior separate-outdir "
                    "probe this program has deployed. Validity check additionally asserts "
                    "the record's own 'd' field equals the override, per S5's own pinned "
                    "validity-check addendum."
                ),
            ))
            seq += 1
    return jobs


def ncr_next_lever_probe_b16_jobs() -> list[dict]:
    """Q2 Probe B-16 -- K=16, d=32 (mapping default, no override), 80K
    steps, anneal_frac=0.75 (vs the implicit 0.5 baseline), n=4. THE
    disentangling cell for Q1's budget/anneal confound (S1.2): does longer
    anneal alone reproduce a material part of the 1x->2x delta drop?
    NCR_NEXT_LEVER_DESIGN.md S2.2/S5."""
    jobs = []
    seq = 74
    ceiling = 1.0   # S2.2's own pinned per-cell breaker
    per_cell = 1.705 / 4   # S11.2's own K=16 80K 4-seed rate (same K/d/steps as the baseline)
    for seed in (0, 1, 2, 3):
        jid = f"{seq:03d}_laneA_annealshape_K16_s{seed}"
        cmd = (
            f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K 16 --seed {seed} "
            f"--steps {STEPS_MAIN} --outdir {ANNEALSHAPE_OUTDIR} --anneal-frac 0.75 "
            f"--ceiling-gpuh {ceiling} --stop-file {ANNEALSHAPE_OUTDIR}/STOP"
        )
        vcheck = (
            f"{PY} -c \""
            f"import json; d=json.load(open('{ANNEALSHAPE_OUTDIR}/earlyln_K16_s{seed}.json')); "
            f"assert d.get('status')=='COMPLETED'; "
            f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
            f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True; "
            f"assert d.get('anneal_frac')==0.75\""
        )
        jobs.append(dict(
            id=jid, lane="A",
            hypothesis=(
                "Q2 Probe B-16 (K=16, anneal_frac=0.75 vs the implicit 0.5 baseline, "
                "fixed 80K-step budget): does more supported-formation steps (60K vs "
                "40K before alpha->0) improve K=16's write quality (baseline mean "
                "delta=0.1040), and is Q1's 1x->2x delta improvement substantially an "
                "anneal-length effect rather than a budget effect? Both directions are "
                "mechanistically live (more crutch-support vs fewer post-crutch "
                "consolidation steps, 20K vs 40K) -- either outcome is informative "
                "(NCR_NEXT_LEVER_DESIGN.md S1.2/S2.2, Q2(e))."
            ),
            cmd=cmd, gpu_h_estimate=round(per_cell, 4),
            output_dir=ANNEALSHAPE_OUTDIR, validity_check=vcheck,
            notes=(
                "MEASURED cost basis (S11.2's own K=16 80K 4-seed rate -- anneal_frac "
                "changes ONLY when the LN blend reaches 0, not the per-step LN-blend op "
                "count, so the SAME rate applies, NCR_NEXT_LEVER_DESIGN.md S2.2 -- an "
                "analytic prediction, not a measurement, hence the launch-order "
                "mitigation: the first Probe-B job released gets a T+10min live health "
                "check before the rest deploy. Ceiling 1.0 GPU-h/cell. Separate outdir "
                f"({ANNEALSHAPE_OUTDIR}) is REQUIRED: same (K,seed,steps) as the "
                "frac=0.5 baseline in results_earlyln_scale/ -- identical "
                "skip-if-COMPLETED collision hazard as every other probe this program "
                "has deployed. Validity check additionally asserts the record carries "
                "anneal_frac==0.75, per S5's own pinned validity-check addendum."
            ),
        ))
        seq += 1
    return jobs


def ncr_next_lever_probe_b24_jobs() -> list[dict]:
    """Q2 Probe B-24 -- K=24, d=48 (mapping default, no override), 80K
    steps, anneal_frac=0.75, n=4. The Q2 formation-failure schedule-axis
    arm. NCR_NEXT_LEVER_DESIGN.md S2.2/S5."""
    jobs = []
    seq = 78
    ceiling = 1.0
    per_cell = 2.015 / 4   # S11.2's own K=24 80K 4-seed rate
    for seed in (0, 1, 2, 3):
        jid = f"{seq:03d}_laneA_annealshape_K24_s{seed}"
        cmd = (
            f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K 24 --seed {seed} "
            f"--steps {STEPS_MAIN} --outdir {ANNEALSHAPE_OUTDIR} --anneal-frac 0.75 "
            f"--ceiling-gpuh {ceiling} --stop-file {ANNEALSHAPE_OUTDIR}/STOP"
        )
        vcheck = (
            f"{PY} -c \""
            f"import json; d=json.load(open('{ANNEALSHAPE_OUTDIR}/earlyln_K24_s{seed}.json')); "
            f"assert d.get('status')=='COMPLETED'; "
            f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
            f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True; "
            f"assert d.get('anneal_frac')==0.75\""
        )
        jobs.append(dict(
            id=jid, lane="A",
            hypothesis=(
                "Q2 Probe B-24 (K=24, anneal_frac=0.75 vs the implicit 0.5 baseline, "
                "fixed 80K-step budget): is K=24's 0/4 TRAINABILITY-DEAD formation "
                "failure bottlenecked by the LN-crutch withdrawal SCHEDULE rather than "
                "total compute (S11.2: budget alone is flat-to-WORSE at K=24, S0 finding "
                "2)? Both directions are mechanistically live -- more supported-formation "
                "steps could help formation OR the shorter post-crutch consolidation "
                "window could make a withdrawal-cliff failure worse "
                "(NCR_NEXT_LEVER_DESIGN.md S2.2, Q2(e))."
            ),
            cmd=cmd, gpu_h_estimate=round(per_cell, 4),
            output_dir=ANNEALSHAPE_OUTDIR, validity_check=vcheck,
            notes=(
                "MEASURED cost basis (S11.2's own K=24 80K 4-seed rate; same "
                "anneal_frac-is-cost-neutral analytic argument as Probe B-16). Ceiling "
                f"1.0 GPU-h/cell. Separate outdir ({ANNEALSHAPE_OUTDIR}) is REQUIRED "
                "(same collision hazard as Probe B-16). Validity check additionally "
                "asserts anneal_frac==0.75."
            ),
        ))
        seq += 1
    return jobs


def ncr_next_lever_probe_ab_jobs() -> list[dict]:
    """Q2 Probe A + Probe B-16 + Probe B-24, generated together as this
    build round's single additive entry point (job IDs 066-081, directly
    behind Q1's 060-063, per NCR_NEXT_LEVER_DESIGN.md S5's own proposed
    numbering band). 064-065 (the conditional Q1 8x recon) are
    DELIBERATELY skipped -- gated on Q1's own harvest + stopping rule
    (S1.7/S4), unrelated to this build."""
    return (ncr_next_lever_probe_a_jobs() + ncr_next_lever_probe_b16_jobs()
            + ncr_next_lever_probe_b24_jobs())


# ---------------------------------------------------------------------------
# NCR MAPPING-LAW WAVE, 2026-07-12d -- WAVE-1 ONLY (K=32 d(K) grid, K=48
# rate-probe citation-cell, Q2 K24@d25 seed extension to n=12).
# NCR_MAPPING_LAW_DESIGN.md (commit d90abff) S1.2/S1.4/S1.5/S1.6/S2.1/S5 is
# the pre-registration; matrix-thinking/queue/regate_2026-07-12.md S8 for
# this build round's own record. WAVE-1b (K=48's own d(K) grid; the design's
# own S5 proposes IDs 513-524 for it, RESERVED, NOT generated here) is
# DELIBERATELY NOT built by this round -- it auto-fires only after (a)
# K=32's own S1.6 verdict is recorded as not CLOSED-AT-THIS-K, and (b) the
# K=48 rate probe's budget-fit check clears (per this build's own finding
# below, it does NOT clear as measured -- disclosed, not resolved by this
# build). The 108-111 (parked_k24plus, K=48 2K-reference) unpark stays
# gated identically; not moved by this round.
#
# Job-ID BAND, deliberately DIFFERENT from the design's own S5-proposed
# 500-536: the design chose 500-536 purely for COLLISION avoidance ("new
# band... verified collision-free against the live box"; S5's own text
# gives no priority rationale). This build's own dispatch separately
# requires deploy priority "at the FRONT (below current lowest pending)" --
# the live box's lowest currently-pending prefix at deploy time was 215
# (215-451 pending: ~356 GPU-h of already-queued Lane-B seed-extension
# work, ~44.5 GPU-h wall-clock across 8 GPUs). Since this queue's ONLY
# priority mechanism IS the filename/id numeric prefix (ascending =
# claimed first, generate_jobs.py's own docstring; QUEUE_README.md
# "filename = priority order"), landing at 500-536 would place WAVE-1
# BEHIND that entire Lane-B backlog, not "at the front" -- the opposite of
# what was asked. Mirrors this exact program's own precedent
# (lane_a_budget2x_probe_jobs, IDs 050-057: "front of pending, priority
# below the current lowest remaining Lane-A pending prefix at the time",
# regate_2026-07-12.md S2). Band 008-028 (21 jobs) is independently
# verified collision-free across pending/claimed/completed/failed/
# parked_k24plus on the live box this build round (used: 000-007, 050-057,
# 060-063, 066-081 [064-065 correctly excluded -- reserved by
# NCR_NEXT_LEVER_DESIGN.md's own conditional 8x recon], 100-127, 200-214;
# 008-049 is the largest genuinely free gap below the 215 front). The
# design's own S5 job-ID citations (500/501-512/513-524/525-532/533-536)
# remain valid as SEMANTIC labels for cross-referencing this design
# document's own tables; they are not the on-box filenames this build
# deploys. A future WAVE-1b build round should make its own front-of-queue
# ID choice at ITS OWN deploy time, not assume 513-524 is still the front.
# ---------------------------------------------------------------------------
DRATIO125_OUTDIR = f"{NCR_DIR}/results_earlyln_dratio125"   # Q1 K32 1.25K arm --
        # NEW, separate from DRATIO_OUTDIR/results_earlyln_scale/: within one
        # outdir the skip-if-COMPLETED check keys on (K,seed) only, so a
        # SECOND non-default d at the same (K,seed) in an outdir already
        # holding a different d's COMPLETED record would silently return the
        # WRONG-d record unrun (NCR_MAPPING_LAW_DESIGN.md S1.2's own outdir-
        # collision discipline, identical reasoning to DRATIO_OUTDIR itself).
DRATIO150_OUTDIR = f"{NCR_DIR}/results_earlyln_dratio150"   # Q1 K32 1.5K arm -- same reasoning.


def ncr_mapping_law_k48_probe_citation_job() -> list[dict]:
    """S1.4's mandatory K=48 rate probe (K=48, seed 0, d=96 mapping default,
    --steps 500). NOT a fresh measurement -- an IDENTICAL cell (same K, seed,
    steps, outdir, hence same on-disk filename earlyln_K48_s0.json) already
    ran to completion under job 002_laneA_probe_K48_s0 (lane_a_jobs()'s own
    000-block, predating this wave -- the design's own S0 evidence table did
    not cross-check this). Verified this build round by reading the real
    completed record on box: status=COMPLETED, train.step=500,
    gpu_h=0.014541613790724012, train.elapsed_s=10.9191 (the top-level
    elapsed_s is 52.35s -- the full cell including post-train eval/deep-probe
    overhead; train.elapsed_s is the sub-field measuring training alone). Per
    run_earlyln_cell's own skip-if-COMPLETED resume check (keyed on (K,seed)
    only, ncr_earlyln_scale.py:238-245), deploying this job with this
    BYTE-IDENTICAL cmd is safe and idempotent: it returns the SAME existing
    record instantly (zero incremental GPU-h), giving this wave its own
    canonical completed/ record for the design's own semantic label (S5:
    '500: K48 rate probe... generate + deploy now' -- deployed here at ID
    008, a different front-of-queue band; see the module comment above)
    rather than silently citing an untracked-by-this-wave job number.

    S1.4's re-derivation trigger, applied to the REAL measured number (not
    formula-extrapolated): naive x160 step-scale = 0.014541613790724012 *
    160 = 2.3267 GPU-h vs the design's own 0.55 GPU-h planning value ->
    ratio 4.23, far above the 1.25 disagreement bar -- THE TRIGGER FIRES.
    Exactly the failure mode S1.4's own prose predicted (fixed post-train
    eval/deep-probe overhead -- here ~41s against only 10.9s of actual
    500-step training -- dominates a 500-step cell's wall-clock and inverts
    at 80,000 steps, so the naive linear scale-up overstates true K=48
    main-cell cost). Disclosed here, NOT resolved: per S1.4, WAVE-1b's K=48
    arms' --ceiling-gpuh and total GPU-h MUST be re-derived from a real
    80K-equivalent K=48 rate before any K=48 main cell (WAVE-1b's new-build
    513-524 or the unparked 108-111 2K-reference) launches -- a hard
    pre-condition on WAVE-1b's own future build/deploy, unrelated to
    WAVE-1's own K=32 cells (no WAVE-1 cell is a K=48 main cell)."""
    jid = "008_laneA_mappinglaw_probe_K48_s0"
    cmd = (
        f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K 48 --seed 0 "
        f"--steps {STEPS_PROBE} --outdir {EARLYLN_PROBE_OUTDIR} "
        f"--ceiling-gpuh 0.5"
    )
    vcheck = (
        f"{PY} -c \""
        f"import json; d=json.load(open('{EARLYLN_PROBE_OUTDIR}/earlyln_K48_s0.json')); "
        f"assert d.get('status')=='COMPLETED'; "
        f"assert d.get('train',{{}}).get('step')=={STEPS_PROBE}; "
        f"assert 'eval' in d and 'deep_probe' in d\""
    )
    return [dict(
        id=jid, lane="A",
        hypothesis=("S1.4's mandatory pre-K48-main-cell rate probe (K=48, d=96 mapping "
                    "default, 500 steps) -- de-risks WAVE-1b's K=48 planning rate BEFORE "
                    "any 80,000-step K=48 main cell commits budget, mirroring S9.9's own "
                    "Phase-0a discipline. An identical cell already completed as job "
                    "002_laneA_probe_K48_s0 -- this job's own cmd is byte-identical and "
                    "resolves via the script's own skip-if-COMPLETED path (verified this "
                    "build round: real gpu_h=0.014541613790724012 at 500 steps), not a "
                    "fresh measurement. Not itself a trainability readout."),
        cmd=cmd, gpu_h_estimate=0.0145,
        output_dir=EARLYLN_PROBE_OUTDIR, validity_check=vcheck,
        notes=("MEASURED, not estimated -- cites the pre-existing completed record "
               "(002_laneA_probe_K48_s0), byte-identical cmd, expected to skip-if-"
               "COMPLETED at zero incremental GPU-h. Re-derivation trigger FIRES "
               "(naive x160 scale 2.3267 GPU-h vs 0.55 planning value, ratio 4.23 > "
               "1.25) -- flagged for the WAVE-1b gate, not resolved by WAVE-1."),
    )]


def t2a3_falconmamba_calibration_job() -> list[dict]:
    """PARAM_AXIS_SCALING_DESIGN.md sec 13.5's deferred T2a-3 (SSM causal
    calibration, witness C1=falcon-mamba-7b) -- queued by the T2a attempt-2
    execution agent (2026-07-13) rather than run inline, per that section's
    own instruction ("leave C1/T2a-3 as a separately scheduled cell") and
    this task's explicit brief.

    NOT a C1-only invocation: t2a_reference_driver_v2_rd.py's `mode_gate`
    hard-REFUSES any --witness/--corpus set that is not EXACTLY
    REQUIRED_WITNESSES=(W1_rwkv7, W2_gpt2large, C1_falconmamba) x
    REQUIRED_CORPORA=(openr1-mix-ext, wikitext-mix-ext) (D5 round-3
    SERIOUS-1's anti-subsetting refusal, hardened over 6 adversarial
    rounds -- "a subset run simply cannot produce a verdict, so it is
    refused rather than silently narrowed"). This is a genuine, disclosed
    gap between sec 13.5's stated intent (W1+W2 inline, C1 deferred) and
    what the pinned CLI actually supports: there is no flag that runs C1
    alone. Decoupling T2a-2/T1c's computation from the full witness loop,
    or relaxing the refusal to admit the specific {W1,W2} subset sec 13.5
    pre-authorizes, would require a driver code change -- a build step
    this execution agent's charter does not cover (CLAUDE.md: the
    implementer does not review/run their own work; a fresh audit would be
    needed). So this job runs the SAME full, unmodified, required-witness
    `--gate` invocation the live read used -- it REDUNDANTLY re-computes
    W1/W2/T2a-2/T1c (cheap, harmless, no laundering) in addition to finally
    reaching C1's T2a-3 legs. This is the only way to close T2a-3 without
    a code change.

    Cost: unknown with real precision. The design's own sec 11.3 estimate
    ("well under a minute of H100 time" per witness/corpus) assumed a
    fused Mamba kernel path; falcon-mamba-7b falls back to the SEQUENTIAL,
    non-fused implementation on this box (no `kernels`/`mamba-ssm`/
    `causal-conv1d` installed -- sec 13.5(c) explicitly declines to install
    them: a compiled dependency in a venv shared by 7 other live training
    jobs is too risky for a witness that cannot change the T2a-1 verdict
    either way). The attempt-1 read's C1/openr1-mix-ext cell alone ran
    3h49m of continuous GPU time WITHOUT completing before that session
    stopped watching (sec 12.4) -- so the per-corpus cost floor is already
    known to exceed the brief's own "~4 GPU-h" figure for a single cell,
    and wikitext-mix-ext's larger split (317,474 vs 230,074 train docs)
    is expected to cost at least as much. gpu_h_estimate below is a
    disclosed, generous guess (W1+W2 fast + 2x C1-corpus cost at
    >=4h/corpus), not a measurement -- flagged rather than presented as
    calibrated.

    Numbered 990 (see the priority-convention addendum above): claimed
    only after every currently-pending Lane A/B/C job (000-431) is
    exhausted, so it never preempts the 392M/1.31B rung cells this
    program's compute depends on."""
    out_dir = f"{DELTANET_RD_DIR}/results/param_axis_t2a3_queued"
    out = f"{out_dir}/t2a_gate_result.json"
    cmd = (
        f"mkdir -p {out_dir} && cd {DELTANET_RD_DIR} && "
        f"{PY} t2a_reference_driver_v2_rd.py --gate --i-am-the-t2a-execution-agent "
        f"--data-dir {DATA_DIR} --out {out} "
        f"2>&1 | tee {out_dir}/t2a_gate_run.log"
    )
    vcheck = (
        f"{PY} -c \""
        f"import json; d=json.load(open('{out}')); "
        f"g=d.get('instrument_gate'); "
        f"assert isinstance(g, dict), 'instrument_gate missing -- run did not reach the end'; "
        f"assert 't2a3' in g and 'INSTRUMENT_VALID' in g; "
        f"assert 'C1_falconmamba/openr1-mix-ext' in d.get('cells', {{}}); "
        f"assert 'C1_falconmamba/wikitext-mix-ext' in d.get('cells', {{}})\""
    )
    return [dict(
        id="990_t2a3_falconmamba_ssm_calibration",
        lane="B",
        hypothesis=("PARAM_AXIS_SCALING_DESIGN.md sec 11.4.2's T2a-3: does "
                     "falcon-mamba-7b (pure Mamba-1 SSM, demoted from the 0.90 "
                     "ceiling, causal-only) show T2b-1 + T2b-1b passing "
                     "(p<0.001) and KS>0 with a bootstrap 95% CI excluding 0, "
                     "on both corpora? Closes the last open leg of sec 11.11's "
                     "EXECUTION ORDER step (2); a precondition for step (3) "
                     "(T2b + rung admissibility) regardless of what the "
                     "inline T2a-1/T2a-2/T1c read found (sec 13.5)."),
        cmd=cmd, gpu_h_estimate=10.0,
        output_dir=out_dir, validity_check=vcheck,
        notes=("DEFERRED cell, not a fresh design decision -- sec 13.5 pre-"
               "authorized excluding C1 from the T2a attempt-2 INLINE run; "
               "this job is how that deferral actually gets a result instead "
               "of being silently forgotten. gpu_h_estimate=10.0 is a "
               "disclosed, UNCALIBRATED guess (see docstring) -- the real "
               "number could plausibly run higher given the sequential-"
               "Mamba fallback and zero completed reference point. This job "
               "REDUNDANTLY re-runs W1/W2/T2a-2/T1c (harmless, cheap "
               "relative to C1) because the pinned CLI refuses any witness "
               "subset that omits a REQUIRED_WITNESSES member -- see "
               "docstring for why a C1-only invocation does not exist. Do "
               "NOT install kernels/mamba-ssm/causal-conv1d to speed this "
               "up (sec 13.5(c); shared venv, 7 co-resident live training "
               "jobs, compiled-dependency risk not worth it for a witness "
               "that cannot change T2a-1's verdict)."),
    )]


def ncr_mapping_law_k32_grid_jobs() -> list[dict]:
    """Q1 WAVE-1 -- K=32 d(K) grid (d=33 K+1, d=40 1.25K, d=48 1.5K), n=4
    seeds each, 1x/80K steps. NCR_MAPPING_LAW_DESIGN.md S1.2/S1.4/S1.5. The
    d=64 (2K) reference arm is NOT built -- ALREADY MEASURED (S0, mean
    gpu_h/cell 0.4795, mean delta 0.803-1.05, 0/4 DEAD) -- cited, not
    relaunched, exactly the design's own S1.2 ledger treatment (0 cells,
    'reused'). ID band 009-020, not the design's own S5 semantic label
    (501-512) -- see the module comment above this section."""
    jobs = []
    seq = 9
    ceiling = 1.0   # S1.5: 2.0x planning rate, floor 1.0 GPU-h -- every K32
                     # arm's planning rate below is <0.5, so the floor binds
                     # for all three (2x0.443=0.886, 2x0.46=0.92, 2x0.47=0.94,
                     # all < 1.0).
    specs = [
        (33, "K+1", 0.443, DRATIO_OUTDIR),
        (40, "1.25K", 0.46, DRATIO125_OUTDIR),
        (48, "1.5K", 0.47, DRATIO150_OUTDIR),
    ]
    for d_ov, ratio_label, per_cell, outdir in specs:
        if outdir == DRATIO_OUTDIR:
            outdir_note = ("shared with the existing K16/K24 d=K+1 Probe-A records -- "
                            "zero-collision, K32 is a new (K,seed) key there")
        else:
            outdir_note = (f"NEW ({outdir}), required: within-outdir skip-if-COMPLETED "
                            "keys on (K,seed) only, not d, so a second non-default d at "
                            "this (K,seed) in an already-occupied outdir would silently "
                            "return the wrong-d record unrun")
        for seed in (0, 1, 2, 3):
            jid = f"{seq:03d}_laneA_mappinglaw_K32_d{d_ov}_s{seed}"
            cmd = (
                f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K 32 --seed {seed} "
                f"--steps {STEPS_MAIN} --outdir {outdir} --d-override {d_ov} "
                f"--ceiling-gpuh {ceiling} --stop-file {outdir}/STOP"
            )
            vcheck = (
                f"{PY} -c \""
                f"import json; d=json.load(open('{outdir}/earlyln_K32_s{seed}.json')); "
                f"assert d.get('status')=='COMPLETED'; "
                f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
                f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True; "
                f"assert d.get('d')=={d_ov}\""
            )
            jobs.append(dict(
                id=jid, lane="A",
                hypothesis=(
                    f"Q1 WAVE-1 (K=32@d={d_ov}, ratio {ratio_label}): does K=32 converge/"
                    f"compose at this spare-convention, extending Probe A's own K=16/K=24 "
                    f"tight-spare win one K rung further, against K=32's own ALREADY-"
                    f"MEASURED d=2K=64 reference (0/4 DEAD, mean delta 0.803-1.05)? Feeds "
                    f"S1.6's REOPENS/CONVERGES-ONLY/CLOSED-AT-THIS-K verdict map and the "
                    f"mechanical optimal-d(K) ranking (NCR_MAPPING_LAW_DESIGN.md S1.2/S1.6)."
                ),
                cmd=cmd, gpu_h_estimate=per_cell,
                output_dir=outdir, validity_check=vcheck,
                notes=(
                    f"Planning-rate cost basis (S1.4, measured-basis interpolation from "
                    f"K32's own real d=2K=64 rate 0.4795 GPU-h/cell, S0): d=K+1 priced ~7.6% "
                    f"below the 2K anchor (the twice-replicated K16/K24 discount), 1.25K/1.5K "
                    f"priced at the anchor rate (conservative, disclosed interpolation not a "
                    f"measurement). Ceiling {ceiling} GPU-h/cell (S1.5: 2.0x planning rate, "
                    f"floor 1.0 -- floor binds for every K32 arm). Outdir {outdir_note}. "
                    f"Validity check additionally asserts the record's own 'd' field equals "
                    f"the override ({d_ov}), per S5's own pinned validity-check addendum."
                ),
            ))
            seq += 1
    return jobs


def ncr_mapping_law_q2_seedext_jobs() -> list[dict]:
    """Q2 WAVE-1 primary -- K=24@d=25 seed extension n=4->n=12 (8 NEW seeds,
    {4..11}), SAME cell/outdir/budget as the already-CONFIRMED Probe-A
    K24@d25 cohort (seeds 0-3). NCR_MAPPING_LAW_DESIGN.md S2.1/S5. Zero new
    confound: identical K, d, steps, anneal to the existing 4/4 CONVERGED
    cohort -- purely a seed-count deepening for the far-depth
    seed-variance/covariate (Spearman rho(delta, front)) read. ID band
    021-028, not the design's own S5 semantic label (525-532) -- see the
    module comment above this section."""
    jobs = []
    seq = 21
    ceiling = 1.0      # S2.1's own pinned breaker, matches the existing
                        # Probe-A K24 cells' own ceiling exactly.
    per_cell = 0.468    # K24@d25's own measured rate (S0 table, 0.4680)
    d_ov = 25
    for seed in range(4, 12):
        jid = f"{seq:03d}_laneA_mappinglaw_Q2_K24_d25_s{seed}"
        cmd = (
            f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K 24 --seed {seed} "
            f"--steps {STEPS_MAIN} --outdir {DRATIO_OUTDIR} --d-override {d_ov} "
            f"--ceiling-gpuh {ceiling} --stop-file {DRATIO_OUTDIR}/STOP"
        )
        vcheck = (
            f"{PY} -c \""
            f"import json; d=json.load(open('{DRATIO_OUTDIR}/earlyln_K24_s{seed}.json')); "
            f"assert d.get('status')=='COMPLETED'; "
            f"assert d.get('train',{{}}).get('step')=={STEPS_MAIN}; "
            f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True; "
            f"assert d.get('d')=={d_ov}\""
        )
        jobs.append(dict(
            id=jid, lane="A",
            hypothesis=(
                "Q2 WAVE-1 primary (K=24@d=25 seed extension, n=4->n=12): is the far-depth "
                "seed-variance already observed in the n=4 cohort (fronts {21,93,189,189}, "
                "sweep_min_rec max 0.0511) genuinely bimodal/unreliable, or would more seeds "
                "reveal a graded (Spearman rho(delta,front)) relationship -- n=4 cannot "
                "structurally distinguish these (NCR_MAPPING_LAW_DESIGN.md S2.1). No new "
                "confound vs the existing 4/4 CONVERGED cohort (identical K/d/steps/anneal)."
            ),
            cmd=cmd, gpu_h_estimate=per_cell,
            output_dir=DRATIO_OUTDIR, validity_check=vcheck,
            notes=(
                "MEASURED cost basis (S0/S2.1: K24@d25's own real rate, 0.468 GPU-h/cell). "
                "Ceiling 1.0 GPU-h/cell (S2.1's own pinned figure, matches the existing "
                "Probe-A K24 cells' ceiling exactly). Shared outdir (results_earlyln_dratio/, "
                "zero-collision: seeds 4-11 are new (K,seed) keys, the existing seeds 0-3 "
                "records at this same K/d are untouched). Validity check additionally "
                f"asserts the record's own 'd' field equals the override ({d_ov}), per S5's "
                "own pinned validity-check addendum."
            ),
        ))
        seq += 1
    return jobs


def ncr_mapping_law_wave1_jobs() -> list[dict]:
    """WAVE-1 -- all NCR mapping-law cells committed unconditionally this
    round (21 cells nominal: 1 K48-probe-citation + 12 K32-grid + 8
    Q2-seedext, 9.2505 ~= 9.25 GPU-h, matches NCR_MAPPING_LAW_DESIGN.md
    S1.5's own '9.246 ~= 9.25' ledger). WAVE-1b (K48's own d(K) grid, IDs
    513-524) is DELIBERATELY NOT included -- see the module comment above
    this section and NCR_MAPPING_LAW_DESIGN.md S1.2/S1.6/S5."""
    return (ncr_mapping_law_k48_probe_citation_job()
            + ncr_mapping_law_k32_grid_jobs()
            + ncr_mapping_law_q2_seedext_jobs())


# ---------------------------------------------------------------------------
# NCR MAPPING-LAW Q4, 2026-07-12e -- K=32/d=33 BUDGET-RESCUE DISSOCIATION
# PROBE (2x/4x budget, n=4 seeds each, 8 cells). Pre-registration:
# NCR_MAPPING_LAW_DESIGN.md (this file's own commit) S4 -- opened per S11.5's
# own "Recommended next experiment"
# (NOVEL_ARCH_WATERFALL.md:4987-4998): does extra budget rescue K=32/d=33
# over the Gate-1 bar (S11.5: 3/4 PARTIAL, best seed 0.8711) the way it did
# for K16/d=32 (S11.3/S11.4: 1/4->3/4->4/4 CONVERGED under 2x/4x), and if so
# does far-depth composition come WITH that convergence (as tight-spare does
# at K<=24) or NOT (as d=2K does at every K)? Reuses
# NCR_NEXT_LEVER_DESIGN.md's own Q1 budget-probe recipe verbatim: SAME step
# counts (STEPS_BUDGET2X/STEPS_BUDGET4X, already defined above, not
# invented), SAME linear-in-steps cost-extrapolation method, SAME outdirs
# (results_earlyln_budget2x/, results_earlyln_budget4x/ -- reused, not
# freshly created; K=32 is a new (K,seed) key in both, verified empty of any
# K32 record on the live box this round). Zero model/training-logic changes
# (--d-override/--steps/--outdir/--ceiling-gpuh/--stop-file all pre-existing;
# --anneal-frac deliberately NOT passed, stays at its implicit 0.5 default).
# Pricing basis: MEASURED, not the stale S1.4 planning value (0.443) --
# S11.5's own real K32@d=33 1x gpu_h fields, mean 0.5687847 GPU-h/cell
# (individual seeds 0.5417-0.5942), linearly extrapolated to 2x/4x. Job IDs
# 192-199: verified collision-free on the live box this round (used bands:
# 000-007, 008-028, 050-057, 060-063, 066-081, 100-127, 200-236, 300-315,
# 400-431, 450-451; 192-199 sits inside the 128-199 gap, the slice closest to
# the true front at deploy time, box lowest-pending-prefix=222) -- a
# deliberate deviation from any single fixed numbering convention, mirroring
# every prior wave's own "front of pending, verified at deploy time, not
# assumed" discipline (regate_2026-07-12.md S8.2).
# ---------------------------------------------------------------------------
STEPS_BUDGET2X_K32D33 = STEPS_BUDGET2X    # 160,000 -- reused constant, not reinvented
STEPS_BUDGET4X_K32D33 = STEPS_BUDGET4X    # 320,000 -- reused constant, not reinvented


def ncr_mapping_law_q4_k32_budget_jobs() -> list[dict]:
    """Q4 -- K=32@d=33 budget-rescue dissociation probe, 2x + 4x budget, n=4
    seeds each (8 cells). NCR_MAPPING_LAW_DESIGN.md S4 (this file). Cost
    basis MEASURED (not extrapolated from a formula): S11.5's own real
    K32@d=33 1x rate (mean 0.5687847 GPU-h/cell over 4 seeds
    0.5941774/0.5656882/0.5737207/0.5415527, 80,000 steps), linearly
    extrapolated to 2x/4x -- the SAME method Q1/budget2x used at K16 (no
    K32 2x/4x data exists yet to check sub-/super-linearity directly; K16's
    own precedent found the ladder mildly SUB-linear if anything at this
    shape family, so linear extrapolation is a conservative planning basis,
    not an optimistic one)."""
    jobs = []
    per_cell_1x = 0.5687847442097134   # S11.5's own measured mean, S4.7
    specs = [
        (2, STEPS_BUDGET2X_K32D33, per_cell_1x * 2.0, BUDGET2X_OUTDIR, 192),
        (4, STEPS_BUDGET4X_K32D33, per_cell_1x * 4.0, BUDGET4X_OUTDIR, 196),
    ]
    for budget_mult, steps, per_cell_nominal, outdir, seq0 in specs:
        ceiling = max(1.0, round(per_cell_1x * budget_mult * 2.0, 1))
        seq = seq0
        for seed in (0, 1, 2, 3):
            jid = f"{seq:03d}_laneA_mappinglaw_Q4_K32budget{budget_mult}x_d33_s{seed}"
            cmd = (
                f"cd {NCR_DIR} && {PY} ncr_earlyln_scale.py --cell --K 32 --seed {seed} "
                f"--steps {steps} --outdir {outdir} --d-override 33 "
                f"--ceiling-gpuh {ceiling} --stop-file {outdir}/STOP"
            )
            vcheck = (
                f"{PY} -c \""
                f"import json; d=json.load(open('{outdir}/earlyln_K32_s{seed}.json')); "
                f"assert d.get('status')=='COMPLETED'; "
                f"assert d.get('train',{{}}).get('step')=={steps}; "
                f"assert 'eval' in d and d.get('blank_out',{{}}).get('passed') is True; "
                f"assert d.get('d')==33\""
            )
            jobs.append(dict(
                id=jid, lane="A",
                hypothesis=(
                    f"Q4 (K=32@d=33, {budget_mult}x budget = {steps} steps): does extra "
                    f"budget push K=32/d=33's Gate-1 rate (S11.5: 3/4 PARTIAL at 1x, best "
                    f"seed 0.8711) over the CONVERGED-ROBUST bar, as it did for K16/d=32 "
                    f"(S11.3/S11.4: 1/4->3/4->4/4 CONVERGED under 2x/4x)? And if so, does "
                    f"far-depth composition (failure front toward/past K=32's own h*=253) "
                    f"come WITH that convergence (the tight-spare pattern at K<=24) or NOT "
                    f"(the d=2K pattern at every K tested to date)? Feeds the 3-way "
                    f"BUDGET-RESCUES-BOTH / BUDGET-RESCUES-CONVERGENCE-ONLY / "
                    f"BUDGET-DOES-NOTHING verdict map (NCR_MAPPING_LAW_DESIGN.md S4.6); "
                    f"only BUDGET-RESCUES-BOTH unblocks WAVE-1b."
                ),
                cmd=cmd, gpu_h_estimate=round(per_cell_nominal, 4),
                output_dir=outdir, validity_check=vcheck,
                notes=(
                    f"MEASURED cost basis (S11.5's own real K32@d=33 1x rate, mean "
                    f"0.5687847 GPU-h/cell, linearly extrapolated {budget_mult}x -- NOT "
                    f"the stale S1.4 planning value 0.443, which under-measured the true "
                    f"rate by 28%, NCR_MAPPING_LAW_DESIGN.md S4.7). Ceiling {ceiling} "
                    f"GPU-h/cell (~=2.02x nominal, matching every prior wave's own "
                    f"~2x-nominal breaker convention). Outdir ({outdir}) is an EXISTING, "
                    f"REUSED outdir from the K16/K24 budget-probe waves (S11.3/S11.4) -- "
                    f"verified empty of any K32 record on the live box this round; K=32 is "
                    f"a new (K,seed) resume-key there, zero collision (skip-if-COMPLETED "
                    f"keys on (K,seed) within the outdir, not on steps). Validity check "
                    f"additionally asserts the record's own 'd' field equals 33, per this "
                    f"program's own standing d-override validity-check addendum."
                ),
            ))
            seq += 1
    return jobs


def param_axis_14m_fulltoken_jobs() -> list[dict]:
    """PARAM_AXIS_SCALING_DESIGN.md S22 -- the 14M full-token extension.

    S21 (commit b6b0b18) confirmed H-1: the param-axis primary fit has only
    |A| = 2 admissible rungs (14M, 98M) and can ask NO trend verdict at all.
    Root cause: S9.6 item 2 admits a rung only at >= 1.0 token/param at the
    COMMON token slice, and that slice is T = min_r tokens_max(r) = 0.32768B
    -- set by the 14M rung, which was capped at 20,000 steps and never
    extended. At that slice 392M reads 0.836 tok/param and 1.31B reads 0.250;
    both are excluded, and |A| = 2 < 3.

    Extending 14M to the SAME 67,547-step budget 98M and 392M (cells 029/030)
    already run raises T to 1.10669B, which puts 14M / 98M / 392M at exactly
    step 67,547 -- IDENTICAL token counts, tok/param 78.8 / 11.3 / 2.82. All
    three clear the >= 1.0 floor, |A| = 3, the trend verdict becomes askable,
    and S7-F2's token-mismatch confound is discharged by EXACT MATCH rather
    than by argument. At 1.71 GPU-h total this is the cheapest unblock in the
    campaign; without it the 31.4 GPU-h already sunk into 029/030 buys nothing.

    FROM SCRATCH, not resumed from the existing 20,000-step checkpoint. This
    is deliberate. lm_pretrain_rd.py has NO resume path: its only warm-start is
    --init-checkpoint, which its own help text (lm_pretrain_rd.py:3103) states
    begins "a fresh LR warmup/decay cycle, by design". The LR schedule is a
    function of the TOTAL step count (get_lr(step, lr, warmup, total_steps=
    args.steps), linear warmup + cosine decay). Warm-starting the 20K
    checkpoint would therefore produce a model that saw 20,000 + 67,547 =
    87,547 steps (1.43B tokens) under two stitched cosine cycles -- neither
    token-matched to the other rungs nor trained under the single-cosine
    schedule they got. That defeats the entire purpose of the extension. A
    from-scratch 67,547-step run costs 0.86 GPU-h/cell; the warm start would
    have saved ~0.25 GPU-h and destroyed the comparison.

    Arm/config/seed match cells 029/030 (the 392M full-token pair) exactly:
    per_token, lambda 0.58, seed 3. Seed 3 also exists at 98M's own 67,547-step
    cells (fixscale_seedext_arm_per_token_98m_{corpus}_s3), so seed 3 is common
    to all three rungs at the full-token budget.

    Priority: filenames 031/032 sort above the entire pending backlog (which
    begins at 400), so these claim the first GPU freed. They cannot preempt
    029/030/000 -- those are already claimed and running, and a worker only
    claims when its OWN GPU is free.
    """
    jobs = []
    steps_full = 67_547           # the 98M/392M full-token budget
    # Measured 14M rate: 0.04564 s/step -- from a REAL 14M run JSON
    # (wall_s = 912.87 over 20,000 steps; PARAM_AXIS_SCALING_DESIGN.md S21.0
    # V-8), same script, same dm256/ds64/L2 config, same batch 32 / seq 512.
    rate_s_per_step = 0.04564
    est_gpu_h = steps_full * rate_s_per_step / 3600.0     # = 0.856 GPU-h
    # TIMEOUT PRICING. The mispriced-timeout bug has bitten this project TWICE
    # (job 200's 160000s against a true 71.2h need; the _fixscale_cell 36000s
    # default against 392M's 15.69h need). It does not get a third. 36000s =
    # 10.0h against a 0.86h estimate is 11.7x headroom: the run still completes
    # even if the true rate is 10x worse than measured.
    timeout_s = 36000
    for seq, corpus in ((31, "openr1-mix-ext"), (32, "wikitext-mix-ext")):
        name = f"fixscale_fulltoken_arm_per_token_14m_{corpus}_s3"
        cmd, vcheck, ckpt_dir, out = _fixscale_cell(
            name, corpus, 256, 64, 2, steps_full, 3, "per_token",
            timeout=timeout_s)
        jid = f"{seq:03d}_laneB_14m_fulltoken_per_token_{corpus}_s3"
        jobs.append(dict(
            id=jid, lane="B",
            hypothesis=(
                "PARAM_AXIS_SCALING_DESIGN.md S22 / S21's H-1: the param-axis "
                "primary fit currently has |A| = 2 admissible rungs and can ask "
                "no trend verdict, because the common token slice T = "
                "min_r tokens_max(r) is pinned at 0.32768B by the 14M rung's "
                "un-extended 20,000-step cap, which excludes 392M (0.836 "
                "tok/param) and 1.31B (0.250). Extending 14M to the same "
                "67,547-step budget 98M and 392M already run raises T to "
                "1.10669B and puts 14M/98M/392M at EXACTLY step 67,547 -- "
                "identical token counts, tok/param 78.8/11.3/2.82, |A| = 3. "
                "This does not pre-judge the verdict; it makes one askable."),
            cmd=cmd, gpu_h_estimate=round(est_gpu_h, 2),
            output_dir=FIXSCALE_RESULTS_ROOT, validity_check=vcheck,
            notes=(
                f"cost basis: MEASURED 0.04564 s/step (S21.0 V-8: a real 14M run "
                f"JSON, wall_s=912.87 / 20,000 steps, same script and config) x "
                f"{steps_full:,} steps = {est_gpu_h:.2f} GPU-h. HIGH confidence "
                f"(measured on this exact config, not extrapolated). "
                f"--internal-timeout {timeout_s}s (10.0h) = 11.7x the estimate: "
                f"deliberately generous because a mispriced timeout has already "
                f"cost this project two runs (job 200's 160000s vs a true 71.2h "
                f"need; the _fixscale_cell 36000s default vs 392M's 15.69h need). "
                f"FROM SCRATCH, not warm-started: lm_pretrain_rd.py has no resume "
                f"path, and --init-checkpoint explicitly restarts the LR "
                f"warmup/decay cycle (lm_pretrain_rd.py:3103), so warm-starting "
                f"the existing 20,000-step checkpoint would yield 87,547 total "
                f"steps / 1.43B tokens under two stitched cosine cycles -- NOT "
                f"token-matched to 98M/392M and not on their single-cosine "
                f"schedule. Arm/config/seed match cells 029/030 exactly "
                f"(per_token, lambda 0.58, seed 3); seed 3 also exists at 98M's "
                f"own 67,547-step cells, so it is common to all three rungs. "
                f"Filename prefix {seq:03d} sorts above the pending backlog "
                f"(which starts at 400) so this claims the first freed GPU; it "
                f"cannot preempt 029/030/000, which are already claimed and "
                f"running."),
        ))
    return jobs


def param_axis_rung_y_dstate64_jobs() -> list[dict]:
    """DSTATE_CONFOUND_PREREG.md (commit 1a4add5) Option 1 -- RUNG Y.

    THE ATTRIBUTION CELL. The param ladder runs d_state = 64 / 64 / 128 at
    14M / 98M / 392M, so the DeltaNet state (d_state x d_state per layer)
    QUADRUPLES at the single interval a 3-point fit must span. The pre-reg's
    S3 algebra shows the resulting 3-point/3-parameter fit is SATURATED (zero
    residual df) and that, in closed form,

        beta_hat = (M_98 - M_14) / (x_98 - x_14)     EXACTLY

    -- the 392M rung contributes NOTHING to the parameter slope, while the
    d_state step term is PERFECTLY ALIASED with curvature. So a RISES verdict
    (the only headline this design can still produce) cannot be attributed to
    parameter count, and NO re-analysis fixes it. Only a new cell does.

    Rung Y is that cell: 392M-scale params at d_state = 64.
      - It completes a clean 3-rung d_state=64 ladder {14M, 98M, Y}, which is
        S9.6's >=3-rung minimum met with the SECOND AXIS HELD FIXED -- i.e.
        CLAUDE.md's own hard rule ("hold any second architectural axis FIXED
        when testing a primary hypothesis") finally obeyed. GATE-A (pre-reg
        S5) flips from FAIL to PASS.
      - It costs 0.46% power: S_xx = 1.0446 on the clean ladder vs 1.0542 on
        the confounded one (both recomputed here against the MEASURED param
        counts; they reproduce the pre-reg's S4 figures exactly).
      - It turns the EXISTING 392M (d_state=128) rung into a MATCHED-PARAMS
        measurement of the state-width effect: gamma = M(392M,128) - M(Y,64)
        at a 1.609% param mismatch. That is a designed experiment on
        fast-weight state capacity -- a first-class result on this program's
        own thesis, not a nuisance adjustment.

    MEASURED PARAM COUNT (not arithmetic -- the pre-reg's own S1/S4 hand
    arithmetic is WRONG here and this correction is the reason the cell was
    priced from a real instantiation). Instantiating the REAL DeltaNetLM on
    the box at vocab 50257:
        dm=1536 ds=128 L=16 -> 391,869,440   (reproduces S21 V-1 EXACTLY,
                                              which validates the method)
        dm=1536 ds= 64 L=16 -> 385,564,672   (RUNG Y, measured)
    The pre-reg predicted 385,577,984 for rung Y -- over by 13,312 params
    (= 16 layers x 832). The missing terms are both d_state-scaled and both
    omitted from its hand arithmetic: the short causal conv over q/k/v
    (3 * d_state * conv_size = 768/layer) and a d_state-wide norm (64/layer).
    Nothing material moves: the param delta vs the 392M rung is -1.609% (the
    pre-reg said -1.61%) and S_xx/power are unchanged to 4 decimals. Recorded
    because the design doc's stated figure is off and a later agent will
    otherwise "correct" the right number back to the wrong one.

    Arm/config/seed/budget match cells 029/030 (the live 392M full-token
    per_token pair, verified by reading their deployed specs on the box):
    per_token, lambda 0.58, seed 3, 67,547 steps, batch 32, seq 512. The ONLY
    difference from 029/030 is d_state 128 -> 64. Same step count => the same
    token count => the common-slice token match (T = 1.10669B) holds BY
    CONSTRUCTION and rung Y does not move T.

    TIMEOUT PRICING -- MEASURED, and deliberately NOT assuming a speedup.
    The 392M d_state=128 rate was re-measured on THIS box from the two live
    jobs (elapsed wall / steps completed, i.e. including startup and real
    8-way GPU contention):
        029: 39,454 s / 47,000 steps = 0.839 s/step
        030: 38,302 s / 45,600 steps = 0.840 s/step
    (both slight OVER-estimates: elapsed includes data load, and the step
    read precedes the ps snapshot). Rung Y differs ONLY by d_state 128 -> 64
    with d_model/n_layers/conv_size/batch/seq/steps identical, and EVERY
    d_state-dependent cost term is monotone non-decreasing in d_state (q/k/v/o
    projections 4*d_model*d_state; the short conv 3*d_state*conv_size; the
    chunked recurrent kernel's T*d_state^2 term), while the terms that
    dominate -- the FFN's 8*d_model^2 and the 50,257-way logits head -- are
    UNCHANGED. Therefore
        rate(d_state=64) <= rate(d_state=128) = 0.840 s/step
    is a rigorous UPPER BOUND, not an assumed speedup. The cell is priced as
    if d_state=64 costs exactly the same as d_state=128 -- strictly
    conservative -- giving 67,547 * 0.840 / 3600 = 15.76 GPU-h/cell (31.5 for
    the pair; the pre-reg budgeted ~30, and 31.4 at its own pessimistic rate).

    --internal-timeout 86400s (24.0h) = 1.52x that already-conservative upper
    bound, and ~1.8x the likely true d_state=64 need. This is deliberately
    MORE headroom than 029/030's 72000s (which is only 1.27x). The
    mispriced-timeout bug has bitten this project TWICE (job 200's 160000s
    against a true ~71h need, ~44 GPU-h burned for nothing; and the
    _fixscale_cell 36000s default against 392M's 15.69h need). It does not get
    a third. The asymmetry is the whole argument: a too-SMALL timeout
    guarantees a wasted run, while a too-LARGE one only costs extra burn on a
    job that is already pathological -- the timeout is a rail, not a budget,
    and a healthy cell exits on step count long before it fires.

    Priority: ids 033/034 sort above the entire pending backlog (400-431, and
    990) so they claim the first GPUs freed -- 031/032 (14M) finish within the
    hour and 029/030 (392M) land ~23:00 UTC. They CANNOT preempt anything:
    queue_worker.sh only claims when its OWN GPU is free (it never evicts), so
    029/030/031/032/000 and the live 231/233/234 are untouchable by this.
    """
    jobs = []
    steps_full = 67_547           # IDENTICAL to 029/030/031/032 -- token-matched by construction
    rate_s_per_step = 0.840       # MEASURED ds=128 upper bound (see docstring); NO speedup assumed
    est_gpu_h = steps_full * rate_s_per_step / 3600.0     # = 15.76 GPU-h
    timeout_s = 86400             # 24.0h = 1.52x the conservative upper bound
    for seq, corpus in ((33, "openr1-mix-ext"), (34, "wikitext-mix-ext")):
        name = f"fixscale_fulltoken_arm_per_token_392mY_ds64_{corpus}_s3"
        cmd, vcheck, ckpt_dir, out = _fixscale_cell(
            name, corpus, 1536, 64, 16, steps_full, 3, "per_token",
            timeout=timeout_s)
        jid = f"{seq:03d}_laneB_392mY_ds64_per_token_{corpus}_s3"
        jobs.append(dict(
            id=jid, lane="B",
            hypothesis=(
                "DSTATE_CONFOUND_PREREG.md (H-3) Option 1 -- THE ATTRIBUTION "
                "CELL. The param ladder bundles a second architectural axis: "
                "d_state = 64/64/128 at 14M/98M/392M, so the DeltaNet state "
                "quadruples at the one interval the 3-point fit must span. The "
                "fit is SATURATED (3 points, 3 params, zero residual df) and in "
                "closed form beta_hat = (M_98 - M_14)/(x_98 - x_14) EXACTLY -- "
                "the 392M rung contributes ZERO information to the parameter "
                "slope, and the d_state step is perfectly aliased with "
                "curvature. A RISES verdict therefore CANNOT be attributed to "
                "parameter count, and no re-analysis fixes it. Rung Y (392M-"
                "scale params at d_state=64) completes a clean 3-rung "
                "d_state=64 ladder {14M, 98M, Y} -- second axis held fixed, "
                "GATE-A flips FAIL -> PASS -- at a 0.46% power cost, AND turns "
                "the existing 392M (d_state=128) rung into a matched-params "
                "(1.609% mismatch) measurement of the state-width effect, "
                "which is a first-class result on this program's own "
                "fast-weight-capacity thesis. Does not pre-judge the verdict; "
                "it makes an ATTRIBUTABLE one possible."),
            cmd=cmd, gpu_h_estimate=round(est_gpu_h, 2),
            output_dir=FIXSCALE_RESULTS_ROOT, validity_check=vcheck,
            notes=(
                f"PARAMS: 385,564,672 -- MEASURED by instantiating the real "
                f"DeltaNetLM (dm=1536, ds=64, L=16, vocab 50257) on the box, NOT "
                f"by arithmetic. Method validated: the same instantiation "
                f"reproduces the existing 392M rung at 391,869,440 EXACTLY "
                f"(S21 V-1). NOTE the pre-reg's own hand arithmetic says "
                f"385,577,984 and is WRONG by 13,312 (= 16 layers x 832): it "
                f"omits the d_state-scaled short conv (3*d_state*conv_size = "
                f"768/layer) and a d_state-wide norm (64/layer). Immaterial "
                f"(delta vs 392M = -1.609%, S_xx = 1.0446 clean vs 1.0542 "
                f"confounded, power loss 0.46% -- all reproduce the pre-reg's "
                f"S4) but recorded so it is not 'corrected' back. "
                f"COST: MEASURED 0.840 s/step at d_state=128 from the two LIVE "
                f"392M jobs on this box (029: 39,454s/47,000 steps; 030: "
                f"38,302s/45,600 steps, both including startup and real 8-way "
                f"contention). d_state=64 is priced at the SAME rate -- NOT a "
                f"speedup assumption: every d_state-dependent cost term is "
                f"monotone non-decreasing in d_state while the dominant FFN "
                f"(8*d_model^2) and 50,257-way head are unchanged, so 0.840 "
                f"s/step is a rigorous UPPER BOUND. x {steps_full:,} steps = "
                f"{est_gpu_h:.2f} GPU-h/cell, {2*est_gpu_h:.1f} for the pair. "
                f"--internal-timeout {timeout_s}s (24.0h) = 1.52x that "
                f"conservative bound and ~1.8x the likely true need -- MORE "
                f"headroom than 029/030's 72000s (1.27x), on purpose: the "
                f"mispriced-timeout bug has already cost this project two runs "
                f"(job 200's 160000s vs a true ~71h need, ~44 GPU-h burned; the "
                f"_fixscale_cell 36000s default vs 392M's 15.69h need). A "
                f"too-small timeout guarantees a wasted run; a too-large one "
                f"only burns extra on an already-pathological job. "
                f"ARM/CONFIG/SEED/BUDGET match cells 029/030 exactly (per_token, "
                f"lambda 0.58, seed 3, 67,547 steps, batch 32, seq 512, verified "
                f"by reading their deployed specs on the box); the ONLY "
                f"difference is d_state 128 -> 64, so the common-slice token "
                f"match (T = 1.10669B) holds BY CONSTRUCTION and T does not "
                f"move. Ids {seq:03d} sort above the whole pending backlog "
                f"(400-431, 990) so they claim the first freed GPUs; they cannot "
                f"preempt anything, since queue_worker.sh only claims when its "
                f"OWN gpu is free and never evicts."),
        ))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "jobs", "pending"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    all_jobs = (lane_a_jobs() + lane_b_jobs() + lane_c_jobs()
                + regate_20260712_jobs() + ncr_next_lever_q1_jobs()
                + ncr_next_lever_probe_ab_jobs() + ncr_mapping_law_wave1_jobs()
                + ncr_mapping_law_q4_k32_budget_jobs()
                + param_axis_14m_fulltoken_jobs()
                + param_axis_rung_y_dstate64_jobs())

    total_by_lane = {}
    for j in all_jobs:
        out_path = os.path.join(args.outdir, j["id"] + ".json")
        with open(out_path, "w") as f:
            json.dump(j, f, indent=1)
        total_by_lane.setdefault(j["lane"], [0, 0.0])
        total_by_lane[j["lane"]][0] += 1
        total_by_lane[j["lane"]][1] += j["gpu_h_estimate"]

    grand_total = sum(v[1] for v in total_by_lane.values())
    print(f"Generated {len(all_jobs)} job specs into {args.outdir}")
    for lane in sorted(total_by_lane):
        n, h = total_by_lane[lane]
        print(f"  Lane {lane}: {n} jobs, {h:.2f} GPU-h estimated")
    print(f"  TOTAL: {len(all_jobs)} jobs, {grand_total:.2f} GPU-h estimated")


if __name__ == "__main__":
    main()
