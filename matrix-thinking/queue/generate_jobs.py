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
    # 1.5B tokens/run @ batch=16/seq=512 = 183,105 steps, measured rung-3 rate
    # per_step_s=0.7135 -> 183105*0.7135/3600 = 36.29 GPU-h/run for the plain
    # architecture; the frozen-bias blend itself measured <=1.2% overhead in
    # the S13 timing pilots, folded into the estimate below as a flat +2%
    # -> 37.0 GPU-h. --internal-timeout is set well above the 36.29h runtime
    # (audit MAJOR M1: the prior 36000s=10h default guaranteed a timeout).
    dm, ds, L = 2560, 128, 22
    steps_1p31b = 183_105
    name = f"queue_1p31b_arm_per_token_openr1-mix-ext_s0"
    ckpt_dir = f"{TRACKC_CKPT_ROOT}/{name}"
    out = f"{TRACKC_RESULTS_ROOT}/{name}.json"
    cmd = (
        f"mkdir -p {ckpt_dir} {TRACKC_RESULTS_ROOT} && "
        f"cd {DELTANET_RD_DIR} && {PY} lm_pretrain_rd.py "
        f"--corpus openr1-mix-ext --data-dir {DATA_DIR} "
        f"--d-model {dm} --d-state {ds} --n-layers {L} --seq-len 512 "
        f"--batch-size 16 --steps {steps_1p31b} --ckpt-every 10000 "
        f"--seed 0 --internal-timeout 160000 "
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
        cmd=cmd, gpu_h_estimate=37.0,
        output_dir=TRACKC_RESULTS_ROOT, validity_check=vcheck,
        notes=("cost basis: measured rung-3 per_step_s=0.7135 (SCALE_TRANSFER_DESIGN.md "
               "S5.6, batch=16) x 183,105 steps / 3600 = 36.29 GPU-h, +2% for the "
               "frozen-bias blend (S13's own <=1.2% measured overhead, rounded up) = "
               "37.0h. --internal-timeout 160000s (44.4h) leaves headroom above the "
               "36.29h estimate (audit fix: the original 36000s=10h default guaranteed "
               "a timeout). HIGH confidence (measured rate, not formula-extrapolated). "
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(HERE, "jobs", "pending"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    all_jobs = lane_a_jobs() + lane_b_jobs() + lane_c_jobs() + regate_20260712_jobs()

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
