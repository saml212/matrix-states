# Correction memo — "The Gradient Does Not See Rank" (ICML 2026 MI workshop) — 2026-09-01

Status: FOUND by a raw-archive trace and CONFIRMED by an independent second
trace the same evening. NOT yet applied. Sam signs off on every changed
number before the arXiv v1 is built. Two decisions are his (§3 below).

## 1. The error

`EXPERIMENT_LOG.md:1425` (Round-PC summary table, commit 9e6a1c1) lists the
γ=0 seed-1337 matrix-CODI ProsQA best-accuracy as **80.47%**. That number is
the **GPT-2-medium vanilla-SFT seed-1337** control
(`experiment-runs/2026-04-13_round6_scale/vanilla_gpt2medium/results.json`,
`mode=pure_sft`, `best_accuracy=0.8046875`). The γ=0 seed-1337 matrix-CODI
run's own archive reads **78.91%**
(`experiment-runs/2026-04-12_round3_gamma0/results.json`
`train_matrix.best_accuracy=0.7890625`, config `gamma=0.0, seed=1337,
batch 16, epochs 25`; `SUMMARY.txt` and `final_eval.json` agree; the log's
own Round-3 section at lines 1074/1213 also says 78.91 and −2.86pp).
Corrected three-seed set: {78.91, 81.25, 82.81}, mean **80.99**, sample SD
**1.96pp** (v1: {80.47, 81.25, 82.81} = 81.51 ± 1.19).

Independently: §6's "matrix-CODI at 82.03%" is the **γ=1**, effective-batch-32,
single-seed Round-2 run (`2026-04-12_round2_prosqa/run_b_matrix/…/results.json`),
a different configuration from the γ=0 sweep, not a seed draw of it. The
README's open item ("seed variance?") had a wrong premise on both ends.

What is unchanged: the SFT baseline 81.77 (mean of round4 seeds
1337/42/7 = 81.25/82.03/82.03), the GPT-2-medium gap −0.78 (80.47 vanilla
vs 79.69 matrix), every rank-k ablation curve, the Spearman statistics, the
probe and negative-control results, and the qualitative conclusion (rank
varies 3× across seeds at similar accuracy; no accuracy gain over SFT).

## 2. Every site that must change (v1 → v2)

| File | Line | Old → New |
|---|---|---|
| `main.tex` (abstract) | 75 | `$81.5 \pm 1.2$pp` → `$81.0 \pm 2.0$pp` |
| `sections/01_intro.tex` | 49 | `$81.51\!\pm\!1.2$pp` → `$80.99\!\pm\!2.0$pp` |
| `sections/03_rank_blind_readout.tex` | 120 | `$81.51\pm 1.2$pp` → `$80.99\pm 2.0$pp`; list seeds `78.91/81.25/82.81` |
| `sections/04_depth_scale.tex` | 44–46 | `$-1.3$pp at GPT-2 small … both within … $\pm 1.2$pp` → see decision A; SD → `$\pm 2.0$pp` |
| `sections/05_positive_control.tex` | 42 | `flatten baseline ($80.47\%$)` → `($78.91\%$)` (the svd_aug 78.12 < flatten comparison now rests on 1/128) |
| `sections/06_related_work.tex` | 8–14 | name γ=1 explicitly; drop "comparable margin" (see replacement text below) |
| `sections/07_discussion.tex` | 18 | `\{81.25, 82.81, 80.47\}` → `\{81.25, 82.81, 78.91\}` |
| `figures/generate_figures.py` | 94 | `[80.47, 81.25, 82.81]` → `[78.91, 81.25, 82.81]` |
| same | 110 | `ax.set_ylim(79.5, 84.0)` → `ax.set_ylim(78.0, 84.0)` (else seed 1337 is clipped off fig 2) |
| same | 126 | `matrix = [80.47, 79.69, np.nan]` → `[78.91, 79.69, np.nan]` (line 125 `vanilla` is correct, do not touch) |
| same | 93 | `12.9` → `12.7` (raw `rank_dynamics.json` epoch-25 mean; optional) |
| `arxiv/metadata.md` | 33 | abstract `81.5 +/- 1.2` → `81.0 +/- 2.0` |
| `poster/poster.tex` | 123 | `81.5\pm1.2` → `81.0\pm2.0` |
| `PAPER_READER_VIEW.md` | 32, 112, 144, 262 | same substitutions |
| `EXPERIMENT_LOG.md` | 1371, 1425, 1433, 9020 | 80.47 → 78.91 with a DATED correction note, never silent |
| `pebble-ai-site/findings/matrix-codi-rank-blindness.html` | 492, 574–611, 711, 719 | same; deploys on push |

Replacement for §6 lines 8–14 (proposed, Sam edits):
> The qualitative phenomenon replicates at our operating point under the
> $\gamma=1$ (distillation-loss-active) configuration: matrix-CODI reaches
> $82.03\%$ against pure SFT's $81.77\%$, a single-seed comparison. The
> $\gamma=0$ configuration used in \S3–\S4 removes the distillation loss and
> averages $80.99\pm2.0$pp over three seeds ($78.91/81.25/82.81\%$); the
> $\gamma=1$ point falls inside that spread. Latent feedback does not move
> accuracy beyond vanilla SFT under either setting.

## 3. Decisions for Sam

**A. Which GPT-2-small gap to quote in §4.** Options: (i) seed-1337 γ=0 vs
the 3-seed SFT mean = 78.91 − 81.77 = **−2.86pp** (the log's own Round-3/4
framing; then "both within the three-seed SD" is false and must go); (ii)
3-seed matrix mean vs 3-seed SFT mean = 80.99 − 81.77 = **−0.78pp**
(apples-to-apples; the "within SD" clause survives); (iii) seed-matched
1337 vs 1337 = 78.91 − 81.25 = −2.34pp. Recommendation: (ii) in the text
with (i) in a footnote, because fig 3's small-scale bar is a single seed
and the reader should see both.

**B. Eval-set wording.** `best_accuracy` is the max over 25 per-epoch
evaluations on the FIRST 128 of the 500 ProsQA test problems (resolution
0.78pp; a best-of-25 selection on test). `sections/02_background.tex:70–72`
calls this "the 128-problem test split from the original CODI release";
per the archived script it is the first 128 of COCONUT's 500-problem test
file. The 500-problem last-checkpoint `final_accuracy` exists for some runs
(γ=0 s1337 75.0; SFT 76.4/78.6/79.6) and trails SFT by 3.2pp for the γ=0
run. The paper must define the field it quotes in one sentence.

**C. Ranks {4, 12, 13}.** Seed 1337's 13 is verified from raw
(`rank_dynamics.json`, epoch-25 mean 12.71). Seeds 42 and 7 have ONLY
`SUMMARY.txt` (accuracy, no rank field) in the repo and on the SSD; their
ranks 4 and 12 come from the log table alone and no checkpoint survives to
re-measure. The paper should say so in the fig-2 caption or reproducibility
section.

## 4. arXiv comments-field note (draft)

v1 on arXiv corrects a data-entry error in the workshop version: the
seed-1337 entry of the three-seed flatten-readout replication (§3.3, Fig. 2,
§4.2/Fig. 3, §5, §7) was reported as 80.47%, the accuracy of the vanilla-SFT
GPT-2-medium control, not of the γ=0 matrix-CODI run; the archived value is
78.91%. The three-seed set is {78.91, 81.25, 82.81}, mean 81.0 ± 2.0pp
(was 81.5 ± 1.2pp). The GPT-2-medium gap, all rank-k ablation curves,
Spearman statistics, probe and negative-control results are unchanged.
Accuracies are best-of-25 per-epoch evaluations on the first 128 of the
500 ProsQA test problems. The conclusion, that rank varies 3× across seeds
at similar accuracy, stands with a wider accuracy spread (3.9pp) than
stated.

## 5. Process lesson (goes to the learnings DB)

The README flagged this as "verify before any arXiv revision" on
2026-04-28 and nobody closed it against raw files. Figure scripts that
hard-code numbers "from EXPERIMENT_LOG.md" are the failure mode; fig 1
reads results.json directly and was never wrong.
