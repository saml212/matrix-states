# Sprint handoff — state at 2026-09-01 19:15 PDT, runbook for 2026-09-02

For the paper agent after context compaction, and for Sam. Source of
record remains `EXPERIMENT_LOG.md` (2026-09-01 #3–#6) and
`papers/TRIAGE_2026-09-01.md`. Board: https://claude.ai/code/artifact/3ccf9fc2-367d-49e2-93bc-c9ac4700c531
(update in place; never create a second one).

## 0. Access (verified)
- Air is a window. Work from the Mac mini: `ssh -o BatchMode=yes samuels-mac-mini '…'`
  (Tailscale). Box: nest `ssh -o BatchMode=yes youthful-indigo-turkey "…"`
  inside it (alias also `brev-h100`; cloudflared tunnel; brev CLI logged out
  but irrelevant). Mini repo `~/Experiments/learned-representations` on
  main; keep it `git pull --ff-only`'d. Mini has tectonic + `.venv` with
  torch/matplotlib. Local Bash commands containing `python3 …py` need the
  `DRY_RUN_BYPASS=1` prefix or the pre-train-gate hook blocks them. Commits
  of new .md files need `CLEAN_BYPASS=1` (documented escape hatch).
- Remote is `git@github.com:saml212/learned-representations.git` (GitHub
  says it moved to `saml212/matrix-states`; pushes still work). Push with
  `git push origin HEAD:main` from this worktree (branch
  claude/matrix-state-paper-triage-82d87d).
- HF: dataset `Slamin/ncr-scaling-artifacts` PUBLIC, 3.28 TB verified. A
  write token is at `~/.hf_token` on the box (placed 18:13 PDT; uploader
  `hf_preserve.py` pid 1891220 authenticated and streaming grace files).
  That token passed through chat: Sam revokes it Wed afternoon AFTER the
  grace checkpoints upload (check `/ephemeral/hf_preserve/preserve.log`).

## 1. Box queue of record (all PDT; box clock is UTC = PDT+7)
Workers: 8 tmux `queue_worker_g0..7`, claim `~/queue/pending` by `ls|sort`,
run `CUDA_VISIBLE_DEVICES=$GPU bash -c "$cmd"` then `bash -c "$vcheck"`;
pass → `completed/`, fail → `failed/`. Logs `~/queue/logs/<id>.log`.
1. 0470–0477 K=16 grace PRIMARY (1.31B, seeds 9–16) — running, end ≈23:18 Tue.
2. 0477a–d re-measures at base seed 31337 of K=16 excursions (primary
   s5/s6/s7, compB s6); 0477e–j battery scoring of the six never-scored
   392M K=16 thickening cells (s3–5 both recipes). Eval-only, ~2 min each.
3. 0478–0485 K=16 grace compB (incl. 0479 that the campaign session had
   dropped) — ≈23:25 → ≈07:12 Wed.
4. 0599 recall CUDA probe (300 steps of C2, own dirs) — runs ≈07:15 Wed.
5. 0640–0645 embedding-ablation phase-A probes (500 steps, 3 arms × S/M).
6. GATED by `tmux stage_after_probe` (`~/stage_after_probe.sh`): when
   0599 lands in completed/, runs `~/chapter2/deltanet_rd/strengthen_reprice.py
   --probe-dir ~/chapter2/deltanet_rd/results/h2h_rung1/strengthen_probe
   --joblog ~/queue/logs/0599_….log`; on `REPRICE: PASS` copies the 27
   specs from `~/strengthen_staging/specs/` (0600–0620, 0624–0629) into
   pending; else writes `~/strengthen_staging/STOP.*`. Log
   `~/strengthen_staging/stage_after_probe.log`. Kill thresholds (s/step):
   C2@60k >0.720, C2@20k >0.900, C1@60k >0.240, C1@20k >0.270, C0@60k
   >0.180; if r_C2 > 0.51 regenerate TIMEOUT_HOURS = 2× re-priced before
   staging (generator `h2h_strengthen_specs_gen.py`, TRIM keeps 0621–0623
   in `strengthen_specs_deferred/`, conditional re-add iff mean
   acc_A(C2,3e-4,20k) > mean acc_A(C2,1e-3,20k)).
7. GATED by `tmux stage_phase_b` (`~/stage_phase_b.sh`): when 0640–0645
   are all in completed/, runs `~/embed_ablation/embed_ablation_rd.py
   --check-admission --probe-results-dir ~/embed_ablation/results/probes
   --intended-steps 2000 --intended-batch 64`; exit 0 → copies the 24
   phase-B specs from `~/embed_staging/specs/` (0646–0669) into pending;
   else `~/embed_staging/STOP.admission` (then re-derive STEPS_B per
   `matrix-thinking/embed_ablation_specs/phase_B_seeded/README.md`).
8. 0900–0917 fallback training (392M K=16 seeds 6–8 both recipes;
   98M K=16 seeds 3–8 both recipes) + 0950–0967 their battery scoring.
   Claim only when nothing earlier is pending. ≈23 GPU-h.
Projected drain ≈18:00 Wed if gates pass, ≈10:30 Wed if both fail.
Fri Sep 5 = Brev offboarding; nothing cited depends on the box after
today's pulls, except grace checkpoints (HF upload in progress).

## 2. Wednesday morning runbook (agent, from 07:00)
1. Harvest: from the mini, rsync `~/ncr_scaleaxis/results/remeasure1b_*`,
   `sweep_scaleaxis392m_thicken_K16_*`, `sweep1b_*grace*` (after scoring),
   `/ephemeral/scaleaxis1b/results/*grace*` (JSON+log, not .pt) into
   `experiment-runs/2026-09-02_grace_harvest/` (repo + SSD). Score grace
   cells with `bash ~/ncr_scaleaxis/score_grace.sh <gpu>` after compB
   completes (~07:15) — one GPU, ~22 min for 16 cells; run it on a GPU that
   just freed, or after the gated waves if all GPUs are busy.
2. Adjudicate the re-measures: an excursion that collapses into band
   [0.0171,0.1079] at K=16 is an outlier; one that reproduces counts.
   Update the K=16 h=1 three-scale table (98M / 392M n=6→9 / 1.31B n=9→17)
   — descriptive only; TEST-W/TEST-X/scaling-law verdicts stay at n=3.
3. Check both watcher logs; if a STOP.* exists, read the reason and decide
   (recall: regenerate timeouts or trim further; embedding: re-derive
   STEPS_B). Confirm `~/queue/failed` is empty.
4. Update the board.

## 3. Wednesday with Sam
- 09:00 ICML correction sign-off (memo
  `matrix-thinking/submissions/icml-mi-workshop-2026/CORRECTION_2026-09-01.md`):
  decision A (small gap −0.78 three-seed means [recommended in text] vs
  −2.86 single-seed [footnote]); decision B (eval-set sentence: best of 25
  per-epoch evals on the first 128 of 500 ProsQA test problems). Then apply
  the edit table (abstract, §1, §3, §4, §5, §6, §7, generate_figures.py
  lines 93/94/110/126, metadata.md, poster, PAPER_READER_VIEW, log lines
  1371/1425/1433/9020 with a dated note, site page), regenerate fig2/fig3
  on the mini venv, rebuild package (tectonic clean-room), Sam checks the
  arXiv-compiled PDF, submit BEFORE 11:00 PDT (announces Thu). Metadata:
  cs.AI primary, cs.LG cross-list (both endorsed; do NOT add cs.CL — not
  endorsed), CC BY 4.0, comments = correction note from memo §4 + "Accepted
  at the ICML 2026 Mechanistic Interpretability Workshop (virtual poster).
  9 pages." Authors "Samuel Larson (Pebble ML)". Then post an erratum
  comment on OpenReview forum Spof4PusVI. Package v2 (affiliation fixed,
  numbers NOT yet fixed) is at `~/Desktop/arxiv-572-submission-v2.tar.gz`;
  v3 must be rebuilt after the number edits.
- Rest of day: causal rank law. Merge `papers/unireps-ea` into
  `papers/neurreps-ea` (Candidate A title "The Rank the Task Demands…"),
  named (de-anonymized) rebuild without the draftwatermark/NeurReps
  track stamp, restore the cut self-citation, whiteboard defense (write a
  DEFENSE_BRIEF like mstar's), Will's recompute queue for its 14 rows +
  `rank-recruitment-ws`'s 12 rows. Target submit Mon Sep 8.
- Thu morning: recall defense (`papers/mstar-colm-er/DEFENSE_BRIEF_2026-09-01.md`)
  with the strengthening-sweep result; then rewrite its comparison section
  under the headline rule (outcome A/B/C from `h2h_strengthen_rd.py --harvest`).
- Will arrives Thu eve; his queue is TRIAGE §4. Named rebuilds of every
  batch-1 draft are his first logistics task (all current PDFs are
  anonymized "Under review at COLM 2026" builds — never share them).

## 4. Decisions of record today (do not relitigate)
Headline rule; NeurReps not submitted → arXiv; merge neurreps/unireps;
fold kwall into the NCR scale tree; cluster 4 strengthen-or-drop via the
embedding ablation; HF public; "Samuel Larson, Pebble ML" everywhere;
solo author unless Will earns it; ORCID 0009-0006-1253-9187 linked;
arXiv account samuellarson endorsed cs.AI+cs.LG (Ayush Noori); CC BY;
trim the recall sweep to 27 cells; the grace wave's breach-rate statistic
is post-hoc (log #3) and the K=16 h=1 sentence is the defensible one there.

## 5. Open items nobody owns yet
- humor paper (rank 9): tree at `~/Experiments/good-humored/paper/reward-stack/`
  on the mini (committed 20bba50 in that repo); C14/C18 need live
  GLM-4.5-Air audience inference (~1 GPU-h on the box) — decide before Fri.
- measurement-ws: two cited logs missing from the archive.
- Barnfield 2026 subtitle wrong in rank-recruitment refs.bib (Will).
- FINAL_TABLES regen with correct labels/ref constants before any 1.31B
  TEST-W number is cited (aggregate_1b.py lines 149-153 / 274).
- capped_M2 columns mislabeled at C1/C2 in the strengthening harvest (report-only).
