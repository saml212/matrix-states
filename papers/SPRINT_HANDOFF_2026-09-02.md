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

---

# Handoff addendum — state at 2026-09-03 ~13:30 PDT, runbook for Fri 2026-09-04

Read this addendum FIRST after compaction; the sections above are Wed's
state and are superseded where they conflict. Log entries of record:
`EXPERIMENT_LOG.md` 2026-09-02 #1–#3 and 2026-09-03 #1–#5. Latest commit
95a9603 (pushed to `origin main`; mini checkout in sync).

## A. What is DONE (do not redo)

1. **ICML paper on arXiv: submit/8024707, submitted Wed 12:00 PDT**, status
   "processing" when last seen; announces Thu Sep 3 evening ET (Fri
   mailing); arXiv ID arrives by email to samlarson16@gmail.com. cs.LG
   primary (classifier recommendation accepted), cs.AI cross-list, CC BY,
   comments field carries the workshop, the OpenReview link
   (forum Spof4PusVI) and the 400-char correction note. Corrections applied
   everywhere per `matrix-thinking/submissions/icml-mi-workshop-2026/
   CORRECTION_2026-09-01.md` (decisions A: 3-seed-mean gap −0.78 with
   single-seed −2.86 in the same sentence, fig 3 small bar = 3-seed mean;
   B: eval-set sentence in §2; C: not added). Package v4 on both Desktops.
   NOT yet done for ICML: OpenReview erratum (text in the memo §4, updated
   for decision A; needs the arXiv ID) and the site page
   `pebble-ai-site/findings/matrix-codi-rank-blindness.html` (hand-drawn
   SVG; numbers at lines 492, 574–611, 711, 719).
2. **Rank law paper** `papers/neurreps-ea/arxiv-v2/` (named, 11 pp, clean
   tectonic + pdflatex builds; PDF `ranklaw-arxiv-v2.pdf` on both Desktops):
   merged neurreps+unireps (equivalence section, instrument appendix);
   Samuel Larson / Pebble ML / samlarson@pebbleml.com; neurips[final]
   template; razor sufficiency 5/5 at n=4 (§1.36b: S4/A5/A6; §1.36c: S5 at
   a 20k pin, 8k failure disclosed in ONE Appendix F sentence by Sam's
   decision, override recorded in CAPABILITY_SEPARATION_DESIGN.md §1.36c);
   whole-matrix rank Appendix C at n=4 (N15/N17); stable-rank disclosure
   (N18); novelty gate PASSED twice (Sep 2 two adversarial sweeps + Sep 3
   arXiv re-check; verdicts in `papers/neurreps-ea/brief.md`), DeltaProduct
   §5.2 named as precursor, 12 new citations. Metadata for the form:
   `papers/neurreps-ea/arxiv/metadata.md` "v2" section (abstract must be
   re-copied from arxiv-v2/main.tex because it changed after that section
   was written; comments "8 pages…" is stale → count pages in the final
   PDF). Evidence rows N1–N19 in brief.md.
3. **Companion paper** `papers/rank-recruitment-ws/arxiv-v1/` (named, 6 pp,
   clean; PDF `rankrecruit-arxiv-v1.pdf` on both Desktops); cites the rank
   law by title (`larson2026ranklaw`, "posted concurrently"); metadata in
   `papers/rank-recruitment-ws/venue-requirements.md` "arXiv named build
   v1" section. Two stale EA sentences corrected.
4. **Recall paper** `papers/mstar-colm-er/main-arxiv.tex` (named preprint
   variant; PDF `recall-arxiv-v1.pdf` on both Desktops, 11 pp): baseline-
   strengthening paragraph in §3 (27-cell sweep, OUTCOME A, C12), abstract
   clause, limitations widened. Defense brief
   `papers/mstar-colm-er/DEFENSE_BRIEF_2026-09-01.md` (Q2's answer now
   includes the 27-cell sweep). No arXiv metadata file yet → build the
   abstract from sections/00_abstract.tex; comments "11 pages, 3 figures";
   cs.LG primary, cs.AI cross-list; CC BY.
5. **Box work COMPLETE, box RELEASED** (all GPUs 0 MiB, no processes).
   Archives in repo + SSD: `experiment-runs/2026-09-03_recall_strengthen/`
   (27 cells + verdict), `2026-09-03_m3fix_ext4/` (72 cells), `2026-09-03_
   m3fix_s5_20k/` (24 cells), `2026-09-03_embed_ablation/` (24 cells + both
   probe sets + harvest). HF: all checkpoints uploaded (uploader finished
   twice); **Sam should revoke the HF token that transited chat** (may
   already have). Brev offboarding Fri Sep 5 as scheduled or earlier by
   Sam's email to the admin.
6. **Cluster 4 (matrix embeddings) DROPPED**: phase B ran under Sam's
   override of the twice-failed admission gate; STRENGTHEN-01 DROP,
   STRENGTHEN-04 DROP (dense flat-P beats the matrix embedding at both
   sizes). Recorded in EMBEDDING_ABLATION_DESIGN.md §9, TRIAGE §8, log #4.
7. **Humor paper C14/C18** live repro NOT reproduced (3 serve attempts;
   launch-env drift; log #5). Rows stay `[ARTIFACT MISSING]`; artifacts at
   mini `~/Experiments/good-humored/paper/reward-stack/verification/
   repro_2026-09-03/` (uncommitted in that repo).
8. **Study page** for Sam: "Rank Law Whiteboard" artifact
   https://claude.ai/code/artifact/06747d63-20d8-4da9-a495-ebcf2e2c3e40
   (source `scratchpad/ranklaw-whiteboard.html`; republish same path).
   Board artifact unchanged URL (see top of this file); source
   `scratchpad/sprint-board.html`.
9. **GPT scrub agent**: prompt at `~/GPT_SCRUB_PROMPT.md` on the mini and
   `GPT_SCRUB_PROMPT.md` on both Desktops; kickoff text given to Sam (clone
   to `~/scrub-clone`, use repo venv python + tectonic, push
   `scrub/<paper>` branches, rank law first). UNKNOWN whether Sam started
   it. Review procedure for its branches: pull the branch, diff every
   number/citation/evidence tag vs `main` (the prompt's Step D), render
   figures, read SCRUB_LOG.md, merge only what passes; never let it touch
   main directly.

## B. Friday runbook (agent)

1. 07:30: `git pull` here and on the mini; check `git branch -r | grep
   scrub` for the scrub agent's branches; review per A.9 if present (rank
   law first). If the scrub is not done, submit WITHOUT it (prose polish is
   not worth missing the Sunday announce) and apply it in a v2.
2. arXiv re-check (2 min): arXiv search, announced-date order, queries
   "state tracking", "word problem" group, "fast weight"; anything new on
   learned state rank/dimension on group word problems → stop and
   adjudicate before posting. Record the result in
   `papers/neurreps-ea/brief.md` (append to the re-check block).
3. Clean-room builds (mini, tectonic ×3) of all three trees; confirm page
   counts, 0 undefined refs, no "Under Review/Anonymous/NeurReps"; refresh
   the three Desktop PDFs and the three tarballs (`ranklaw-arxiv-v2.tar.gz`,
   `rankrecruit-arxiv-v1.tar.gz`, `mstar` needs a tarball: main-arxiv.tex,
   main-arxiv.bbl (from a build), refs.bib, *.sty, *.bst,
   math_commands.tex, sections/, figures/ incl. tables_generated.tex).
   Strip macOS `._` files: `tar --no-xattrs` or `COPYFILE_DISABLE=1 tar`.
4. Sam reads all three PDFs as a reviewer. Then submission ×3 in Sam's
   Chrome (Claude in Chrome tools; load with ToolSearch
   `select:mcp__claude-in-chrome__tabs_context_mcp,navigate,computer,
   read_page,find,form_input,file_upload,get_page_text`), same flow as
   Wed: arxiv.org/user → START NEW SUBMISSION → Start (Sam ticks the two
   agreement boxes; agent sets author radio, CC BY, cs + subject) → Add
   Files (file_upload the tarball from the scratchpad path, Upload, Check
   Files) → Review Files (UNCHECK the suggested deletion of main.bbl; keep
   bbl/bst/bib) → Accept and Continue → Process (verify SUCCEEDED, page
   count) → Metadata (title, "Samuel Larson (Pebble ML)", abstract plain
   text, comments ≤400 chars) → Preview (classifier may suggest cs.LG:
   accept; add cs.AI cross-list if not auto-added; Sam clicks View PDF,
   refresh, Sam clicks Submit). Order: rank law, companion, recall. All
   before 11:00 PDT (14:00 ET) → announce Sun evening ET. Comments fields:
   rank law "N pages, 2 figures. Companion paper: When the Gradient Sees
   Rank (posted concurrently)."; companion "6 pages, 2 figures. Companion
   paper: The Rank the Task Demands (posted concurrently)."; recall
   "11 pages, 3 figures." Note both paper submit-IDs.
5. After the ICML arXiv ID email: post the OpenReview erratum on forum
   Spof4PusVI (text in CORRECTION memo §4 + decision A), and fix the site
   page (A.1). Then a v2 of rank law/companion adding each other's arXiv
   IDs (optional, later).
6. Log each submission in EXPERIMENT_LOG.md (2026-09-04 #n), update the
   board, update memory `arxiv-sprint-state-2026-09-01.md`.

## C. Open decisions for Sam (not blocking Friday)

- Flagship T1 wording flag (`papers/flagship/brief.md`): the rank law's
  substrate is the Stage-1 encoder, not a delta-rule memory. ICLR 2027:
  abstract Sep 18, paper Sep 25 AoE; full gauntlet after batch 1 posts.
- Venue for the rank law after arXiv: watch unireps.org/2026 daily for
  the EA deadline (placeholder Sep 25); else ICLR 2027 workshops (Feb).
  Record in `papers/neurreps-ea/VENUE_REQUIREMENTS.md`.
- Will (arrives Thu eve): queue TRIAGE §4 + the updated evidence tables
  (N15–N19, C12); optional humor C14/C18 retry with the allreduce fusion
  disabled (see log #5).
- Rank law whiteboard round with Sam (never happened Thu; defense brief
  `papers/neurreps-ea/DEFENSE_BRIEF_2026-09-02.md`, Q14 = the S5 re-run
  question, control 8 = stable rank).

## D. Gotchas learned this week (keep)

- Local Bash: any command string containing `python3 …py` is blocked by
  the pre-train hook → prefix `DRY_RUN_BYPASS=1`; `--include=*.py` also
  trips it. Heredoc `python3 - <<EOF` is fine with the prefix.
- Commits need the clean sentinel: run `python3 .claude/skills/clean/
  audit.py --scope staged` in its OWN Bash call, then commit in the next
  (the pre-commit hook checks before the command runs). New .md files →
  `CLEAN_BYPASS=1` (user-sanctioned only).
- cwd drifts after `cd` inside a Bash call; use absolute paths.
- zsh: `echo ====X` fails ("not found"); avoid leading `=` in echo args.
- Nested ssh quoting: write scripts locally, scp via the mini, or pipe
  `ssh … 'ssh … "python3 -"' < script.py`.
- Air has pdflatex at /Library/TeX/texbin but no matplotlib and lacks
  environ.sty; mini has tectonic + full venv. Clean-room checks: tectonic
  on the mini; arXiv itself uses pdflatex (page counts can differ from
  tectonic; the ICML paper was 10 pp tectonic / 9 pp pdflatex).
- macOS tar adds `._` files (arXiv strips them, harmless).
- arXiv Comments ≤400 chars; arXiv v1.5 suggests deleting main.bbl (keep
  it); arXiv assigns IDs at announcement, so v1 cross-citations are by
  title.
- Artifact republish: if refused as stale, `action: read` the URL, then
  publish again from the same file path.
- Claude in Chrome: `form_input` works for text/select/radio; some submit
  buttons need a coordinate click; refs go stale after navigation.
