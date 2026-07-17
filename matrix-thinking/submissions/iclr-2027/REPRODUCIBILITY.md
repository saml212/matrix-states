# Reproducibility Plan — ICLR 2027 Submission

**Status: DRAFT — round 1, companion to `NARRATIVE.md`.**

## 1. What gets released

### 1.1 Code (the harness, not a cleaned-up reimplementation)

Release the actual harness that produced every number in the paper, not a
distilled/rewritten version — the pre-registration/attack/audit trail only
has evidentiary value if it points at the code that was actually run.

- `matrix-thinking/deltanet_rd/{model_rd.py, run_deltanet_rd.py,
  grammar_rd.py, rank_utils.py, f15_lm_checkpoint.py,
  analyze_eval_truncation_rd.py, geo3_simulator.py, extract_wavef.py,
  summarize_wavef.py}` — the real-data DeltaNet exactness harness, including
  the F-geo-3 orthogonalization implementation and its pre-registered
  drift simulator (importable, CPU-smoke-tested — the same object cited
  by name in the paper's methodology paragraph, not a post-hoc
  reconstruction).
- `matrix-thinking/deltanet/{model_dn.py, deltanet_core.py, run_deltanet.py,
  run_deltanet_sweep.py, task_dn.py, rank_utils.py, probe_trunc.py,
  analyze_eval_truncation.py}` — the synthetic-harness DeltaNet causal-rank
  code (the Fig. 1 left panel / razor-cliff source).
- `analysis_lm_w2.py` — the Wave 2 (Waves C+D) corpus-contrast analysis
  script, CPU-only, no GPU dependency to re-run against archived JSONs.
- The three design documents themselves — `DELTANET_CAUSAL_RANK_DESIGN.md`,
  `DELTANET_REALDATA_DESIGN.md`, `DELTANET_RD_EXACTNESS_DESIGN.md` — released
  as supplementary material, unredacted (see §3 below on why this is the
  actual novelty of the reproducibility artifact, not a formality).

### 1.2 Data

- Task grammar/probe generation code (`grammar_rd.py`) is released in full
  — the synthetic binding task is entirely procedural, no licensing
  concern, exact regeneration is a code property not a data-hosting one.
- Real-text pretraining data: GPT-2-tokenized OpenR1-Math and WikiText-103
  spans. Release the exact tokenization/windowing script and the `meta.json`
  provenance record (`<data-root>/data/
  {reasoning,wikitext103_tokenized}/meta.json` fields, per
  `DELTANET_REALDATA_DESIGN.md` §13) rather than re-hosting the raw token
  arrays — both source corpora are already public (OpenR1-Math on HF,
  WikiText-103 standard) so the paper points to the public source + our
  exact preprocessing, not a redistribution.
- Result JSONs: every archived run JSON under `experiment-runs/
  {2026-07-02_deltanet_waves, 2026-07-03_deltanet_rd_waves,
  2026-07-04_lm_rd_wave2}/` is released as-is — these are the literal
  files every table/figure number in the paper was read from
  (`checkpoints[-1]`, never `AGGREGATE.json`/`SUMMARY.txt`, per the
  standing operational note in `DELTANET_REALDATA_DESIGN.md` §16.6 — state
  this explicitly in the artifact README so a re-user doesn't quote the
  stale aggregator fields by mistake).
- Model checkpoints (`.pt` files) are **NOT** released — per house
  convention (too large, `experiment-runs/` archives JSONs + logs only,
  checkpoints stay on the training box/SSD) — but every number the paper
  reports is derivable from the released JSONs alone (they carry
  per-checkpoint diagnostics, not just final-step summaries), so this is
  not a reproducibility gap for the paper's own claims, only for
  independent extensions.

### 1.3 What is explicitly withheld or simplified

- No author-identifying material (session transcripts, internal Slack-
  style commentary, dated internal filenames referencing hardware account
  names) — scrub the training cluster's hostname/account alias and any
  other environment-identifying strings (local disk mount paths, personal
  machine names) from the anonymized submission copy; keep them in the
  camera-ready.
- The design documents are released with their FULL revision history
  (Rev 1/2/3, attack-round findings, verification-round findings) rather
  than a cleaned "final" version — this is a deliberate choice (§3), not
  an oversight; a reviewer who wants to sanity-check that a negative
  result (Wave F) was not quietly re-run until it looked better can
  verify that directly against the dated revision log rather than take
  the paper's word for it.

## 2. The pre-registration / audit trail AS an artifact

This is the one non-standard element of this paper's reproducibility
package, and it is treated as a first-class release artifact, not
supplementary color:

- **Every hypothesis, success bar, and admission criterion cited in the
  paper was written down and dated BEFORE the corresponding data
  existed.** The design documents carry this literally — dated revision
  markers, "finding → change" maps between attack rounds, and (for
  F-geo-3 specifically) a committed, importable simulator
  (`geo3_simulator.py`) whose `launch_read()` function is the actual
  go/no-go gate that authorized the Wave 1 GPU spend, not a narrative
  description of one.
- **Two independent attack rounds are preserved per major design
  decision**, not just the final "clean" version — e.g. F-geo-3's Rev A
  (attack round, found the cross-episode-drift failure mode) and Rev B
  (round-2 verification, found the launch-read gate cited an undefined
  instrument and the substitute admission stack lacked a task-performance
  floor) are both kept in the released document, with their finding→change
  maps intact (`DELTANET_RD_EXACTNESS_DESIGN.md` §14.11/§14.13).
- **The negative results are released with exactly the same rigor as the
  positive ones** — the Wave F soft-arm failure (§15 of
  `DELTANET_RD_EXACTNESS_DESIGN.md`) is not summarized away; its full
  18-cell table, the pre-registered bars it missed, and the honest
  "demonstration-claim tier: false, for all three arms" verdict are
  released verbatim, because the paper's own methodology claim (a
  mechanism-matched fix, not a fix found by trying things until one
  worked) depends on the reader being able to verify that F-geo-1/F-geo-2
  were run to completion under the anti-Goodhart cap before F-geo-3 was
  even designed.
- **Purpose, stated plainly (avoid self-congratulation in the actual
  paper text, but state it here where it belongs):** the artifact lets an
  independent reader check whether a given headline number in the paper
  passed a criterion that was fixed before the number existed, or whether
  the criterion was chosen after seeing the number. For every number in
  the claim-tier cheat sheet (`NARRATIVE.md` §4), the corresponding
  pre-registration date and criterion are traceable in the released
  documents.

## 3. Known reproducibility caveats (state these, do not omit them)

1. **Hardware-dependent kernel path.** The real-data results depend on
   `fla-org/flash-linear-attention`'s `chunk_delta_rule` (bf16-only,
   production kernel) and the pre-registered `svd_lowrank` fallback for
   force-rank arms (chosen after `eigh` crashed on real states,
   `DELTANET_REALDATA_DESIGN.md` §14.1) — pin the exact package version
   (`0.5.1`, per §13) in the release, since kernel-path/precision was
   itself named as an untested confound (mechanism (f),
   `DELTANET_RD_EXACTNESS_DESIGN.md` §2) that this paper does not fully
   resolve.
2. **The F-geo-3 substitute admission-stack caveat (§14.10) travels into
   the released artifact, not just the paper text.** The paper carries it
   at footnote tier (the K=32 `n_iter=20` escalation cells passed every
   substitute criterion including zero fallback steps, and the
   fallback-irrelevance check showed the criterion was conservative, not
   explanatory), but any downstream user re-running F-geo-3 and comparing
   its "N/3 admissible seeds" against a different arm's admission count
   must still be pointed at the two stacks' differing filters and
   realized pass rates — the artifact README states this explicitly, not
   just the paper.
3. **Small, uneven per-checkpoint sample sizes in Wave 2's Q1/Q4
   analyses** (e.g. openr1 rank-probe n drops as low as 3 windows at one
   checkpoint, `DELTANET_REALDATA_DESIGN.md` §19.2) — released alongside
   the full per-checkpoint counts so a reader can judge statistical
   weight themselves rather than trust only the pooled headline number.
4. **Seed-count asymmetry in the eval-truncation staircase aggregation**
   (K=16 n=10, K=32 n=7, K=8/K=24 n=2 each, §17.3) — the K=32 ceiling
   number underlying Fig. 1/Fig. 2's most dramatic contrast rests on the
   smallest-n cell reported; state this in the artifact and in the
   paper's limitations, not just here.

## 4. Release mechanics

- Anonymous code link at submission time: `https://anonymous.4open.science/`
  mirror of the harness + design docs (house convention, per
  `matrix-thinking/PAPER_WRITER_BRIEF.md`), de-identified per §1.3.
  De-anonymized GitHub link substituted at camera-ready.
- Result JSONs and analysis scripts ship as a single archive alongside the
  anonymous code link (small enough — the largest single wave archive,
  Wave F, is 37MB — no external hosting needed).
- A top-level `REPRO.md` in the released archive (distinct from this
  planning document) maps each paper figure/table number to the exact
  archive subdirectory and extraction script, mirroring the "data source"
  column already drafted in `NARRATIVE.md` §2's figure plan — write this
  mapping once, in the actual figure-generation scripts' docstrings, so
  the paper and the artifact cannot drift apart.
