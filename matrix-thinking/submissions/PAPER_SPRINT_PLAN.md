# Paper Sprint Plan — 2026-07-07

**Status: campaign-closing planning memo.** Written after the key-anchoring
program's final closure (`matrix-thinking/KEY_ANCHORING_DESIGN.md` §10.14,
§11.12) and `matrix-thinking/submissions/iclr-2027/NARRATIVE.md` round 5.
The anchoring program is DONE; the only experiment still running anywhere
in the project is Track C rung-3 (1.31B params, ETA imminent). This is a
writing-only planning document — no GPU, no code changes. Everything below
is either (a) a concrete page-cut plan for the already-drafted workshop
paper, (b) an exact data-file-to-figure mapping so figure regeneration is
mechanical, (c) a section skeleton for the still-unwritten ICLR full paper,
or (d) a decision list for the user. Nothing here commits to any of it —
it is a plan to execute, not an executed plan.

---

## 1. Workshop paper (`neurips-ws-2026/`) — page-cut options, 10pp → 4-5pp

### Current state, measured directly

`main.pdf` (single-blind build) compiles to **13 pages total**: title +
abstract on p.1, body sections 1–9 spanning p.1–p.10 (roughly: intro/setup
p.1–2, Task D p.2–4, Task E p.4–6, frontier p.6–7, related
work/discussion/conclusion/reproducibility p.8–10), references p.11–12
(part), appendix p.12–13. This matches `VENUE_DECISION.md`'s own
`pdftotext`-measured estimate (body ≈10pp, refs ≈1.3pp, appendix ≈1.5pp).
Section line counts (`sections/*.tex`, a rough proxy for relative weight,
not page count directly since density varies):

| Section | Lines | Approx. page range |
|---|---|---|
| 01_intro | 102 | p.1–2 |
| 02_setup | 123 | p.2 |
| 03_task_d | 158 | p.2–4 |
| 04_task_e | 204 | p.4–6 |
| 05_frontier | 138 | p.6–7 |
| 06_related_work | 98 | p.8 |
| 07_discussion | 88 | p.8–9 |
| 08_conclusion | 28 | p.9–10 |
| 09_reproducibility | 66 | p.10 |
| 10_appendix | 137 | p.12–13 (after refs) |

The target venue (per `VENUE_DECISION.md`, still the PI's open call — see
§4 below) is almost certainly a **4pp Extended Abstract track** (NeurReps/
UniReps) or, as a long-shot parallel path, a **5pp Short track** (COLM
Actionable Interpretability, currently closed for this cycle but worth an
email). Either way this is a genuine ~60% cut, not a formatting pass.

### Three concrete cut strategies

**Strategy A — "Task D only, Task E as one paragraph."** Keep: intro
(trimmed to the two contribution bullets that matter — the provable rank
bound and its causal confirmation), setup (compressed to the model +
readout-pinning paragraph, cut the position-decomposition digression to one
sentence with a footnote), Task D in full (this is the paper's cleanest,
most citable result: `rank(Z)≥K` necessity + the razor-sharp force-rank
step). Cut: Task E's full mechanism (entity-subspace decomposition, the
rank-starved depth-decay prediction) down to 2-3 sentences reporting the
headline number only (4/5 seeds recover exactly to h=21); cut the frontier
section (§5, the d-scaling exactness degradation) to a single sentence in
the discussion citing it as "an open frontier, detailed in the full
version"; cut related work to the 2-3 most load-bearing citations
(Nichani/Lee/Bietti for the argmax-vs-exact-recovery distinction; the
descriptive rank-measurement foils). **What this keeps:** the single
cleanest, most defensible causal result in the paper — a workshop reviewer
can verify the whole claim in one figure (force-rank staircase) and one
paragraph of setup. **What this drops:** the composition result (Task E)
and the exactness-frontier result (§5), which are scientifically
interesting but require more setup to land in 4pp. **Best for:** if the
committee/reviewers skew toward wanting one clean, verifiable, citable
number rather than a research narrative.

**Strategy B — "The whole arc, compressed uniformly."** Keep all three
results (Task D, Task E, frontier) but compress each to its single
headline number + one figure, cutting all secondary detail (the
entity-subspace decomposition's full derivation, the budget-independent-
plateau methodological aside, the concurrent DeltaNet teaser). Intro
becomes a single paragraph stating the three-part contribution list as
one sentence each. Setup becomes the readout-pinning paragraph only (this
is the single most load-bearing methodological point in the paper — it is
what makes the rank bound a *necessity* result rather than a construction
that argmax decoding could trivially satisfy — do not cut this). Related
work compresses to a related-work paragraph woven into the intro rather
than its own section (a common Extended-Abstract-track convention).
Discussion/conclusion merge into 2-3 sentences. **What this keeps:** the
full three-act structure (capacity → composition → frontier), which is
the paper's actual scientific arc and its strongest asset vs. the
companion ICML MI-workshop negative result. **What this drops:** enough
individual-result detail that a reviewer must trust the compressed claims
rather than verify them within the page budget (the appendix/arXiv
version becomes load-bearing for verification). **Best for:** if the venue
audience already knows the companion ICML paper and is looking for "what's
new since the negative result," where the three-part arc IS the pitch.

**Strategy C — "Figure-led, prose-minimal."** Keep 2 figures maximum (the
force-rank staircase from Task D, and the d-scaling exactness-degradation
curve from §5 — these are the two figures that are self-explanatory
without much surrounding prose) and write the entire body as captions +
a compressed narrative that explicitly references the figures rather than
re-deriving the numbers in prose. Cut Task E to a single sentence
("a compositional extension shows the same operator composes under
repeated self-application to depth h=21 in 4/5 seeds — full derivation in
the appendix/arXiv version"). This uses the Extended Abstract format's own
convention (many NeurReps/UniReps 4pp submissions are figure-dense,
prose-light). **What this keeps:** visual self-sufficiency — a reviewer
skimming figures gets the paper's shape without reading dense prose.
**What this drops:** the methodological framing (readout-pinning,
budget-independent-plateau naming) that distinguishes this paper's rigor
from a typical workshop submission — that framing is exactly what a
careful reviewer would want, and it is hard to convey in a caption.
**Best for:** if the actual reviewing bar is skim-speed (true for most
workshop tracks) and the arXiv/ICLR full version is where rigor-minded
readers will actually go.

**Recommendation (not a PI decision item, offered as a default):**
Strategy B. It preserves the three-act structure that is this paper's
actual novelty relative to the companion ICML negative result (Task
D/E/frontier as a connected arc, not three disconnected findings), and the
readout-pinning paragraph — this program's single most defensible
methodological point — survives intact in every version of Strategy B's
own cut. Strategy A is the safer fallback if the actual page count after a
Strategy-B pass still runs over; Strategy C is worth trying only if a
first Strategy-B draft is still too dense at 4pp.

### Mechanical note on the cut process

Whichever strategy is chosen, do the cut in `main.tex`/`sections/*.tex` as
its own editorial pass, not by writing new prose — every sentence surviving
the cut should be a subset (possibly re-ordered) of what already exists in
the 10-page body, since every claim in that body is already independently
audited and cited. New prose written under page pressure is exactly the
kind of thing that reintroduces an unverified claim; the discipline this
whole project has followed (audit before claiming) applies to editing too.

---

## 2. Figure regeneration list

Two families of figure need attention before either paper is camera-ready:
the workshop paper's existing 4 figures (already built, may need
re-checking after any body cut changes what's cited) and the ICLR paper's
11 figures (per `NARRATIVE.md` round 5 §2), several of which now have exact
new data sources from this closing period.

### 2a. Workshop paper (`neurips-ws-2026/figures/`) — status check, not a rebuild

Existing committed figures: `fig_taskd_forcerank.pdf`,
`fig_frontier_exactness.pdf`, `fig_eval_trunc_staircase.pdf`,
`fig_taske_depth.pdf`, built by `figures/make_figures.py` from archived
experiment JSONs (Task D/E/frontier waves, all pre-dating this closing
period — none of these depend on the key-anchoring program, the Track C
ladder, or Track B, which are all ICLR-paper material). **No regeneration
needed unless the page-cut pass (§1) drops a figure** — if Strategy A or C
drops Task E's or the frontier's figure, remove the corresponding
`\includegraphics` call and the file can stay committed but unused (do not
delete the PDF; the appendix/arXiv version may still reference it).

### 2b. ICLR paper — exact data-file → figure mappings (11 figures, per `NARRATIVE.md` round 5 §2)

This is the actionable list: which figures can be generated right now from
data already on disk, and exactly which files feed them.

| Fig | What it shows | Exact source file(s) | Status |
|---|---|---|---|
| 1 | Synthetic-vs-real eval-truncation staircase | `experiment-runs/2026-07-02_deltanet_waves/eval_trunc/`; `experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA}/` via `matrix-thinking/deltanet_rd/analyze_eval_truncation_rd.py` | Ready — all data archived, pre-dates this period |
| 2 | K-axis exactness frontier on real text | `experiment-runs/2026-07-03_deltanet_rd_waves/waveA/` | Ready |
| 3 | Rank recruited vs. K (capacity is not the issue) | `experiment-runs/2026-07-03_deltanet_rd_waves/waveA/` + `experiment-runs/2026-07-02_deltanet_waves/wave0/` | Ready |
| 4 | Embedding-source interpolation attractor | `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/` | Ready |
| 5 | The 2×2 existence proof | `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/` (arms i, iv, iii-β) + `.../exactness/wavegeo3/` (geo3-alone) | Ready |
| 6 | Soft fixes fail proportionally (ε^h law) | `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/waveF/` | Ready |
| 7 | F-geo-3 headline demonstration | `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/` (incl. n_iter=20 escalation cells) | Ready |
| 8 | Wave 2 corpus generality (appendix) | `experiment-runs/2026-07-04_lm_rd_wave2/{waveC,waveD}/` via `analysis_lm_w2.py` | Ready |
| 9 Panel A | Track C pure-scale attractor ladder | **Seed-0-only (archived):** `experiment-runs/2026-07-04_trackc_rung1/`, `2026-07-05_trackc_rung2/`, `2026-07-05_wave1ext/`. **Dense per-seed (NEW, this period):** `experiment-runs/2026-07-06_trajectory_probes/trajectories_tidy.{json,csv}` — 540 rows, all 6 seeds × both corpora × 4 cells (mixcontrol/wave1/wave1ext/wave2) at 7–35 checkpoints each. **Use `archived4_span_frac` as the plotted quantity** (cross-scale-comparable convention — raw `gram_deviation_mean` is not comparable across `d_state`). Validated exactly against archived harvests (66/66 finals ≤1e-6). **Missing: rung-3** (1.31B, still running) — add as a 4th point when it lands, same instrument (md5 `3fb0f80028477d0b1cefe468c81b1da4`), same convention | **Ready now for 3 points; 4th point (rung-3) pending — the only genuinely blocked figure in this list** |
| 9 Panel B | Production-scale measurement (Track D) | `experiment-runs/2026-07-04_track_d/` | Ready |
| 10 | Anchoring wave's complete arc (3-panel: behavioral gain, Outcome-C engagement scatter, constancy-suffices resolution) | Panel A: `experiment-runs/2026-07-05_keyanchor_wave/`, `2026-07-05_keyanchor_confirm/`, `2026-07-06_keyanchor_mech/`. Panel B: `2026-07-06_keyanchor_mech/` (same archive, `engaged_frac_v3`/`r_e` fields). Panel C: `experiment-runs/2026-07-07_keyanchor_e/` | **Ready — all data now archived** (this is new since round 4; previously blocked on the mechanism-tier and candidate-(e) waves, both now complete) |
| 11 | Capacity curve (NEW — the program's headline figure) | K=16/32: `experiment-runs/2026-07-06_keyanchor_mech/`. K=48: `experiment-runs/2026-07-07_keyanchor_k48/` (candidate-(d) cells, fresh reference arms, and `BANDS_PINNED_K48.json` for the λ=1 ceiling values) | **Ready — all data now archived**, this is the newest figure in the whole plan |
| fig:cliff | Capacity cliff, located (NEW, round 6 — supersedes Fig. 11 as the headline) | fig:cliff ← `experiment-runs/2026-07-06_keyanchor_cliff/` (4 new K's + `fit_cliff_curve_results.json`), plus the K=32/K=48 archived anchors already listed above | Ready — built and rendered this round (`iclr-2027/figures/make_fig_cliff.py`) |

**Bottom line: 10 of 11 ICLR figures are fully ready to generate right now
from already-archived data.** The sole blocker is Fig. 9 Panel A's fourth
point (Track C rung-3) — every other figure, including both of this
period's new figures (10 and 11), has its complete data already on disk
and verified. No figure-generation script currently exists for the ICLR
paper (there is no `iclr-2027/figures/` directory yet, unlike the workshop
paper's `neurips-ws-2026/figures/make_figures.py`) — building one, reusing
the workshop paper's `make_figures.py` as a structural template, is a
build task for whenever full-paper drafting starts, not before.

---

## 3. ICLR full-paper section skeleton, mapped to the round-5 narrative

This maps `NARRATIVE.md` round 5's content directly onto a standard ICLR
paper skeleton (page-budget-agnostic — ICLR full papers run ~9-10pp body
+ unlimited appendix, a much more generous budget than the workshop
track). Each row names the exact narrative section(s) it draws from so
drafting is a transcription task, not a re-derivation.

| Paper section | Draws from `NARRATIVE.md` round 5 | Notes |
|---|---|---|
| Abstract | §1 (full draft, ~11 sentences — needs a compression pass per round 5's own §10 item 6(i) flag) | Trim the anchoring-arc sentences (currently "Seventh" through "Ninth") to 2-3 total for the actual submission |
| §1 Introduction | §3's "§1 Introduction" block (9 contribution bullets) | Bullet 6-8 are new this round (anchoring rejection, constancy-suffices, capacity cliff) |
| §2 Related Work | §5 (full related-work map, 10 items) | Item 10a is the anchoring program's complete citation; item 10c (Stage-G) needs the explicit "NOT a DeltaNet finding" caveat preserved verbatim |
| §3 Setup and Theory | §3's "§3 Setup and Theory" block | Includes the "two primary K" framing, now extended to "three primary K's" logic (16/32/48) |
| §4 The Phenomenon | §3's "§4" block | Synthetic + real-data transfer, depth-amplification law, K=48 rider (now fully resolved by §9/11) |
| §5 Mechanism (Attribution) | §3's "§5" block, ending with the 2×2 and its closing anchoring-arc paragraph | This is where the anchoring program's full arc now lives as the section's *closing* move, not a forward-pointer |
| §6 Soft Fixes Fail | §3's "§6" block | Unchanged from round 3/4 |
| §7 The Fix (F-geo-3) | §3's "§7" block, including the full 4-wave anchoring arc as the section's final paragraph | The anchoring program's build/attack/run history belongs here in full |
| §8 Results | §3's "§8" block | Now includes the anchoring program's results as a completed sub-section (behavioral gain → rejection → ablation → capacity boundary), not a placeholder |
| §9 Discussion/Limitations | §6 (9 numbered limitation items, several rewritten this round) | Items 4, 5, 8, 9 are the ones most changed by this round's closure |
| §10 Conclusion | §3's "§10" block | Restates the four-part anchoring resolution explicitly |
| §11 Reproducibility | Pointer to `iclr-2027/REPRODUCIBILITY.md` (162 lines, already exists — checked directly this session) | Needs a currency check against the round-5 material (candidate (e), K=48), not a from-scratch write |
| Appendix | Fig. 8/9 Panel B (Track D), the Rev-6 rejected-rescore full writeup, Stage-G's full H_e manifest, M²RNN citation-verification note | Standard "generality + honesty" appendix material per round 3/4/5's own convention |

**Reviewer pre-mortem (§7 of `NARRATIVE.md`, 16 items) and the methodology
weave-in (§8)** transcribe directly into the paper's own response-to-reviews
prep and the "framing notes" a co-author/editor would use while drafting
§3/§7/§11 — these are process documents for the drafting phase, not
paper sections themselves.

**Correction, checked directly this session — a full section skeleton
already exists, and it predates this closing period's material.**
`iclr-2027/sections/` already has all 12 body/appendix files drafted
(`00_abstract_placeholder.tex` through `11_reproducibility.tex`, plus
`A1_claim_tier_appendix.tex`/`A2_full_tables_appendix.tex`, 1002 lines
total across `sections/` + `main.tex`), and `iclr-2027/REPRODUCIBILITY.md`
(162 lines) already exists too — so the "§11 Reproducibility" and
appendix rows in the table above are not gaps, they are **update
targets**. Checked directly: `04_phenomenon.tex`, `08_results.tex`,
`09_discussion_limitations.tex`, and `10_conclusion.tex` already mention
anchoring, but only as forward-looking future work ("EMA-anchored or
identity-registered orthogonalization" as a *proposed* follow-on) — i.e.
these four files were drafted before or during the anchoring program's
early stages and do **not** yet reflect any of its actual four-wave
outcome (behavioral replication, Outcome C, the candidate-(e) ablation, or
the K=48 capacity cliff). **This is the single most concrete, immediately
actionable drafting task this memo can point to:** the next drafting pass
should specifically update these four files (and, per the mapping table
above, `05_mechanism.tex` and `07_the_fix.tex`, which should also carry
the anchoring arc's closing paragraphs) against `NARRATIVE.md` round 5's
corresponding sections, rather than starting any section from a blank
page. A full pre-drafting step (recommended, not performed here since
this memo is planning-only): diff each of the 12 existing `sections/*.tex`
files against its mapped `NARRATIVE.md` round-5 section to find every
place still reflecting round 3/4-era material.

---

## 4. PI decision list

Four decisions, each with a concrete recommendation and the reasoning
behind it. None of these have been acted on — no submissions, no emails,
no author-block edits.

### Decision 1 — Venue (workshop paper)

**Recommendation: NeurIPS 2026 NeurReps or UniReps, Extended Abstract
track — hold the draft ready and submit the moment the CFP opens (expected
~2026-07-11 accepted-workshop list, CFPs shortly after, historically
late-Aug deadlines).**

Rationale: this is the only candidate in `VENUE_DECISION.md`'s own survey
that is simultaneously (a) still open for this cycle, (b) confirmed
non-archival, and (c) confirmed dual-submission-friendly (explicitly
permits the planned ICLR 2027 full-paper submission on the same arc
without burning novelty). The two alternatives surveyed (COLM Actionable
Interpretability, the Mechanistic Interpretability Workshop) are both
administratively closed for this cycle — COLM's extended deadline passed
2026-06-24, and the MI workshop's alternating host-conference pattern puts
its next instance at NeurIPS 2027, too late to matter here. The parallel,
low-cost move `VENUE_DECISION.md` already recommends (emailing COLM
organizers about a late add) is still worth doing given its best-worded
dual-submission policy of anything surveyed, but it should not be treated
as the primary path — it is a hedge, not a plan.

**What's changed since `VENUE_DECISION.md` was written (2026-07-03) that
bears on this decision:** nothing about the venue landscape itself has
changed, but the underlying science has grown substantially (the entire
key-anchoring program, now closed, all belongs to the ICLR paper, not the
workshop paper — the workshop paper's scope, Task D/E/frontier, is
unaffected by any of this closing period's work). This actually
strengthens the case for NOT waiting on a venue: the workshop paper is
scientifically complete and stable, and the ICLR full paper is the one
absorbing all new material — there is no reason for venue uncertainty on
the smaller paper to hold up the larger one's drafting timeline.

### Decision 2 — Title (workshop paper)

**Recommendation: keep the current default title, "When the Gradient Sees
Rank: Provable Necessity, Causal Recruitment, and Exact Composition in
Trained Matrix Memories" — do not adopt a fifth, constancy-suffices-themed
title.**

The four existing options in `main.tex`'s comments (default; (B)
question-framed; (C) explicitly two-paper-arc; (D) architecture-first) are
all about the Task D/E/frontier arc — none of them concern key-anchoring,
DeltaNet, or constancy-suffices, because **the workshop paper's scope does
not include any of that material** (it is the Chapter 2/matrix-native
paper; key-anchoring belongs entirely to the DeltaNet/ICLR paper). A
"constancy suffices"-themed title would only make sense for the ICLR
paper, and even there it would be a poor fit: constancy-suffices is one
resolved sub-thread (the anchoring program) inside a paper whose actual
headline is the capacity-vs-exactness separation and the F-geo-3 fix — see
`NARRATIVE.md` round 5 §10 item 6(i), which explicitly flags the risk of
letting the anchoring arc's material crowd out the paper's actual
headline claims. If a fifth title option is wanted for the ICLR paper
specifically (not asked for here, but worth flagging since the task
brief mentions "a new one reflecting constancy-suffices if warranted"),
the honest answer is **not warranted** — the ICLR paper's own title should
stay anchored to "capacity vs. exactness" (its actual, from-round-1
headline), not to a secondary mechanistic finding discovered four waves
into a follow-on program.

Among the four existing workshop-paper options, if a change is wanted for
other reasons: (B) is the best second choice — it is the most legible to
a reader coming from the companion ICML negative-result paper, which is
this paper's actual primary audience (workshop attendees interested in
the "does the gradient use rank" question specifically), more so than (C)
(presumes prior familiarity) or (D) (least narrative, worst fit for a
paper whose contribution is explicitly structured as a three-act story
per `NARRATIVE.md`'s own pre-mortem item 10).

### Decision 3 — Author block (workshop paper)

**Recommendation: fill in the author block now** — `Sam Larson, Pebble AI,
pebbleml.com` — **contingent on confirming the chosen venue's
double-blind requirement first (a 30-second check once the CFP text is in
hand), not before.**

`main.tex` already has the exact fill-in text staged as a comment
("CAMERA-READY TODO"), and the file's own build system already supports
both a single-blind and an anonymized (`\anon{1}`) build from one source —
so there is no reason to leave this as a placeholder once the venue is
confirmed, and it should be one of the fastest steps once Decision 1
resolves. Two things to check at that point, not now: (a) whether the
confirmed venue's CFP requires double-blind review (if so, build with
`\anon{1}` and keep the real author block only in the eventual
camera-ready); (b) per `VENUE_DECISION.md`'s own open question, whether
any DeltaNet co-contributor should be added given the discussion section's
concurrent-work paragraph — this is a substantive question the PI should
answer, not a formatting one, and it has not been resolved by anything in
this closing period's work.

### Decision 4 — Timeline

**Recommendation: sequence as (1) workshop paper page-cut + submission
the moment NeurIPS 2026 workshop CFPs open (~mid-to-late August deadline,
historical pattern), running in parallel with (2) ICLR full-paper drafting
starting now, since the ICLR paper's own material is fully closed except
for one pending number (Track C rung-3).**

Rationale: these two papers are on genuinely independent critical paths.
The workshop paper (Task D/E/frontier) needs a page-cut editorial pass
(§1 above) that can start immediately regardless of venue — the content
itself has not changed and does not depend on anything still running. The
ICLR paper's content is now, per `NARRATIVE.md` round 5's own accounting,
complete except for rung-3's single readout, which does not block drafting
any section except the exact numeric value in Fig. 9 Panel A and a
handful of "fourth rung running" phrases (`NARRATIVE.md` round 5 §9 item 3
and §10 item 5/6 already scope this precisely) — drafting can and should
start now, with a small, well-defined patch pending once rung-3 lands
(expected imminently, per `STATE.md`). ICLR 2027's own submission deadline
is not yet in this memo's source material and should be confirmed
separately before this timeline is finalized into calendar dates — this
recommendation is about sequencing (workshop-cut-now,
full-paper-draft-now, both in parallel, rung-3 as a small patch whenever
it lands), not about specific dates, since no ICLR 2027 deadline was
surfaced in any of the read documents for this memo.

---

## Sources consulted for this memo

`STATE.md` (2026-07-07 state), `matrix-thinking/KEY_ANCHORING_DESIGN.md`
(§9–§11.12, the full program record), `EXPERIMENT_LOG.md` (tail entries
through the K=48 capacity-curve verdict), `matrix-thinking/
SCALE_TRANSFER_DESIGN.md` (§5.10 + Addendum), `matrix-thinking/
submissions/iclr-2027/NARRATIVE.md` (round 5, this session's own
deliverable 1), `matrix-thinking/submissions/neurips-ws-2026/main.tex`
(measured directly: `pdftotext`/`pdfinfo` page count, section line
counts), `matrix-thinking/submissions/neurips-ws-2026/VENUE_DECISION.md`,
`experiment-runs/2026-07-06_trajectory_probes/TRAJECTORY_SUMMARY.txt`
(the dense-trajectory dataset's own README-equivalent), and a direct file
inventory of `matrix-thinking/submissions/iclr-2027/{sections/,main.tex,
REPRODUCIBILITY.md}` (line counts + a targeted grep for anchoring-related
content, to correct this memo's own initial draft assumption that no ICLR
section skeleton existed yet). No new research was performed for this
memo; every recommendation above traces to one of these already-written,
already-audited sources.
