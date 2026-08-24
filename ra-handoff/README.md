# RA Handoff — the paper portfolio

**For:** Will Larson (Williamlarson2023@gmail.com)
**From:** Sam Larson (sam@pebbleml.com), PI
**Repo:** `saml212/matrix-states` (private) — this folder is `ra-handoff/`
**Written:** 2026-08-24

You are taking over the entire paper portfolio. Your job is to understand the
research motion, rewrite the papers in your own voice, improve the figures and
presentation, decide or confirm where each one goes, post them to arXiv, and run
each submission end to end. You sign as co-author on every paper you take
through that process — the PI's instruction, for your editorial and writing
contribution.

Read this file first. Then `PAPERS.md` (what exists), then
`SUBMISSION_PLAYBOOK.md` (how to ship it), then `DATA_MAP.md` (where the numbers
live).

---

## 1. The program in two pages

### The object of study

Standard transformers carry a key-value cache that grows with the sequence.
Linear-attention and *fast-weight* models replace it with a fixed-size matrix
state `S` of shape d×d, written by an outer-product rule and read by a
matrix-vector product. The delta-rule family (DeltaNet and its gated
descendants) is the current standard bearer; its update is

```
S_t = S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```

a rank-one correction that overwrites whatever value was previously bound to the
incoming key.

The literature evaluates these models on one axis almost exclusively: quality
per unit of compute or memory, relative to attention. The state is treated as an
implementation detail. **This program treats the d×d state as the object of
study** — a representational medium with a capacity structure, a causal role,
and failure modes that never show up on a perplexity leaderboard.

That reframe is the whole thesis. Every paper in the portfolio is one measurement
of it.

### The four results, in the order they were established

**(1) The rank law.** Train a matrix-state encoder on permutation-group word
problems whose minimal faithful representation dimension `d_min` runs 2 to 5.
Recruited effective rank tracks `d_min` (Spearman ρ = 0.9747, the maximum the
tie structure allows). A designed pair — S₄ versus A₅, same `d_min` = 3,
different solvability — comes out *equivalent* under a TOST equivalence test: the
state follows dimension, not solvability. Then the causal razor: force rank to
`d_min − 1` at train time and recovery reads exactly 0.000 in all five groups;
restore `d_min` and it returns. SGD recruits exactly the rank the task needs, no
more and no less.

**(2) The capability separation.** A two-layer delta-rule model does single-pass
episodic recall at accuracy 0.9995. A parameter-matched vector-state ablation
(Δ = 1,024 params, 0.007%) reads chance. A compute-matched transformer reads
chance too — and stays at chance after an explicit four-point learning-rate
search, where, notably, the rate that best optimizes the language-modeling loss
reads recall *furthest below* chance. The capability is causally localized:
zeroing layer 0's state collapses recall to chance, zeroing layer 1's changes
nothing. And it is stored *nonlinearly* — no linear probe at any state-level tap
reads it; only the model's own forward pass decodes it. It is also
memory-stable: ≥0.998 out to 1798 tokens on a fixed 32,768-byte state, while a
cache-capped transformer reads chance at every cap.

**(3) The pathology.** The same write mechanism, run at language-model scale,
drives the write keys toward a collapsed population geometry. The span fraction
of that drift climbs monotonically 0.248 → 0.344 → 0.389 → 0.455 across a
14M → 98M → 392M → 1.31B ladder at held-fixed data mixes. Removing
qk-normalization changes nothing (0.05σ at n=3), so it is not an artifact of the
stock stabilizer. A frozen-key-bias mitigation that is loss-neutral at every
scale fails to transfer its 14M geometric benefit to 98M or 392M. Capability and
pathology are two faces of one storage mechanism.

**(4) NCR — Native Composition Reads — and the two scaling chapters.** This is
the spearhead, and it is what the last two months bought. Graft a composition
head onto a real 98M-parameter delta-rule language model. The head holds one
d×d relation operator `O` written from the context; K entities sit on a single
Hamiltonian K-cycle; the answer to a query at hop depth `h` is the image of the
queried entity under `O^h`, computed by repeated squaring in **O(log h)** matrix
multiplies instead of `h` sequential applications.

Two write regimes are measured through the identical read path and the identical
trained weights:

- **P1b** — exact teacher-forced operator substitution. The operator is computed
  in closed form from the true binding and substituted into the trained
  checkpoint's own read path. This measures the **read path's capability**. Every
  capability number in both scaling chapters is P1b.
- **P0** — the model's own SGD-learned write, read through the identical path.
  **This is the wall.**

The metric is chance-corrected: κ = (acc − 1/K)/(1 − 1/K), so readings at
different K compare directly.

### The two completed chapters

**Breadth (Section 6, `papers/flagship/sections/05a_breadth_scaling.md`).**
(K, d) from (12, 13) to (40, 41) — a 3.3× sweep, 48 cells, both recipes. The
read-path capability is **flat at ceiling** the whole way (floor κ = 0.9708; 3/3
frozen seeds clear the 0.90 band at every K). The curve ends at the *design*
limit — K = 44 is impossible under the 3K/2 ≤ 63 construction constraint — not at
a capability limit. The learned-write wall is absolute at depth everywhere, with
one toehold at K = 12, h = 1 only (0.1211–0.1484 vs chance 0.0833, reproduced at
a second eval seed, 0/42 at deep hops). So the separation **widens** with
breadth. Freezing the entity adapter buys depth robustness, and that advantage
grows with breadth: negligible at 5 squarings, T = 61.5/72 at 11 squarings,
p = 3.071e-5.

**Parameters (Section 7, `papers/flagship/sections/05b_parameter_scaling.md`).**
98M → 392M, 4.008× per arm, everything else held fixed. Capability shows **no
detectable directional shift** (6-stratum governing form, T = 14.5/54). The wall
is **scale-stable** (59–60 of 60 readings per K in band; both excursions
re-measure in band). The freeze ordering **sharpens to perfect separation**
(T_W = 36/36 on both readouts, p = 1.25e-5). Two degradations appeared at the
frontier, and the mandatory attribution arm split them: K = 32 trainable was
**withdrawn by our own control** as token-budget-limited (0.8750 → 0.9113 when
given more budget), and K = 40 trainable **stands as architectural** after a
two-control sequence that first exposed warm-restart schedule damage and then
ruled it out at constant LR. That withdrawal is written up as a methods
strength, not buried.

The through-line: **the separation is scale-stable and its moat widens.**

**The 1.31B point is in flight.** Rung 3 (d_model 2560, 22 layers, d_state 128 —
the repo's own attractor-era config, verbatim) was built and probed on real CUDA
on 2026-08-23: no OOM at batch 32, measured parameter count matches the design
formula exactly at both K, gates 20 PASS / 0 FAIL. Calibration pair plus a
24-cell sweep, ~288 GPU-h, landing ~2026-08-26. **Three things do not transfer
cleanly and are flagged for a ruling, not assumed** — see the commit message on
`79b3d41` and `db9705f`. Most important for you: the cross-scale statistical
instruments are *pairwise* (98M-vs-392M, two-point equivalence margin). A third
rung needs either a second pairwise test or a genuine three-point trend
statistic, and **neither is pre-registered**. Do not let a 1.31B number into a
paper as a three-point trend until that instrument is pinned.

The Brev 8×H100 grant ends **~2026-09-01**. After that there is no cheap way to
run anything new. See `DATA_MAP.md` §5 for what must be pulled off the box
before then.

---

## 2. How the evidence chain works

This is the part that makes the portfolio defensible, and it is the part you
must not break. The science is fully human-auditable from raw artifacts. That is
the strong position, and it is worth protecting.

**`EXPERIMENT_LOG.md` is append-only ground truth.** ~900KB, entries dated and
numbered (`## 2026-08-23 #4 — ...`). Nothing is ever edited in place; when a
number turns out to be wrong, a *new* entry corrects it and names what it
corrects. Search by date, then by entry number.

**`experiment-runs/<date>_<slug>/` holds the raw JSONs.** One directory per
wave. Per-cell result JSONs, harvest summaries, `md5_manifest.txt` files
verifying local copies against the box copies. Files ≤25MB are tracked in git;
anything larger lives only on the SSD. Full policy in
`experiment-runs/README.md` and in `DATA_MAP.md` §1–2.

**The findings pages in `pebble-ai-site/findings/` carry raw-recomputed
numbers.** Before publication, each finding's numbers were recomputed
independently from the raw archives rather than copied from log prose.

**When the log and the findings page disagree, the page's raw-recomputed value
governs.** This is not a style preference — it is a recorded rule, and the
correction entries name each case. Two you will hit immediately:

- `EXPERIMENT_LOG.md` 2026-08-22 #23 and 2026-08-23 #5 — the finding-20
  amendment recompute. Direction-of-effect is **1 helped / 4 flat / 7 hurt** by
  the aggregator's own ±0.01 rule (an earlier entry said 1/5/6; K=24T h_top
  crosses the boundary by a thousandth, and the boundary call is named on the
  page). The 40k wall reading is **118/120 in band** with two unreplicated
  single-seed h=1 excursions, not "all 12" — that earlier phrasing was the
  cell-level statement, not the reading-level one.
- The chapter ledger recomputed to ≈135.7 GPU-h all-in against a 130-h gate. The
  overrun **stands authorized** under a recorded license that anticipated ~136 —
  stated on the page as authorized rather than retrofitted.

Both new flagship chapters carry this rule in their source header comments:
*"the recomputed findings notes govern wherever they differ from EXPERIMENT_LOG
prose."*

**Evidence rows.** Every numerical claim in the flagship maps to a row in
`papers/flagship/brief.md` with three fields: the pre-registered verdict record
(design-doc § or log entry), the raw artifact (path + md5), and the figure or
table that shows it. Rows are R0–R12. The abstract and intro carry inline
`<!-- evidence: R4 -->` comments pointing at them. **A number with no evidence
row is not a claim** — it is a critical finding to be fixed before submission.
If you add a claim, add its row.

---

## 3. How to work

**Edit the markdown, not the LaTeX.** For the flagship and for every tree that
uses the markdown pipeline, `sections/*.md` is the prose source of record and
`latex/sections/*.tex` is generated. Editing the `.tex` directly means your
change is silently reverted the next time anyone regenerates. Check each tree —
`PAPERS.md` records which are markdown-driven and which are hand-written LaTeX.

**The build path.**

```bash
# flagship: regenerate .tex from markdown, then compile
cd papers/flagship/latex
python3 md2tex.py            # sections/*.md -> latex/sections/*.tex
tectonic -X compile main.tex # three passes handled internally; writes main.pdf
```

`tectonic` is the house LaTeX toolchain (`/opt/homebrew/bin/tectonic`). There is
no `pdflatex`/`latexmk` on this machine, so don't reach for them. Trees with a
`Makefile` use `make` — same tectonic underneath. `md2tex.py` is designed so
that regeneration is a **no-op** on unmodified sections; if it produces a diff on
a section you didn't touch, that is a bug worth reporting, not something to
commit past.

Verified 2026-08-24: the flagship compiles clean from a cold cache, exit 0,
26 pages.

**Every claimed number must trace to an archive.** Before a number ships, open
the raw JSON it came from. Not the log prose about it, not the design doc's
summary of it — the artifact. The evidence rows give you the path and the md5.
This is the single discipline the whole portfolio rests on.

**Commits are the persistence events.** This repo follows the `paper` skill's
repo mode: each drafting stage, each review round, each bundle lands as its own
commit, and `git log -- papers/<slug>/` reconstructs how the paper was made. Do
not squash a run into one commit — the history is the audit trail.

---

## 4. Style: write these in your own voice

**The goal is prose a human researcher owns.** Not lightly-edited machine
output — yours. Rewrite. Restructure. Cut what doesn't earn its place. Change
the framing if you think the framing is wrong.

Two things make this more than cosmetic.

First, the flagship's AI-detector gate **hit its cap without passing**. Six
rounds, twelve fresh judges, no judge reused, no round shown a prior verdict —
and it never reached the required two consecutive all-"100% human" rounds
(`papers/flagship/detector/terminal-state.md`). Per the procedure, the draft is
not accept-ready and a human has to adjudicate. You are that human, and the real
fix is not another polish pass; it is a genuine rewrite by an author who owns the
sentences.

Second, several venues have LLM-assistance disclosure policies, and they differ
by venue and change year to year. **Follow the current policy of whatever venue
you submit to, at the time you submit.** Read the CFP's policy section; don't
infer it from another venue. The science here is fully human-auditable from raw
archives, which is a strong position to be in — every number can be recomputed
from a checksummed artifact by anyone. Be straightforward about process and let
the auditability carry the weight.

**Presentation matters — figures are key.** The current figures are functional
and mostly unloved. They are generated by per-tree Python (the flagship's is
`papers/flagship/figures/figure-gen.py`, which asserts input md5s before
plotting — keep that property). Improving them is squarely in scope and is
probably the highest-leverage single thing you can do for how this work lands.
Add figures where a figure would carry an argument better than a table.

**You have latitude.** Add sections. Cut sections. Re-home a paper to a better
venue. Ask Sam for more data or another run — the box is alive until ~Sep 1 and
after that it's a conversation about what's worth renting hardware for. Mine
`EXPERIMENT_LOG.md` for anything: there is far more in there than made it into
papers, including at least one standalone mechanistic result (every tensor reset
in the entity pathway rescues deep composition, no single tensor is a privileged
cause — 2026-08-18 #17) that nobody has written up.

**The two hard rules:** don't claim a number that doesn't trace to an archive,
and don't quietly drop a caveat that a verdict record attached to a claim. The
caveats are what make the strong claims believable.

**Authorship.** You sign as co-author on the papers you take through
submission. Author order and the full author list are Sam's call — the trees
currently carry `[AUTHORS — PI decision pending]` placeholders. Settle that with
him before the first arXiv posting, because arXiv author lists are painful to
change after the fact.

---

## 5. Priority order

The PI's priority is the flagship, and over the next two months that is
correct. But the inventory turned up a calendar problem that outranks it this
week, so read both.

**Priority per the PI:**

1. **`papers/flagship/`** — the ICLR 2027 full paper. Biggest, most complete,
   highest stakes, and it has the most open editorial questions. Its real
   deadline is late September.
2. **The workshop stack** — the eight short papers in `papers/`. Most are
   gauntlet-complete and near-shippable; they are the fastest route to things
   actually being public.
3. **The older `matrix-thinking/submissions/` trees** — one is published
   (reference only), one is a frozen 85%-done full paper with an unresolved
   relationship to the flagship, the rest are superseded.

**⚠ The exception, and please raise it with Sam in week one.** The entire
portfolio was frozen on 2026-07-17 while the GPU program ran, and **most of its
venue deadlines have since passed**. The one clock that may still be live is the
NeurIPS 2026 workshop suggested deadline, **Aug 29 2026 AoE — five days out**.
Three papers (`neurreps-ea`, `unireps-ea`, `rank-recruitment-ws`) are
ACCEPT-READY with built bundles for exactly those venues. Nobody ever ran the
Jul-11 accepted-workshop-list refresh that four separate files schedule, so the
real 2026 deadlines are unknown.

An hour spent on `neurips.cc`'s accepted-workshop list this week is probably the
highest-value action available: it gives the real NeurReps and UniReps
deadlines, supplies the venue that `measurement-ws` has been waiting six weeks
for, and supplies re-homing candidates for the three papers whose COLM deadlines
lapsed. Details in `SUBMISSION_PLAYBOOK.md` §5.

`PAPERS.md` has the full inventory with per-paper status, venue, and next steps.

---

## 6. Glossary

Terms that appear everywhere and are defined nowhere obvious.

| Term | Meaning |
|---|---|
| **NCR** | Native Composition Reads — the composition head that reads `O^h` by repeated squaring in O(log h) matrix multiplies |
| **P1b** | Exact teacher-forced operator substitution: the operator computed in closed form and substituted into the trained model's own read path. Measures **read-path capability** |
| **P0** | The model's own SGD-learned write, read through the identical path. **The wall** |
| **the wall** | P0 sits at chance at depth, everywhere tested. The model can *execute* composition it cannot *learn to write* |
| **κ (kappa)** | Chance-corrected accuracy, (acc − 1/K)/(1 − 1/K), so different K compare |
| **K, d** | Number of bound entities, and operator width. Always d = K+1 in the scaling chapters |
| **h** | Hop depth — how many times the relation operator is applied |
| **freeze ordering** | Frozen entity-adapter arms compose at depth; trainable arms degrade. The gap grows with breadth and sharpens to perfect separation at 392M |
| **span fraction** | The write-geometry pathology metric: how collapsed the write-key population geometry is |
| **the gauntlet** | The adversarial review procedure — attack, defense, rebuttal, style, format, render inspection, then the detector gate. Each stage a fresh agent with no memory of prior rounds |
| **pre-registration** | Decision bands and thresholds written down *before* data is collected. "Bands before data" |
| **verdict of record** | The single adjudicated result for a question, recorded in a design doc § or log entry. Supersedes all prior readings |
| **PI** | Principal investigator — Sam |
| **the box** | The Brev 8×H100 cluster, `youthful-indigo-turkey`. Grant ends ~2026-09-01 |

---

## 7. Contents of this folder

| File | What it is |
|---|---|
| `README.md` | This file — the master brief |
| `PAPERS.md` | Complete portfolio inventory: every tree, status, venue, next steps |
| `SUBMISSION_PLAYBOOK.md` | arXiv mechanics, per-venue flows, the open venue-homing decisions |
| `DATA_MAP.md` | Where every artifact lives, and what must be pulled off the box before ~Sep 1 |
| `EMAIL_DRAFT.md` | The handoff email (for Sam to send) |
| `ra_package.zip` | The five docs plus every current paper PDF |
