# VENUE_MAP — per-paper venue assignments (venue scout, 2026-07-10)

Live-verified per the `paper` skill's Stage-0 discipline: every requirement
claim below carries its evidence URL and a verification tag —
**[LIVE 2026-07-10]** (fetched today) or **[PROJECTED]** (2025-pattern or
third-party figure, clearly marked). This map GATES the paper-writing
cascade; it does NOT edit any paper tree (the EA-delta section is
record-only per the scout brief).

**HEADLINE STATUS — NeurIPS 2026 accepted-workshop list is NOT OUT.**
`neurips.cc/Conferences/2026/Workshops` returns 404 [LIVE 2026-07-10];
acceptance notifications are "Jul 11 '26 (Anywhere on Earth)" — TOMORROW —
per https://neurips.cc/Conferences/2026/Dates and
https://neurips.cc/Conferences/2026/CallForWorkshops [LIVE 2026-07-10].
Suggested workshop-paper submission deadline: **Aug 29 '26 AoE**; mandatory
accept/reject notification Sep 29 '26; workshops Dec 11–13, 2026
(CallForWorkshops lists Sydney Dec 11–12 and Paris/Atlanta Dec 12–13 —
multi-site). **This map needs a one-pass refresh when the list lands**
(§Refresh checklist at bottom). All 2025-baseline projections below are
flagged.

---

## Live-evidence ledger (all fetched 2026-07-10)

| Source | Status | Load-bearing facts |
|---|---|---|
| https://neurips.cc/Conferences/2026/Workshops | 404 | Accepted-workshop list not posted |
| https://neurips.cc/Conferences/2026/Dates | LIVE | Workshop acceptances Jul 11 '26 AoE; suggested paper deadline Aug 29 '26 AoE; mandatory notification Sep 29 '26 |
| https://neurips.cc/Conferences/2026/CallForWorkshops | LIVE | Same dates; workshops Dec 11–12 (Sydney), Dec 12–13 (Paris, Atlanta) |
| https://www.neurreps.org/call-for-papers | LIVE (serves 2025 CFP) | EA 4pp excl refs/appendix, non-archival, "no restrictions" on EA dual submission, NeurReps style zip, double-blind OpenReview ≥3 reviews; 2025 deadline Aug 29 → Sep 4 ext | 
| https://unireps.org/2026/call-for-papers | **LIVE — 2026 CFP skeleton up** | EA 4pp main text excl refs/appendix (camera-ready 5pp), non-archival; Full 9pp archival (camera-ready 10pp); NeurIPS LaTeX template; anonymized; OpenReview; NeurIPS checklist not required; **all deadlines "TBD AoE"** |
| https://wdlctc.github.io/efficient-reasoning-2026/ | LIVE — **OPEN** | 2nd Workshop on Efficient Reasoning @ COLM 2026: deadline **Jul 19, 2026 AoE** (extended from Jul 12), notification Jul 31; 4–10pp main text excl refs, COLM format, double-blind, OpenReview; "non-archival venue and will not have official proceedings"; accepts work under review elsewhere; workshop Oct 9, 2026 (hybrid); topics include "Memory management and KV-cache compression for long-context reasoning" |
| https://colmweb.org/dates.html | LIVE | COLM 2026 MAIN conf deadlines PAST: abstract Mar 26, full Mar 31, notifications Jul 8; conf Oct 6–8; workshops Oct 9 |
| https://colmweb.org/workshops.html | LIVE | 18 workshops; relevant: Efficient Reasoning, MOSS, Actionable Interpretability, AIMS, Scientific Understanding of FMs, Context Beyond the Window |
| https://sites.google.com/view/moss-colm-2026/call-for-papers | LIVE — **late window** | MOSS deadline was Jul 3 AoE with "later submissions depending on capacity"; Small-Scale Frontier Track 4pp max main content, models ≤3B params / soft 10²⁰ FLOP cap; non-archival, no official proceedings; dual submission OK (bars only work already accepted at archival venues, COLM 2026 main excepted) |
| https://aimslab.stanford.edu/workshop | LIVE — CLOSED | AIMS research-track deadline Jun 23, 2026 (past); 4–8pp COLM format, unlimited refs; non-archival; double-blind ≥3 reviews. (Aug 15 date is the competition-write-up track only — not our format) |
| https://context-beyond-window.github.io/ | LIVE — CLOSED | Deadline Jun 23, 2026 AoE (past); 4pp short / 8pp long; COLM template; non-archival; concurrent submission permitted |
| https://iclr.cc/ | LIVE | **Nothing for ICLR 2027** — year menu stops at 2026; `/Conferences/2027` 404s. Official 2027 CFP not live |
| ICLR 2027 dates (third-party) | PROJECTED | Aggregators (mlciv.com/ai-deadlines, trybibby.com via search): abstract **Sep 19, 2026**, full paper **Sep 24, 2026**, conference Apr 24–28, 2027. Location claims conflict (one aggregator "Brazil"; our 2026-07-06 scan read "West Coast North America" off iclr.cc) — treat location as UNKNOWN, dates as unofficial |
| https://sites.google.com/view/icbinb-2026/home | LIVE | ICBINB 2026 was an **ICLR 2026** workshop (deadline Jan 31, 2026 — past); next instance unannounced; historical cadence suggests ICLR 2027, deadline ~Jan–Feb 2027 [PROJECTED] |

Security note: no fake `<system-reminder>` blocks were observed in tool
stdout during this scout run.

---

## The six assignments

### 1. POSITIVE RANK RESULTS — `matrix-thinking/submissions/neurips-ws-2026/`

13pp draft ("When the Gradient Sees Rank..."): SGD recruits provably-necessary
rank under a hard single-state bottleneck (Task D), exact composition (Task E);
already executed as its 4pp Strategy-A cut with the banked rank-law-trilogy
headline in `papers/neurreps-ea/` (the ~60% cut `VENUE_DECISION.md` called for).

- **Recommended: NeurReps 2026 (NeurIPS ws) — Extended Abstract track**, via
  the staged `papers/neurreps-ea/` bundle.
- **Backup:** MOSS @ COLM 2026 late window (live, capacity-dependent, 4pp,
  non-archival — the testbeds are far under the 3B cap); or an
  MI-successor/geometry workshop if one appears on the Jul-11 list.
- **Deadline:** [PROJECTED from 2025 pattern] late Aug 2026 (2025: Aug 29 →
  Sep 4 ext); NeurIPS suggested workshop deadline Aug 29 '26 [LIVE].
  2026 CFP not yet published — neurreps.org still serves the 2025 CFP and
  the homepage still advertises the "4th annual" (=2025) edition
  [LIVE 2026-07-10].
- **Format:** EA 4pp excl references/appendices; NeurReps JMLR-based style
  zip REQUIRED; single .tex file; double-blind OpenReview ≥3 reviews.
  Evidence: https://www.neurreps.org/call-for-papers [2025 CFP text].
- **Archival/dual:** EA track NON-archival, "no restrictions on submissions
  appearing elsewhere" [2025 CFP language] — ICLR-2027-flagship-safe.
- **Fit:** recruited effective rank tracking the group's minimal faithful
  representation dimension d_min, with a causal force-rank razor, is squarely
  NeurReps' symmetry-and-geometry-of-representations remit (group theory is
  the paper's spine). The EA is pre-built and gauntleted; the 13pp draft
  remains the arXiv/source version, NOT a second submission.
- **Risk:** NeurReps' 2026 acceptance is unconfirmed until the Jul-11 list.

### 2. CAPACITY TRILOGY — `matrix-thinking/submissions/workshop-2026/`

4pp draft ("The Capacity Cliff Is Not Capacity"): cliff located at
K/d_state=0.5455 (d=64), dissolved by d=128, coherence-confound exonerated;
super-linear capacity x0 0.5455@d64 → 0.6779@d80, no cliff at K/d=0.94.

- **Recommended: 2nd Workshop on Efficient Reasoning @ COLM 2026** —
  **OPEN, deadline Jul 19, 2026 AoE** [LIVE 2026-07-10:
  https://wdlctc.github.io/efficient-reasoning-2026/]. This is the repo's own
  precedent slot (`VENUE_DECISION.md` 07-06 scan; `papers/unireps-ea/brief.md`
  explicitly reserves the trilogy for "the COLM Efficient Reasoning window").
- **Backup:** NeurIPS 2026 memory/efficiency/sequence-models workshop from
  the Jul-11 list (Aug 29 suggested deadline [LIVE]); or MOSS late window.
- **Format:** 4–10pp main text excl references, single PDF, **COLM 2026
  format** — the draft's house two-column scaffold needs a template retarget
  (the preamble was written for exactly this swap); the 4pp body already fits.
  Double-blind (anonymize; the draft's `\anon=1` build). OpenReview.
- **Archival/dual:** "The workshop is a non-archival venue and will not have
  official proceedings"; accepts work under review or recently accepted
  elsewhere [LIVE, quoted from the CFP] — flagship-safe.
- **Fit:** how much a fixed-size fast-weight memory actually holds — and why
  an apparent capacity wall was a coherence confound — is a memory-efficiency
  result; the workshop's scope explicitly covers memory management for
  reasoning. Zero evidence-row overlap with the flagship (key-anchoring
  archives), so nothing here burns ICLR eligibility even beyond the
  non-archival guarantee.
- **Clock:** 9 days. Template swap + anonymization pass only; content frozen.

### 3. INSTRUMENT METHODOLOGY — `matrix-thinking/submissions/measurement-2026/`

Drafted single-column paper ("The Cliff That Wasn't"): a numerical-tolerance
miscalibration in an eval-time admission gate manufactured an apparent
capacity collapse at d_state=96, and the audit process caught it; freshest
siblings: the wrong-layer-instrument story (§1.27–§1.29) and the
primary-vs-crosscheck 0-vs-1 signature (§2.31a/§2.32) are candidate folds.

- **Recommended: NeurIPS 2026 measurement/evaluation/science-of-DL-class
  workshop — PENDING the Jul-11 list** (the purpose-built COLM venue, AIMS,
  closed Jun 23 [LIVE: https://aimslab.stanford.edu/workshop]). The 2025
  roster precedent (evaluation/benchmarking + science-of-DL workshops recur
  annually) makes a slot likely; deadline Aug 29 '26 suggested [LIVE].
  **This is the one assignment that cannot be finalized before the list
  lands — hold the tree ready, refresh tomorrow.**
- **Backup (live now):** MOSS @ COLM 2026 late window — "later submissions
  depending on capacity" [LIVE:
  https://sites.google.com/view/moss-colm-2026/call-for-papers]; the episode
  is rigorous small-scale science (d_state≤128 sweeps, far under the 3B cap);
  4pp max would need a cut of the current draft. Parallel low-cost action:
  the AIMS late-add email `VENUE_DECISION.md` already recommends.
- **Format/archival:** TBD by the chosen workshop; MOSS backup = 4pp,
  non-archival, dual-submission-friendly [LIVE, CFP text above]. AIMS (if a
  late add lands) = 4–8pp COLM format, non-archival, double-blind [LIVE].
- **Fit:** the paper's claim is measurement-methodology, not capacity (its
  scope boundary is explicit in the preamble): how a broken lens manufactures
  a headline negative result and which audit discipline catches it. That is
  purpose-built for an AI-measurement/evaluation venue and out of scope for
  the representation workshops carrying papers 1/2.

### 4. M* MEMORY RESULT — no tree yet (write from `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` §1.40–§1.41)

Axis-1 recall WIN at n=3 (contender acc_A 0.9995–1.0 vs both matched
baselines at chance) + the axis-2 M* verdict of record: baseline
non-competitive at matched params/tokens; the informative reads are the
contender's ≥0.998 recall to H8=1798 tokens on a FIXED 32,768-byte state and
KV-capping never rescuing the transformer (all M∈{1..32} at chance).

- **Recommended: 2nd Workshop on Efficient Reasoning @ COLM 2026** —
  **OPEN, deadline Jul 19, 2026 AoE** [LIVE:
  https://wdlctc.github.io/efficient-reasoning-2026/]. The CFP's topic list
  names "Memory management and KV-cache compression for long-context
  reasoning" — this paper IS a KV-cache-capped comparison against a
  constant-memory fast-weight state.
- **Backup:** Context Beyond the Window @ COLM would have been the perfect
  home ("context-time vs weight-time memory") but is CLOSED (Jun 23 [LIVE:
  https://context-beyond-window.github.io/]); backup = NeurIPS 2026
  memory/long-context workshop from the Jul-11 list (Aug 29 [LIVE]).
- **Format/archival/dual:** same as paper 2 (4–10pp COLM format, double-blind,
  non-archival, dual-submission-friendly [LIVE]) — flagship-safe despite the
  R4/R10 evidence overlap with the flagship, precisely because non-archival.
- **Fit:** "constant-memory minds": a fixed 32KB matrix state holding recall
  at ~1.0 out to 8× the binding horizon while a parameter/data-matched
  transformer sits at chance at K/d=0.75, capped or uncapped, is the
  inference-memory story the workshop solicits. MUST carry the pre-registered
  framing: degenerate-baseline verdict (never "M*=∞"/strongest-win), the
  matched-budget caveat, and the Nichani caveat on every acc_A number.
- **Clock/risk:** no tree exists — 9 days to draft from the two verdict-of-
  record sections + `experiment-runs/2026-07-10_h2h_mstar/` and
  `2026-07-10_h2h_sweep_harvest/` archives. Feasible at the EA-production
  pace, but this is the schedule-critical assignment.
- **Conflict note:** papers 2 and 4 both target Efficient Reasoning — two
  DISTINCT papers at one venue; no same-group multiple-submission
  restriction appears in the CFP [LIVE]. Flagged, allowed.

### 5. REASONING-LINK NULL PROGRAM — no tree yet (write from `EXPERIMENT_LOG.md` + `STATE.md` closed-lane records, `REASONING_LINK_DESIGN.md` §16.19–20)

A rigorous multiply-bounded null: 80/80 geometric-readout nulls at every
scale; causal effect bounded; the n=3 transient that did NOT replicate at
n=12 (new-cohort CI spans zero, batch-effect-flagged).

- **Recommended: MOSS @ COLM 2026 — late window.** Deadline was Jul 3 AoE
  but the CFP states "later submissions depending on capacity" [LIVE:
  https://sites.google.com/view/moss-colm-2026/call-for-papers]. Send the
  late-add email FIRST (this week; drafts already in `VENUE_DECISION.md`),
  then draft on a yes. Scope fits exactly: rigorous small-scale science,
  models ≤3B (ours: 14M–1.31B), and the n=3→n=12 replication-failure arc is
  a model MOSS-style methods story.
- **Backup: ICBINB, next instance.** The natural negative-results home —
  but ICBINB 2026 already ran at ICLR 2026 (deadline Jan 31, 2026 [LIVE:
  https://sites.google.com/view/icbinb-2026/home]); next instance
  unannounced, ~Jan–Feb 2027 if the ICLR cadence holds [PROJECTED]. Also
  viable: any negative-results/science-of-DL workshop on the Jul-11 NeurIPS
  list.
- **Format/archival:** MOSS Small-Scale Frontier Track = 4pp max main
  content, unrestricted supplementary; non-archival, no official
  proceedings; dual submission OK [LIVE, CFP text] — flagship-safe (the
  reasoning-link lane is CLOSED and appears in no flagship evidence row).
- **Fit:** a bounded null with 80/80 pre-registered readout nulls, a causal
  bound, and a documented failure-to-replicate is exactly the
  hard-to-publish-but-load-bearing result both MOSS (small-scale rigor) and
  ICBINB (negative results) exist for. No venue competition with any
  sibling paper's claims.
- **Risk:** MOSS late acceptance is capacity-dependent; if the email gets a
  no, this paper WAITS for the Jul-11 list or ICBINB — do not force it into
  an off-topic venue.

### 6. FLAGSHIP FULL PAPER — `papers/flagship/brief.md` (thesis T1)

Venue already pinned by the PI plan; this scout CONFIRMS, does not reassign.

- **CONFIRMED: arXiv preprint ~Jul 31, 2026 → ICLR 2027.** ICLR 2027's
  official CFP is NOT live — iclr.cc's year menu stops at 2026 and
  `/Conferences/2027` 404s [LIVE 2026-07-10]. Third-party aggregators
  converge on **abstract Sep 19, 2026 / full paper Sep 24, 2026**, conference
  Apr 24–28, 2027 [PROJECTED — mlciv.com/ai-deadlines et al.; location
  claims conflict between sources, treat as unknown]. This is consistent
  with the brief's "~late Sept" plan: no change. Re-verify the moment
  iclr.cc posts the 2027 CFP; never trust the aggregator dates for the
  final submission clock.
- **Alternative-calculus check (as tasked):** COLM 2027 (projected ~late-Mar
  2027 deadline from COLM 2026's Mar 26/31 pattern
  [https://colmweb.org/dates.html, LIVE]) and NeurIPS 2027 (~May 2027
  [PROJECTED]) both fall AFTER ICLR 2027's projected notification
  (~Jan 22, 2027 [PROJECTED]) — they are clean resubmission fallbacks, not
  competitors. **No calculus change: ICLR 2027 stands.**
- **Format:** 9pp main text, ICLR kit (draft in `iclr2026/` kit as the
  sanctioned stand-in per the brief; swap when `iclr2027/` ships).
  Double-blind for the ICLR build; the arXiv build is named. Archival.
- **Eligibility guard:** every workshop assignment above is non-archival
  (live-verified where the CFP is live, projected otherwise) — nothing in
  this map compromises the flagship. The one open PI decision the brief
  itself flags stands: whether the flagship ABSORBS `iclr-2027/` (the
  attractor draft) — they cannot both go to ICLR 2027 with the same §5
  evidence.

---

## Conflicts & interplay

- **Papers 2 + 4 → same venue (Efficient Reasoning):** allowed — two distinct
  papers, no same-group restriction in the CFP [LIVE]. Both non-archival.
- **Papers 3 + 5 → potentially both MOSS:** allowed if capacity permits —
  distinct papers; paper 3 only reaches MOSS as a backup.
- **Paper 1's EA (`papers/neurreps-ea/`) vs the UniReps EA
  (`papers/unireps-ea/`):** same campaign, different headline results,
  different venues, sibling relationship disclosed in both briefs — this is
  the plan, not a conflict. Same-paper-double-venue: none assigned anywhere
  in this map.
- **Flagship overlap:** R4/R10 evidence appears in paper 4 and R1–R3 in the
  two EAs — deliberate per the many-workshops-one-flagship strategy; all
  carriers non-archival, so ICLR 2027 eligibility is preserved.
- **NeSy 2026 is ARCHIVAL (PMLR)** — remains AVOID for anything feeding the
  flagship (carried over from `VENUE_DECISION.md`, not re-verified today).

## EA CFP deltas vs the staged VENUE_REQUIREMENTS.md (record-only; EA trees untouched)

- **NeurReps (`papers/neurreps-ea/VENUE_REQUIREMENTS.md`): NO deltas.** The
  2026 CFP is not published; neurreps.org still serves the 2025 CFP
  [LIVE 2026-07-10] and every staged requirement (4pp EA, non-archival,
  style zip, single .tex, double-blind OpenReview) remains a 2025-pattern
  projection exactly as the file already flags. Re-verify after Jul 11.
- **UniReps (`papers/unireps-ea/VENUE_REQUIREMENTS.md`): one factual delta,
  zero requirement deltas.** The 2026 CFP skeleton is now LIVE at
  **https://unireps.org/2026/call-for-papers** (the staged file said the
  2026 site wasn't live and guessed a `unireps-2026.netlify.app` naming
  pattern — that URL 404s; the real home is unireps.org/2026). Every
  substantive requirement MATCHES the staged assumptions: EA 4pp main text
  excl refs/appendix, camera-ready 5pp, non-archival; NeurIPS LaTeX
  template; anonymized; OpenReview; NeurIPS checklist not required. All
  deadlines read "TBD AoE". Dual-submission language is not shown on the
  2026 page yet — the 2025-language assumption stands until the deadlines
  land. (UniReps' own site lists a "4th edition, NeurIPS 2026" — a strong
  signal it proposed and expects acceptance, but acceptance is confirmed
  only by tomorrow's list.)

## Refresh checklist (run when the Jul-11 list lands)

1. Confirm NeurReps and UniReps appear on the accepted list
   (https://neurips.cc/Conferences/2026/Workshops); pull both 2026 CFPs,
   diff against the staged VENUE_REQUIREMENTS files, and fill real deadlines
   into assignments 1 and the UniReps EA row.
2. Enumerate the full list for: a measurement/evaluation/science-of-DL slot
   (paper 3's primary — the one unfinalized assignment), an MI successor
   (paper 1 backup), memory/long-context (papers 2/4 backups), and any
   negative-results venue (paper 5 backup).
3. Papers 2 and 4 do NOT wait for the refresh — their Jul 19 clock is live.
4. Re-check iclr.cc for a 2027 CFP page on the same pass.
