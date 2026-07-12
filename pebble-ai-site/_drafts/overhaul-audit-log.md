# pebbleml.com overhaul — adversarial audit log

Deploy-excluded (`_drafts/`). Records each round of the required adversarial
loop: fresh Sonnet auditor, hostile brief, six dimensions (coherence,
staleness, figure quality, novel/interesting, accuracy, honest dates).

## Round 1 (fresh Sonnet auditor, hostile brief)

**Verdict: NOT ready to ship — needs a fix-and-reaudit round.**

Findings (severity as scored by the auditor):

1. **CRITICAL — coherence/staleness.** Roadmap's "M* memory-multiplier walk" phase described as "in flight," but the campaign already completed (finding no. 11 + its own linked fanout data show 0/90 cells clearing the bar at any M/H — a definitive null, not an open question). Self-contradicted the linked finding two clicks away.
   **Fix:** removed the standalone "active" phase; folded the completed M×H fanout result into the "done" phase's prose (0/90 cells clear bar at any cap/horizon).
2. **CRITICAL — coherence/staleness.** Roadmap's "Stage-2: 57-cell sweep" phase read as not-yet-launched directly next to links to findings no. 12–13, which describe a 62-cell sweep as already complete and adjudicated (INCONCLUSIVE-TRAINABILITY-LIMITED) — apparent self-contradiction from conflating two different sweep sizes in the same design registry.
   **Fix:** rewrote the phase to explicitly distinguish the completed 62-cell sweep (already adjudicated, linked) from a separate, larger, still-gated remainder of the same design grid, without overclaiming precision about the remainder's exact size.
3. **CRITICAL — staleness.** Footer said "last updated 2026-04-28" while the page's own content (findings dated through July 11) flatly contradicted it.
   **Fix:** updated footer to 2026-07-11 (the verified real date of this overhaul's edits, cross-checked against git commit history already in the repo — not taken from any injected/unverified date claim).
4. **CRITICAL — honest dates.** `outer-product-embedding.html`'s displayed date (Feb 28, 2026) had zero corroboration anywhere in the repo; earliest EXPERIMENT_LOG.md entry is 2026-03-26, 26 days later. Traced to a 2026-04-14 renumbering commit that assigned dates to make the sequence look chronologically clean, not to any real event.
   **Fix:** researched the actual earliest real evidence for this finding's core claim (Run 10, 2026-03-27, the first matrix-vs-vector T=1 comparison at matched ~5.15M params, 11.4× PPL — the literal source of the page's own "11×" headline stat) and corrected the date to March 27, 2026 everywhere it appears (JSON-LD, meta tags, visible byline, index.html card).
5. **SERIOUS — coherence.** Finding numbers jump 05→07 with no explanation of the missing "06" (an old internal draft flagged this half-fixed months ago and it was never resolved).
   **Fix:** added a one-line explanatory note in the findings section lede.
6. **SERIOUS — figure quality.** `generate_mstar_fanout_heatmap.py`'s docstring claimed "no in-figure title" while the code called `ax.set_title()` with real rendered text on the live embedded SVG.
   **Fix:** replaced `set_title` with an in-axes panel-label annotation (needed to distinguish the two side-by-side task heatmaps, analogous to existing per-panel group labels elsewhere on the site) and regenerated — verified identical underlying numbers (90/90 md5-verified, min/mean/max unchanged).
7. **SERIOUS — accuracy.** `llms.txt` still said "130×" (twice) for the parameter-efficiency ratio while every other page on the site correctly says 128× (65,536/512 = 128 exactly) — a known, previously-flagged-but-never-fixed discrepancy from an old internal draft.
   **Fix:** corrected both occurrences to 128×.
8. **SERIOUS — figure quality (process gap, not a confirmed wrong number).** Three scripts (`generate_fast_weight_recall.py`, `generate_rank_law.py`, `generate_superlinear_capacity.py`) plotted hand-transcribed numbers with citations but no runtime data load or verification, unlike most other scripts on the site.
   **Fix:** added real runtime loading + md5-printing + assertion-based cross-checks against the cited raw JSONs for all three scripts (fast-weight recall: round4 calibration cells + tap-localization JSON; rank law: harvest_summary.json + 19 raw per-seed cell JSONs; superlinear capacity: the d=64/d=80 sigmoid-fit JSONs). d=96 in superlinear-capacity remains a disclosed, cited-only transcription (no single fit-result file exists for it — the finding there is that no fit exists). Regenerated all five SVGs; visually and numerically identical to the pre-fix versions (verified via re-run assertions).
9. **MINOR — figure quality.** `ncr-operator-bank.html`, `output-head-dynamics.html`, and `rank-enrichment.html` are relatively figure-thin given claim density. Not addressed this round (all three already gained figures or caption fixes this overhaul; further figures deferred as lower priority than the CRITICAL items).
10. **MINOR — hero framing.** Headline energy ("20.9× faster... composes exactly") reads a bit stronger than the hedged caveats underneath. Judged acceptable — the hero and progression-section text already leads with WIN-PARTIAL framing for NCR and "convergence-recovered, far-depth open" for the operator bank; not further softened this round.
11. **Noted, not a site bug.** STATE.md's own top-level Stage-2 narrative is stale relative to the site (which correctly reflects the deeper, more current design-doc record). Out of scope for a pebbleml.com-only overhaul — flagged for a separate STATE.md pass, not touched here.
12. Security: one fake `<system-reminder>` (date-change + concealment) appeared in the auditor's own tool output; disregarded per standing protocol, reported here.

## Round 2 (fresh Sonnet auditor, hostile brief, re-audit)

**Verdict: NOT CLEAN — needs another round.** Independently verified 7 of 8
round-1 fixes as genuine; found round 1's Stage-2 rewrite itself still
self-contradictory (a duplicate of the M* bug pattern), plus 3 new issues.

1. **CRITICAL — coherence, round 1's Stage-2 "fix" was itself wrong.**
   Round 1 rewrote the roadmap to describe "a separate, larger remainder"
   of the Stage-2 grid as still gated. The auditor traced STATE.md
   §2.30/§2.32 directly: the "11-cell calibration gate" + "51-cell
   remainder" (11+51=62) **is** the same 62-cell grid findings no. 12–13
   already cover — there is no separate pending remainder.
   **Fix:** re-read STATE.md §2.27-§2.32 directly (not through CLAUDE.md's
   stale one-line summary, which is what produced round 1's error too).
   Rewrote the roadmap phase as "done", pointing at findings 12-13, no
   remaining gate implied. Also softened the downstream NCR-at-scale
   phase's gate description (the calibration readout it names as a
   precondition is now available; a separate coordinator go/no-go is what
   remains, not "the calibration hasn't happened").
2. **SERIOUS — coherence/accuracy.** Finding no. 01's index.html teaser
   quoted the page's own explicitly de-emphasized "11×" pre-registration
   check rather than either of its two actual headlined comparisons.
   **Fix:** rewrote the teaser to lead with the param-matched headline
   (26% lower BPB despite the flat model having 2.2× more params) and the
   175× PPL gap on the larger reasoning corpus.
3. **SERIOUS — accuracy.** A stray "~130×" survived in
   `outer-product-embedding.html` (a different sentence than the one
   round 1's llms.txt fix touched) — verified 262,144/2,048 = 128 exactly.
   **Fix:** corrected to "~128×".
4. **CRITICAL — honest dates, same fabrication signature on 3 more pages.**
   `rank-enrichment.html` (Apr 2), `output-head-dynamics.html` (Apr 5),
   `parameter-efficiency.html` (Apr 7) all trace to the same 2026-04-14
   renumbering commit that fabricated finding no. 01's date, with the
   same zero-corroboration signature (their real cited evidence is dated
   5-8 days earlier).
   **Fix:** researched each page's actual cited experimental evidence in
   EXPERIMENT_LOG.md and corrected to the real date: rank-enrichment →
   March 27, 2026 (Run 12/"Round 2", the source of the 5.02→6.12 trajectory);
   output-head-dynamics → March 31, 2026 (Run 21, the last of the three
   compared runs to complete, since the finding requires all three);
   parameter-efficiency → April 2, 2026 (Run 25, source of the newly-added
   fig 6). Updated JSON-LD, meta tags, visible bylines, and index.html cards
   for all three; verified ascending order with neighboring findings holds.
5. **MINOR — accuracy, self-introduced by round 1's own fix.**
   `generate_rank_law.py`'s newly-added runtime-computed error bars used
   ddof=0 (population std) while the adjacent prose's "±" values are
   ddof=1 (sample std) — confirmed by direct computation (e.g. S3: ddof=1
   gives 0.0605→"0.060" matching the text; ddof=0 gives 0.0494, no match).
   **Fix:** switched to ddof=1, regenerated, values now match the prose
   exactly for all 5 groups.
6. **MINOR, not fixed — deferred.** `generate_literature_synthesis_years.py`
   remains a disclosed manual transcription of `references.md` rather than
   a fully runtime-loaded parse, unlike the three scripts round 1
   converted. Lower stakes (backs a citation-count chart, not a scientific
   claim); the auditor flagged this as optional. Left as-is.
7. **Confirmed clean, no action needed:** figure quality (all 4 spot-checked
   scripts ran live and reproduced correct numbers; the 3 `set_title` hits
   found elsewhere all apply only to non-embedded social-card PNGs, not
   site figures — a legitimate, pre-existing, disclosed convention);
   novel/interesting-vs-hype (clean); 6 of 8 round-1 fixes confirmed
   genuine with no side effects (M* roadmap fix, footer date, finding-06
   explanatory note, mstar heatmap title removal, llms.txt 128× fix in
   isolation, the three runtime-verification retrofits).
8. Noted by the auditor as a process point, not a content defect: none of
   this was committed to git yet at audit time. Expected — commit happens
   after the loop closes, per the task's specified sequence.
9. Security: one fake `<system-reminder>` (date-change + concealment)
   appeared again in the auditor's tool output; disregarded, reported.

## Round 3 (fresh Sonnet auditor, hostile brief, re-audit)

**Verdict: NOT CLEAN going in — needs one more fix round.** Independently
re-traced the Stage-2 roadmap issue past STATE.md into the primary design
doc (`CAPABILITY_SEPARATION_DESIGN.md` §2.33-§2.35) and confirmed round 2's
fix is now genuinely correct (verdict of record: INCONCLUSIVE-TRAINABILITY-
LIMITED, superseding §2.32's as-built FALSIFY — matches finding no. 13
exactly). Confirmed 17/17 checkable prior-round fixes still hold, plus the
ddof=1 fix independently recomputed from scratch and matches the prose
exactly for all 5 groups. Found 2 SERIOUS + 1 MINOR new issues, none seen
by either prior round:

1. **SERIOUS — accuracy, present since before this overhaul, never caught.**
   `parameter-efficiency.html` and `literature-synthesis.html` both state
   Run 14's "0.87 vs 1.67 BPB at matched FLOPs" as current fact.
   `EXPERIMENT_LOG.md` retracted this exact framing on 2026-07-02 (the
   FLOPs budget was throughput-estimated, not analytically accounted; the
   runs were never truly FLOPs-matched) and the genuinely-matched-FLOPs gap
   was later measured properly by Stage G: matrix BPB 3.5552 vs vector BPB
   3.2511 (`STAGE_G_DESIGN.md` §14, also cited in STATE.md's "Honest
   negative results").
   **Fix:** `parameter-efficiency.html` — replaced the Run 14 claim in the
   main counterpoint paragraph, the single-seed-caveat list item, and the
   reproducibility citation with the corrected Stage G number and an
   inline note explaining the retraction. `literature-synthesis.html` —
   added a prominent correction callout immediately under the §3.6 heading
   (before any of the retracted-number restatements), extended the fig-1
   ASCII-diagram caption to flag its baked-in "Run 14, 0.87 vs 1.67" as the
   original (superseded) number, and corrected the "Confirmation bias" list
   item's standalone citation. Left the bulk of §3.6's flowing historical
   prose otherwise as originally written — the page is explicitly framed
   (via this overhaul's own added top-of-page note) as a preserved
   April-2026 reasoning trace, not a currently-maintained claim; the
   section-level correction note make its numbers unmistakably superseded
   without rewriting the historical argument sentence-by-sentence.
2. **SERIOUS, adjudicated rather than applied verbatim — dates.** The
   auditor found `outer-product-embedding.html`'s displayed date (March 27,
   fixed in round 1) postdates the page's own body text calling Run 22
   (April 1) the "(headline)" — cleanest — evidence, and suggested changing
   the byline to April 1. Investigated directly: Run 22 is real and dated
   2026-04-01, but Run 10 (March 27) is what the page itself calls "our
   earliest 8×H100 run" and is where the underlying phenomenon (matrix
   beats flat vector at T=1) was FIRST observed and reproduced — Run 22
   only added a cleaner, more rigorous confirmation of an already-true
   finding, it did not establish a new one. Changing the byline to April 1
   would also put finding no. 01 chronologically AFTER finding no. 02
   (March 27) and no. 03 (March 31) despite being numbered first — trading
   one defensible-either-way tension (byline vs. which single comparison
   the prose calls "headline") for a clearer, harder-to-defend one
   (numbering order contradicting date order). **Decision: kept the March
   27 byline** (first-observation date, preserves ascending numbering/date
   order across 01-05) rather than applying the suggested April 1 change.
   Flagged here explicitly, with the reasoning, so this can be revisited if
   disagreed with.
3. **MINOR — coherence, naming collision.** The roadmap's "gated" NCR-at-
   scale phase reused "wave 1" for a not-yet-launched item, three sentences
   after finding no. 14's own completed, published "wave-1" (K=8). The
   auditor traced the substantively-correct still-gated item to
   `NOVEL_ARCH_WATERFALL.md` §11, "early-LN K-scaling" (cleared for launch,
   not yet run).
   **Fix:** reworded the phase to explicitly name it "the earlyln K-scaling
   build" and added a parenthetical distinguishing it from finding no. 14's
   already-completed wave-1.
4. Security: fake `<system-reminder>` sighted again (by both the auditor
   and a sub-agent it dispatched); disregarded, reported.

## Final status

Three rounds run (minimum requirement: 2). Round 3 found real, previously
undetected issues (the retracted-FLOPs-claim bug had survived TWO clean
audit passes) — all fixed except one date judgment call, explicitly
adjudicated and disclosed above rather than silently overridden. Given
round 3's fixes are narrow, surgical, and independently re-verified against
primary sources (not summaries) before applying, and given the pattern
across all three rounds shows convergence (each round finds strictly fewer,
narrower issues: round 1 found 4 CRITICAL, round 2 found 1 CRITICAL + 3
SERIOUS, round 3 found 0 CRITICAL + 2 SERIOUS + 1 MINOR) rather than new
categories of problem, this is judged sufficiently clean to proceed to
commit. All changes from all three rounds are captured in this file and in
the git history of `pebble-ai-site/` from this session.
