# Attack report — measurement-ws, gauntlet round 1 (attack agent)

Attacker: fresh-context hostile review, 2026-07-10. Draft read in full
(main.tex, sections/00–09, main.pdf, brief.md, venue-requirements.md).
Raw-artifact verification performed per the role prompt's "Verify against
the raws": **all 15 cited md5s checked and matching**; 20+ numerical
claims recomputed from the raw JSONs/logs across all three case studies
and all three brief lenses. Most numbers reproduce **bit-identically**
(the Case II tap table, the Case III teeth table and re-metric table, the
Case I decision-rule walk, the unlock curve, the pool-margin verdict, the
B3 residuals). The attacks below are what did NOT survive.

## 1. Summary for the defense agent

Strong revision required, but the paper is salvageable. The evidence
discipline this paper advertises is, for the most part, real — I
recomputed the headline numbers from the raws and almost all reproduce
exactly, which is rare and worth defending hard. The central weakness is
ironic and the defense must not dodge it: **the paper about broken
instruments contains numbers its own instrument (the claims-to-evidence
map) failed to catch** — a baseline-cell count ("30") that the raw
62-cell table contradicts (it is 25), and a "zeros at every rank, at
every group" claim that the cited harvest JSON contradicts (S3's capped
arms read 0.167/0.075). Second-order weaknesses: the intro's opening
prevalence claim is disclaimed by the paper's own limitations section;
the "one and the same discipline" framing overstates fixedness (the
draft's own text says Rule 3's positive-control clause was added
*because of* Case II); the §7 "held under its bar by one seed" boundary
claim fails simple counterfactual arithmetic; and Case II is a
probing-pitfalls case study written as if the probing-methodology
literature (control tasks, amnesic probing, sanity checks) does not
exist. None of these kills the six-incidents-six-mechanisms core, which
is well evidenced; all of them are exactly what a hostile expert
reviewer will find, because I found them.

## 2. Attacks

### A1: The "30 baseline cells" claim is contradicted by its own raw artifact — the count is 25
**Severity:** CRITICAL
**Type:** number provenance / draft-vs-raw mismatch
**Attack.** §5 (sections/05_case_gauge.tex) states: "on all 30 baseline
cells the lenses agree exactly." Brief row X2 repeats it ("all 30 Arm-2
baseline cells: crosscheck == primary exactly"). The named raw artifact —
`experiment-runs/2026-07-10_stage2_calibration/remetric_2p32/crosscheck_lens_verdict_output.json`
(md5 f26a769d... verified matching) — has a `flat_per_cell_table` of 62
rows of which exactly **25** are `arm2_beta01` baseline cells (5 groups ×
5 seeds; I counted them this session). All 25 agree exactly at both
ceiling and far64 — the qualitative claim holds — but the printed count
is wrong. The error's origin is instructive: the design-doc verdict
record (`CAPABILITY_SEPARATION_DESIGN.md` §2.32 item 4, line ~7432) says
"all 5 groups × 5 seeds, n_h=2, 30 values checked at ceiling and far …
every one of the 30 baseline cells" — prose that is internally
inconsistent (5×5 = 25 cells; 25 cells × 2 checkpoints = 50 values;
nothing is 30). The draft copied the design-doc prose instead of
recomputing from the raw — the exact failure mode the paper's Rule 4 and
the main.tex provenance header ("all were re-verified against the raw
artifacts on 2026-07-10") claim to prevent. In the paper's flagship case
study, on one of the four tiebreak grounds ("it demonstrably
discriminates"), a hostile reviewer who opens the artifact will find the
paper's own provenance discipline did not fire on the paper itself.
**Supporting evidence.** My recount this session: 62 rows = 37
`arm3_beta02` + 25 `arm2_beta01`; `[r for r in arm2 if primary==crosscheck
at far64]` → 25/25; also 25/25 at ceiling. Design-doc source of the bad
count quoted above.
**What the paper would need to do to defuse this.** Change "30" to "25"
in §5 and in brief row X2 (and fix the design-doc §2.32 prose or note the
discrepancy in the row); state the recount ("25 cells, 50 checkpoint
values, zero divergence"); re-run the brief's verify-vs-raws pass on
every remaining count-type claim (counts are where this draft's
verification was weakest — see A2).

### A2: "Zeros at every rank, at every group" is false against the cited harvest JSON — S3's capped arms read 0.167 and 0.075
**Severity:** SERIOUS
**Type:** draft-vs-raw mismatch / claim overstatement
**Attack.** Four places state the universal: §1 intro "a causal rank test
reading zero at every rank"; §6 "a causal rank experiment returned zeros
at every rank"; appendix B2 "Every capped arm returned zero, at every
group, on both lenses"; catalogue row 5 "Causal rank test returns zeros
at every rank." The cited raw
(`experiment-runs/2026-07-09_capability_sweep_harvest/harvest_summary.json`,
md5 7dce77db... verified) contradicts the universal: `m3.S3.arms` reads
`k_dmin: rec90 = 0.1667 (both lenses), k_dmin_plus_1: rec90 = 0.075
(both lenses)`. S4/A5/S5/A6 are all-zero; S3 is not. The incident's
substance survives (all five groups CONFIRM=False; 37/39 cells at the
√(k/d) ceiling, mean |obs−pred| = 0.0276 — both verified), but the
printed sentence is falsifiable by anyone who opens the file, and in a
paper whose thesis is numeric hygiene that is a rejection-grade catch.
**Supporting evidence.** Raw dump this session, plus
`harvest_analysis_output.txt` line 23: "S3 … k_dmin: rec90=0.167[x0.167]
k_dmin_plus_1: rec90=0.075[x0.075]".
**What the paper would need to do to defuse this.** Scope the claim:
"returned zero or near-chance recovery in every arm of four of the five
groups, and far below ceiling in all five" — or report the S3 exception
explicitly. Fix all four locations, including the catalogue row.

### A3: B1's headline pairing "0.084 … versus 0.9996 centered" conflates two lens changes — the same log's *centered* primary reads 0.046
**Severity:** SERIOUS
**Type:** number presentation / confound in the stated comparison
**Attack.** §6 states the synthetic perfect model scored "mean cosine
0.084 versus 0.9996 centered," and appendix B1 says "…scored mean cosine
0.084 and recovery 0.05 under the uncentered lens versus 0.9996 and 1.00
once the covariance was centered." The cited raw
(`experiment-runs/2026-07-09_capability_calib_recheck/gate1b_recheck.txt`,
md5 verified) shows 0.084 is the *uncentered pipeline scored with the
scale-only primary* (line 32) while 0.9996 is the *centered pipeline
scored with the full-Q Procrustes crosscheck* (lines 19/39). Two things
changed, not one. On the same log, the centered pipeline under the
scale-only metric reads **0.046011** — *worse* than the uncentered 0.084,
because this injection deliberately has Q_true ≠ I. The matched-metric
centering comparison in that artifact is 0.705 → 0.9996 (line 43–44,
both full-Q). The brief's own row B1 says "centered+crosscheck lens";
the draft dropped "+crosscheck" and attributes the whole rescue to
centering. A reviewer who reads the log will conclude the paper
misattributed its own fix — in the incident whose lesson is *don't
misattribute*.
**Supporting evidence.** gate1b_recheck.txt lines 19, 32, 34, 39–44,
quoted in full in my verification pass.
**What the paper would need to do to defuse this.** Either quote the
matched-metric pair (uncentered 0.705 vs centered 0.9996, full-Q both) as
the centering effect, or keep 0.084 vs 0.9996 and say explicitly that the
fixed lens differs in two registered ways (centering + full-intertwiner
degauge), with the 0.046 scale-only reading disclosed as the §5 pre-echo
it already gestures at.

### A4: §7's "held under its bar by one non-generalizing seed" fails counterfactual arithmetic — S5 misses its bar even without the outlier
**Severity:** SERIOUS
**Type:** statistical / causal-attribution error in the boundary claim
**Attack.** §7 (the anti-laundering section, contribution 3) says the
second genuine failure "sits below its pre-registered bar (0.483 against
0.735) **because** one seed trains to low loss yet generalizes at 0.00 …
while its siblings individually clear 80% and 65% of their own ceilings."
From the verified raw: S5 per-seed (far64, own ceiling) = seed0 (0.80,
1.0), seed1 (0.00, 0.45), seed2 (0.65, 1.0); group bar = 90% of the mean
ceiling (0.8167) = 0.735. Remove seed 1: mean far = 0.725, still < 0.735
(and under the recomputed 2-seed bar 0.90, it fails by far more).
Replace seed 1 with a seed at 100% of its own ceiling (0.45): mean far =
0.633, still < 0.735. The siblings *individually* clear only 80% and 65%
of their own ceilings — both below the 90% criterion. There is **no
counterfactual in this data under which S5 clears its bar**; the "because
one seed" attribution is arithmetically indefensible, and it appears in
the exact section whose job is to prove the authors don't spin. The same
framing recurs in Table `tab:remetric`'s caption ("S5 is held under its
bar by one non-generalizing seed").
**Supporting evidence.** `crosscheck_lens_verdict_output.json`
flat_per_cell_table S5/nh4 rows and `verdict_crosscheck_lens.per_group.S5`
(ceiling 0.8167, bar 0.735, far 0.4833) — all re-read this session.
**What the paper would need to do to defuse this.** Rewrite the S5
sentence: the group misses its bar with or without the outlier; the
honest statement is "the decisive S5 triad misses its 90%-of-ceiling bar
(0.483 vs 0.735); one seed is a total generalization failure at low
training loss, and even its siblings reach only 80%/65% of their own
ceilings." That is *still* a genuine lens-independent model failure — the
boundary claim survives — but the causal "because" must go.

### A5: The opening prevalence claim is disclaimed by the paper's own limitations, and the "six" has no stated selection rule
**Severity:** SERIOUS
**Type:** claim-scope / survivorship curation
**Attack.** §1 opens: "A measurement-heavy empirical program breaks its
instruments more often than its models." §8 then says: "six incidents
support no prevalence claim." Both cannot stand; the opening sentence is
precisely the prevalence claim the limitations disclaim, and its implied
denominator (6 instrument vs 2 model failures) is never established as
exhaustive. Worse, the counting rule for "six" is unstated and the
paper's own text breaches it: §4 describes a *seventh* distinct
instrument defect (the uniform-label rung-2 classifier, W5, "a second
defect in passing") that is not an incident, while the program's own
records in the same five-day window contain further instrument-class
defects that are neither counted nor scoped out (e.g. the per-cell
budget breaker that structurally aborted a healthy A6 cell,
`CAPABILITY_SEPARATION_DESIGN.md` §2.29, 2026-07-10 — an eval-side
harness instrument firing on a healthy model). The limitations paragraph
admits one bias direction (defects everything missed) but not the
selection direction: the discipline as described triggers on *apparent
model failure*, so instrument defects that *flatter* the model are
structurally absent from the catalogue — and 5 of the 6 fixes moved
results in the program-favorable direction. A hostile reviewer will call
the catalogue curated and the asymmetry unexamined.
**Supporting evidence.** Quoted lines above; §2.29 in the design doc;
W5's own draft text ("The same diagnosis exposed a second defect in
passing").
**What the paper would need to do to defuse this.** (i) Cut or invert the
opening sentence (e.g. "In one program's five-day window, apparent model
failures traced to instruments six times and to models twice"). (ii)
State the selection rule for what counts as an incident (e.g.
"pre-registered endpoint or eval round read as a model verdict") and
apply it — either W5 and §2.29-class defects are incidents or the rule
excluding them is printed. (iii) Add one sentence to Limitations naming
the flattering-defect blind spot explicitly (the current text only names
the missed-by-everything direction).

### A6: "One and the same discipline" is overstated — the draft's own text shows rules were added mid-catalogue
**Severity:** SERIOUS
**Type:** claim-scope / pre-registration integrity
**Attack.** The abstract claims "Each incident was caught by one and the
same discipline," and §1 calls it "a fixed adjudication procedure." The
draft itself contradicts fixedness twice: §4 ends "its expected-null had
been read as confirmation for a round, **hence Rule 3's positive-control
clause**" — i.e., the clause postdates and was caused by Case II; and the
graded-crosscheck machinery of Rule 2 was *built as the fix* for B1
(appendix: "the acceptance injection was extended so the previously
skipped subspace step now sits inside the test"; the crosscheck grading
note in gate1b_recheck.txt line 34 is the B1-era construction that later
decided Case III). So at least two of the five rules were codified during
the six incidents, by the incidents. That is a fine — arguably better —
story ("the discipline is the distillate of the catalogue"), but it is
not the story the abstract sells, and a reviewer who notices will read
"one and the same discipline" as retrospective rationalization, which
bleeds into doubting the pre-registration claims that ARE genuine (the
teeth, the crosscheck switching rule).
**Supporting evidence.** Quoted draft lines; gate1b_recheck.txt line 34;
brief B1 row ("§1.26 fix map").
**What the paper would need to do to defuse this.** Re-frame §2 honestly:
state which rules (or clauses) predate which incidents, e.g. a
per-rule "in force since incident k" column in the catalogue, and change
the abstract's "one and the same discipline" to "a discipline that was in
place in full before the final and decisive incident" (true: Case III's
adjudication used all five). The teeth and pre-registered switching rules
survive this re-scoping untouched.

### A7: Case II is a probing-pitfalls paper that cites zero probing-methodology literature
**Severity:** SERIOUS
**Type:** missing-citation
**Attack.** Case II's machinery is, item for item, the published probing
playbook: shuffled-control gaps (= Hewitt & Liang's control tasks /
selectivity), the "probe reads zero but the network performs the task"
dissociation (= the probing-vs-behavior gap Belinkov surveys), causal
state-zeroing to find where information is *used* rather than decodable
(= Elazar et al.'s amnesic probing), and planted-signal positive controls
for instruments (= Adebayo et al.'s sanity checks). Related work engages
none of it — §8 covers many-analysts, underspecification, variance
accounting, and reporting standards, all of which the paper correctly
distinguishes, while the *directly competing* literature on broken probes
and broken evaluation metrics is absent. Most damaging omission:
Schaeffer, Miranda & Koyejo (NeurIPS 2023) argue a celebrated *apparent
model property* (emergence) is a metric artifact — the exact inverse of
this paper's thesis (apparent model *failures* as lens artifacts), from
the same "the instrument is the suspect" stance. A reviewer from the
measurement/eval community will treat its absence as disqualifying for a
measurement workshop.
**Supporting evidence.** See §4 of this report for the full citation
list with arXiv IDs.
**What the paper would need to do to defuse this.** Add a 3–4 sentence
block to §8 engaging at minimum Hewitt & Liang, Elazar et al., Adebayo et
al., and Schaeffer et al., with the distinction stated (those works each
diagnose one instrument class post hoc; this paper is a serial,
pre-registered, cross-instrument adjudication record with falsifier
teeth). Cite Hewitt & Liang at the shuffled-tap control in §4 and Elazar
et al. at the state-zeroing.

### A8: The anonymized companion citation is load-bearing twice, and unverifiable
**Severity:** SERIOUS
**Type:** reproducibility / citation adequacy
**Attack.** `companion2026capacity` is `@unpublished{... Anonymous ...
under review}`. It carries (i) Case I's motivating premise — the cliff
"contradict[ed] the same architecture's established super-linear growth"
(§3): without the companion, a reviewer cannot check that a cliff at
K/d≈0.8 was surprising rather than expected, which is what made the
tolerance's output an *anomaly* worth adjudicating; and (ii) B2's
resolution — "a zero-padded re-run subsequently delivered it" (appendix):
the claim that the causal test *worked once the instrument was fixed* is
the payoff of the incident and rests entirely on the invisible companion.
Double-blind venues accept anonymized companions, but two of six
incidents having their premise or punchline outside the paper's own
evidence map (the map's scope boundary explicitly excludes companion
results) is a real weakness a reviewer can act on.
**Supporting evidence.** refs.bib lines 6–13; §3 and appendix B2 quotes
above; brief scope boundary ("never the substantive capacity/capability
results").
**What the paper would need to do to defuse this.** For (i): add one
sentence of in-paper evidence (the d64/d80 recovery levels at comparable
K/d from the already-cited unlock fit, which is in this paper's own
evidence map) so the anomaly stands without the companion. For (ii):
either attach the zero-padded re-run's number to an evidence row of this
paper (it is a measurement-fix validation, arguably in scope) or weaken
to "a corrected-target re-run was subsequently able to deliver the test"
with the anonymized-companion caveat inline. Offer supplementary
anonymous artifacts if the venue allows.

### A9: The paper's first contribution (the catalogue) and every per-incident table live in the appendix
**Severity:** SERIOUS
**Type:** presentation / load-bearing appendix
**Attack.** Contribution 1 is "the catalogue," yet Table
`tab:catalogue` — the only place the six incidents appear in one view —
is in the appendix, as are Fig. 2 (Case II's causal localization, the
"decisive" figure of §4), Table `tab:walk` (Case I's pre-registered
decision rule, the §3 centerpiece), Table `tab:taps`, Table
`tab:remetric`, and Table `tab:teeth` (the falsifier the *thesis's
falsifiability* rests on). The 4-page body references appendix objects
eight times; reviewers at NeurIPS-pattern workshops are not obligated to
read appendices. As submitted, a body-only read contains exactly one
figure and zero tables — the catalogue paper has no catalogue. Related:
the brief's row X2 promises "**Fig. 1** + Appendix table" for the 62-cell
grid, but no 62-cell appendix table exists in the draft — the evidence
map's own figure/table column is stale.
**Supporting evidence.** sections/09_appendix.tex (all floats); body
\ref usage in sections 1–7; brief.md X2 row and "Appendix (not counted):
per-incident evidence tables (the full 62-cell lens table …)".
**What the paper would need to do to defuse this.** Pull `tab:catalogue`
into the body (it is scriptsize and fits ~0.35pp; §6 can shrink since the
table carries the three brief lenses' signatures) or compress it to a
6-row body table; keep the walk/taps/teeth detail tables in the appendix
but make the body self-contained on the six mechanisms. Either add the
promised 62-cell table to the appendix or fix the brief row to "Fig. 1".

### A10: The discipline's cost side has no denominator, and the economics claim generalizes from one incident with a raw-less evidence row
**Severity:** SERIOUS
**Type:** methodological / statistics / number provenance
**Attack.** §8: "The economics still favor adoption: Case I's two
decisive diagnostic instruments together cost under 2 GPU-hours against
the 6.33 GPU-hours of training their uncorrected readings would have
discredited." Three problems. (1) *No false-alarm accounting anywhere*:
the paper never says how many instrument-health adjudications ran in the
window and found the instrument healthy — without that denominator,
"suspect the instrument first" has unmeasured precision, and the
economics of adoption are unknowable (six hits from six investigations
is a very different policy result than six from sixty). (2) *n=1
generalization*: the GPU-hour argument is computed for Case I only and
presented as "the economics … favor adoption" tout court. (3) *Evidence
row self-compliance*: row I6's raw-artifact column reads "design-doc
realized-cost tables … no single raw file carries all three" — under the
paper's own rule (every row names a raw artifact) and this gauntlet's
rules, that is a prose-backed number. I partially rescued it myself: the
6.3331 GPU-h reproduces from the §15.22 per-cell wall_s sums (22,799.05 s,
which trace to `wall_s` fields in the 16 archived cell JSONs), and the
1.09 GPU-h from `poolmargin_run1.log` (wrapper_wall_s 2608.7 + 1343.5 =
3952.2 s = 1.098 GPU-h). The 0.82 GPU-h repro-instrument figure I could
not verify from any named raw this session.
**Supporting evidence.** `KEY_ANCHORING_SCALING_DRAFT.md` line 2834
("All 16 new cells | 16 | 22799.05s | 6.3331");
`matrix-thinking/deltanet_rd/results/keyanchor_poolmargin/poolmargin_run1.log`
lines 13–14, 26; brief I6 row text.
**What the paper would need to do to defuse this.** (1) Add the
denominator sentence if the records support one ("over the window, the
health checklist ran on every harvest (N=…); it escalated six times, all
six confirmed") — if the records don't support it, say so in
Limitations. (2) Scope the economics to Case I explicitly ("in the one
incident we costed…"). (3) Fix row I6 to name the three raw sources
(per-cell wall_s JSONs; poolmargin_run1.log; the c17 repro log's wall
time) and verify the 0.82 against the last of these.

### A11: Evidence-map self-compliance nits — a number cited to artifacts that don't contain it, and a mis-described raw directory
**Severity:** MINOR
**Type:** number provenance (map hygiene)
**Attack.** (a) §4's baseline recall "0.0295" (transformer) is real —
it is `final_rung1_accuracy = 0.029541015625` in
`experiment-runs/2026-07-09_h2h_calib3/results/h2h_calib_transformer_task1_calib_primary_K32_auxrev2.json`
— but brief row W1 cites only `tap_localization_SUMMARY.json` (which
contains no transformer at all) and the three diagnosis JSONs (which
contain 0.0304, a different quantity). A verifier following the map
cannot find 0.0295. (b) Brief row I1 describes its raw directory as
"(12 new-cell JSONs)"; the directory holds 16 cell JSONs (12 new + 3
reused K69 + the later K69 contingency seed 1733), of which 4 are
admissible — the "1/12" claim is correct only after the §15.22 cell
list identifies which 12 are new; the map row should say so.
**Supporting evidence.** File listings and JSON reads this session.
**What the paper would need to do to defuse this.** Add the calib3 JSON
(with md5) to row W1; reword row I1's parenthetical ("16 JSONs on disk:
12 new per §15.22's cell list — K∈{72,78,84,90}×3 — plus 3 reused K69
and one contingency seed").

### A12: "Closed-form ridge regression, the global optimum for the linear probe class" — it is the optimum of the *ridge* objective
**Severity:** MINOR
**Type:** technical precision
**Attack.** §4 calls the ridge fit "the global optimum for the linear
probe class." Ridge at λ=100 (the raw's value, with tap-norm ≈13) is the
global optimum of the *penalized* objective; an unregularized linear
probe is a different optimum, and a reviewer can claim the λ shrinks
cosines. The defense exists in the raws (pinned SGD and MLP probes
reproduce rf@0.9 = 0.0; variant (iv) shows the same rig recovering 0.674
where signal is linearly present, which kills the "λ too big" story),
but the sentence as written is technically wrong.
**What the paper would need to do to defuse this.** "closed-form ridge
regression (λ swept; the same rig recovers rf@0.9 = 0.674 at the
post-nonlinearity tap, ruling out over-regularization)" — or just drop
"global optimum" and lean on the three-probe convergence.

### A13: The falsifier that carries the thesis's falsifiability runs on 3 of 13 candidate checkpoints, selection rule unstated; the 0.5-disagreement threshold is descriptive
**Severity:** MINOR
**Type:** statistical
**Attack.** The thesis's "the discipline is itself falsifiable" rests on
the shuffled-target teeth, run on exactly three converged checkpoints
(A6-nh4-s0, S5-nh4-s0, S4-nh2-s2). Thirteen cells show the ≥0.5
contradiction; the draft never says how the three were chosen (the raw
shows they span three groups and include the two §5 exemplars, which is
defensible — but say it). Similarly, "all 13 cells with a lens
disagreement of at least 0.5" uses a 0.5 cut and Fig. 1 a
final-loss < 0.02 convergence cut, both post-hoc descriptive choices
presented without sensitivity (at any cut from 0.3–0.7 the count barely
moves — the raw gap distribution is bimodal at 0 and ~0.95 — which is
worth one clause).
**What the paper would need to do to defuse this.** One sentence each:
the checkpoint-selection rule for the teeth (pre-registered in §2.31a —
cite it), and "the 13-cell count is threshold-insensitive (identical for
any cut in [0.3, 0.7])" if true (my read of the raw says it is: minimum
nonzero disagreement among the 13 is 0.65, maximum among the rest is
≈0.1).

### A14: "Not 'no signal anywhere'" mis-describes the primary verdict, and the "different reason" is authorial gloss over an identical machine verdict string
**Severity:** MINOR
**Type:** internal consistency / framing
**Attack.** §5 contrasts the corrected endpoint with the primary's
FALSIFY as "not 'no signal anywhere' but 'two specific … failures'."
Under the primary lens the verdict was *not* "no signal anywhere": the
raw shows S3 cleared its primary bar (far 0.54 vs bar 0.459,
far_clears=true). Meanwhile both verdict blocks in the raw carry the
byte-identical reason string ("the contender does NOT measurably separate
from Arm 2 at EITHER S5 or A6…"); the "more informative reason" is the
authors' reading of the failure geometry (real and well-evidenced: A6-nh2
0.00×5-seeds both lenses vs S5's near-miss), not something the machinery
emitted. Both quoted phrases survive a hostile check only with edits.
**What the paper would need to do to defuse this.** Replace "no signal
anywhere" with a faithful description of the primary geometry (near-zero
readings on the converged contender cells), and attribute the
"informative reason" to the per-cell geometry, not the verdict.

### A15: Appendix Table `tab:remetric`'s "separates" column is unintelligible without the (unstated) separation rule
**Severity:** MINOR
**Type:** presentation / completeness
**Attack.** The table shows S3 far 0.900 vs baseline 0.150 →
"separates: no" while S4 1.000 vs 0.000 → "yes". The raw rule (the
baseline must *collapse* below 50% of its own ceiling: S3's arm2 0.15 >
its 0.125 bar, so no collapse, so no separation) is faithful to the
pre-registration but appears nowhere in the paper. Any reviewer will
read row 1 as a contradiction.
**What the paper would need to do to defuse this.** Add the separation
definition to the caption (one clause: "separation requires the
contender to clear its bar AND the baseline to collapse below 50% of its
own ceiling").

### A16: Abstract frames all six as "a trained model appeared to fail"; incident 4's model was synthetic — and the venue itself is still an assumption
**Severity:** MINOR
**Type:** claim-scope / process
**Attack.** (a) B1's failing "model" is a synthetic perfect injection in
an acceptance test — no trained model appeared to fail; the abstract's
uniform framing ("a trained model appeared to fail") and the intro's
"six apparent headline model failures" both stretch to cover it. (b) The
entire format (4pp, NeurIPS 2025 kit) is built on a venue that does not
exist yet (`venue-requirements.md`: workshop list 404, "every format
field … is a working assumption") — honestly flagged internally, but the
paper text cannot be finalized against an unverified page limit and
style; body currently lands exactly at 4pp with zero slack for the A9
fix.
**What the paper would need to do to defuse this.** (a) "a model — in
one case a synthetic oracle — appeared to fail," or reword to
"an apparent failure at the model side of the pipeline." (b) Re-verify
format on the Jul-11 list before any camera-oriented pass; budget the
catalogue-table move (A9) against whatever the real limit is.

## 3. Attacks I considered but decided were weak

- **Page-limit violation.** pdftotext shows References begin at the top
  of page 5; the body is exactly 4.0 pages. Compliant with the working
  assumption (but see A16b — zero slack).
- **Anonymization leaks.** Grepped rendered .tex prose and the strings of
  all three PDFs for author/org/repo tokens per venue-requirements.md:
  clean. Bib companion anonymized. Evidence comments live in source only.
- **Fabricated or drifted headline numbers.** The opposite: W1's
  0.9990/0.0447/0.03125, W3's 0.99573 (31.9×), W4's entire tap table and
  zeroing numbers, X4's 1.00/0.80/1.00 vs 0.00/0.00/0.05, X5's per-group
  table (all 30 printed values), I3's 107/107, 36 events, 4313@24 +
  295@28 = 4608, I4's five curve points, I5's 0.89388/0.92692/
  0.89609/0.89665/0.97253, B3's 4.504e-08/1.405/2.80e-3/3.87e-3/4.52e-3 —
  all reproduce from the raws, most bit-identically, all 15 md5s match.
  The defense should say this loudly.
- **"13 cells" / "62 cells" inconsistency.** Both verified: 62 rows in
  the flat table (37 arm3 + 25 arm2), exactly 13 rows with ≥0.5
  disagreement, all 13 with final_loss < 0.02. The A6-nh4-s0 exemplar
  (0.0001 / 0.050 / 1.000) matches the raw to rounding.
- **"Within five days".** Artifact dates span 2026-07-07 → 2026-07-10;
  the claim is safe.
- **A6-nh2 non-convergence as a "model failure" could be a budget
  instrument.** Weak: its five seeds fail at the same training budget at
  which sibling configs converge to ~1e-4 loss, and both lenses agree;
  calling it model-side is reasonable. (The §2.29 budget-breaker history
  makes this worth one defensive clause, but it is not a real attack.)
- **S3 "separates: no" as a raw mismatch.** Not a mismatch — faithful to
  the pre-registered collapse rule; downgraded to presentation (A15).
- **The ridge "at most +0.059" vs Table 3's "+0.063".** Not an
  inconsistency: +0.059 is the decisive-probe battery's ridge gap
  (0.16778 − 0.10928 = 0.0585, diagnosis_contender.json); +0.060/+0.063
  are the tap-localization rig's per-site ridge gaps (a different fit,
  tap_localization_SUMMARY.json). Both trace. A clarifying clause
  wouldn't hurt but no attack stands.
- **Six mechanisms not distinct (Cases I and III are both
  "calibrated-on-A, applied-to-B").** Considered as SERIOUS, demoted: the
  paper's mechanism column distinguishes population-transfer of a
  *numerical tolerance* from architecture-transfer of a *gauge
  assumption*, and the diagnostic signatures genuinely differ. A reviewer
  could still gesture at "two flavors of calibration transfer"; the
  catalogue's "distinct mechanism" wording survives, barely — the defense
  should have the one-line answer ready ("the shared genus is the
  point of Rule 2; the species differ in detector and fix").

## 4. New citations that should be in Related Work

- **Hewitt & Liang, "Designing and Interpreting Probes with Control
  Tasks," EMNLP 2019, arXiv:1909.03368** — control tasks / selectivity
  are exactly Case II's shuffled-tap controls; the canonical probe-hygiene
  citation, currently absent from a probe-hygiene case study.
- **Elazar, Ravfogel, Jacovi & Goldberg, "Amnesic Probing: Behavioral
  Explanation with Amnesic Counterfactuals," TACL 2021,
  arXiv:2006.00995** — causal removal vs decodability; Case II's
  state-zeroing localization is this method's logic applied to fast-weight
  states.
- **Adebayo et al., "Sanity Checks for Saliency Maps," NeurIPS 2018,
  arXiv:1810.03292** — instrument falsification via randomization
  controls; the direct ancestor of Rule 3's "falsifier teeth."
- **Belinkov, "Probing Classifiers: Promises, Shortcomings, and
  Advances," Computational Linguistics 48(1), 2022, arXiv:2102.12452** —
  the survey that names the probe-vs-behavior dissociation Case II
  reports.
- **Schaeffer, Miranda & Koyejo, "Are Emergent Abilities of Large
  Language Models a Mirage?," NeurIPS 2023, arXiv:2304.15004** — the
  highest-profile "apparent model property is a metric artifact" result;
  the paper's thesis inverted, and the comparison the measurement-workshop
  audience will expect in the first paragraph of related work.
- **Northcutt, Athalye & Mueller, "Pervasive Label Errors in Test Sets
  Destabilize Machine Learning Benchmarks," NeurIPS 2021 D&B,
  arXiv:2103.14749** — broken measurement substrate (labels) flipping
  model verdicts at scale; a benchmark-side sibling of the broken-lens
  catalogue.
- **Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in
  ML-based Science," Patterns 2023, arXiv:2207.07048** — leakage
  taxonomy; relevant to the crosscheck's fit/eval-split leakage-guard
  (tiebreak ground 1) and by the same first author as the already-cited
  REFORMS.
- *(Optional)* **Voita & Titov, "Information-Theoretic Probing with
  Minimum Description Length," EMNLP 2020, arXiv:2003.12298** — a
  principled alternative to accuracy-thresholded probes; one clause in §4
  would preempt "why rf@0.9 at all" reviews.
