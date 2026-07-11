# Defense report — measurement-ws, gauntlet round 1 (defense agent)

Defense: fresh-context author-side disposition, 2026-07-10. Draft read in
full (main.tex, sections 00–09, brief.md, venue-requirements.md). Every
consequential factual premise re-verified this session directly against
the raw JSON/logs (not taken on the attacker's word): the 25-vs-30 cell
count, S3's non-zero capped arms, the gate1b centering/metric confound,
the S5 counterfactual arithmetic, S3 clearing its *primary* bar, the
byte-identical primary/crosscheck reason strings, the 13-cell threshold
sweep, and the location of the 0.0295 transformer number. Results and the
one place the *attacker itself* is numerically wrong are in each entry.

## 1. Summary for the rebuttal agent

**Disposition counts: 0 DEFEND, 3 PARTIAL, 13 CONCEDE + FIX.** This is an
honest tally, not a capitulation: the attacker did real raw-artifact work
and almost every catch reproduces when I open the same file. Zero DEFEND
is the correct outcome when the attacks are correct — I looked hard for a
clean factual disproof on the CRITICAL (A1) and did not find one; the
paper's own recount contradicts it. **None of the fixes is a new
experiment; all are framing, citation, presentation, or number-scoping
edits, so the paper is submittable after the fix wave** (subject to the
external Jul-11 venue confirmation, A16b). The load-bearing fixes, in
order of how much they change acceptance odds: **(A7)** add the
probing-methodology literature — a probing-pitfalls case study that cites
none of Hewitt & Liang / Elazar / Adebayo / Schaeffer will read as
not-engaging-the-field to exactly the measurement-workshop reviewer this
targets; **(A1/A2/A11/A14)** the four number/count corrections the paper's
own provenance discipline should have caught (30→25, "zeros at every
rank"→S3 exception, the 0.0295 map row, "no signal anywhere"→S3 cleared
its primary bar); **(A9)** pull the catalogue table into the body so the
catalogue paper has a catalogue; **(A4/A5/A6/A10)** the four
honest-framing repairs in the anti-spin and discipline sections. Two
tensions the rebuttal must weigh together: the body is at 4.0pp with zero
slack, and A7 (citations) + A9 (catalogue-in-body) both *add* — so the
page-limit confirmation (A16b) and the cut plan are now coupled. One
place I overturn the attacker: **A13's proposed remedy sentence is
factually false** — the 13-cell count is *not* threshold-invariant (it is
16/14/13/12/11 across cuts 0.3–0.7); do not paste that clause in.

## 2. Defenses

### A1: The "30 baseline cells" claim is contradicted by its own raw artifact — the count is 25
**Disposition:** CONCEDE + FIX (framing — one number; then a verification sweep)
**Response.** Confirmed by my own recount, not the attacker's. The
`flat_per_cell_table` has 62 rows = 37 `arm3_beta02` + **25** `arm2_beta01`
(5 groups × 5 seeds, all n_h=2). All 25 agree exactly between primary and
crosscheck at both the ceiling and far64 checkpoints (25/25 and 25/25).
The qualitative claim ("the lenses agree exactly on every baseline cell")
is true; the printed count "30" is wrong. There is no numerically stable
routine to point at here — the design-doc prose the draft copied is
internally inconsistent (it says "5 groups × 5 seeds … 30," but 5×5=25),
and the draft inherited the arithmetic error instead of recomputing from
the raw. That is the one failure mode this paper's Rule 4 and provenance
header exist to prevent, occurring on the paper itself, so I do not
contest the high severity — but the *fix* is a one-character edit plus a
verification sweep, not a re-run.
**Supporting evidence.** This session:
`crosscheck_lens_verdict_output.json` (md5 f26a769d…, matches) →
`Counter(arm)` = {arm3_beta02: 37, arm2_beta01: 25}; arm2 groups
{A5,A6,S3,S4,S5}×{seed0..4}; agreement 25/25 at far64 and 25/25 at
ceiling. Only one "30" occurs in the draft prose: `sections/05_case_gauge.tex:26`.
**What goes in the paper if this defense is accepted.** In
`sections/05_case_gauge.tex:26` change "on all 30 baseline cells the
lenses agree exactly" → "on all 25 Arm-2 baseline cells (5 groups × 5
seeds, 50 checkpoint values) the lenses agree exactly, zero divergence."
Fix brief row X2 identically. As the tail of this fix, re-run the
verify-vs-raws pass over **every count-type claim** (A2, A11, A14 show
counts are where the draft's verification was weakest) and note or correct
the design-doc §2.32 prose.

### A2: "Zeros at every rank, at every group" is false against the cited harvest JSON — S3's capped arms read 0.167 and 0.075
**Disposition:** CONCEDE + FIX (framing — scope the universal)
**Response.** Confirmed by direct read of `harvest_summary.json`. S3's
capped arms are `k_dmin: rec = 0.1667 (xrec 0.1667)` and `k_dmin_plus_1:
rec = 0.075 (xrec 0.075)` — non-zero on *both* lenses. S4/A5/S5/A6 are
all-zero at all three cap points on both lenses. So the universal wording
("every capped arm returned zero, at every group") is falsifiable by
anyone who opens the file, in four places. The incident's substance is
untouched: all five groups read `confirm = False`, S3 is `near_chance
_below = True`, and 37/39 force-rank cells sit at the √(k/d) ceiling — the
"undelivered by the instrument, not failed by the model" reading survives
whole. Only the false universal has to go.
**Supporting evidence.** `harvest_summary.json` (md5 7dce77dc…, matches
the brief) m3 block, this session: S3 {kdmin-1: 0/0, kdmin: 0.1667/0.1667,
kdmin+1: 0.075/0.075}; S4/A5/S5/A6 all 0/0/0. `verdict = INCONCLUSIVE,
m3=false`.
**What goes in the paper if this defense is accepted.** Rescope in all
four locations (§1 intro, §6, appendix B2, catalogue row 5): "a causal
rank test that returned near-chance recovery in four of five groups and
far below the recruitment ceiling in all five" — or state the S3 exception
in one clause ("S3's capped arms read 0.17/0.08, still near-chance"). The
catalogue-table row must change too, not only the prose.

### A3: B1's "0.084 … versus 0.9996 centered" conflates two lens changes — the same log's centered primary reads 0.046
**Disposition:** CONCEDE + FIX (framing — de-confound the number pairing)
**Response.** Confirmed against `gate1b_recheck.txt` line by line. The
0.084 is the *uncentered* pipeline scored with the *scale-only* metric
(line 32); the 0.9996 is the *centered* pipeline scored with the *full-Q
Procrustes crosscheck* (lines 19/39). Two registered axes change between
those two numbers, not one — yet §6 attributes the whole rescue to the
word "centered." The tell is in the same log: the *centered* pipeline
under the scale-only metric reads **0.046** (line 19), i.e. *worse* than
uncentered-0.084, because this injection deliberately sets Q_true≠I; the
matched-metric isolation of centering (both full-Q) is **0.705 → 0.9996**
(line 43 uncentered 0.705, line 44 gap 0.294). The paper picked the
maximally favorable cross-comparison and pinned it on centering, in the
incident whose lesson is "don't misattribute a fix." The appendix (B1) is
already more honest than §6 — it discloses the 0.046 scale-only reading —
which is why this is a bounded framing fix and not a substance problem,
but §6's parenthetical as printed is a genuine confound.
**Supporting evidence.** `gate1b_recheck.txt` (md5 2d170cc0…): line 19
centered `mean_cos=0.046011 … crosscheck(full-Q) 0.999594`; line 32
uncentered `mean_cos=0.084111`; line 43 `uncentered mean_cos=0.705261`;
line 44 `centered-vs-uncentered gap … 0.294334`. brief row B1 says
"centered+crosscheck lens"; §6 dropped "+crosscheck."
**What goes in the paper if this defense is accepted.** In §6 either (a)
report the matched-metric centering effect — "an uncentered covariance …
under which the full-intertwiner recovery fell from 0.9996 to 0.705
(0.084 under the scale-only lens the old harness used)" — or (b) keep
0.084 vs 0.9996 and add "(the fixed lens differs in two registered ways,
centering and a full-intertwiner degauge; the centered scale-only reading
is 0.046, the §5 pre-echo)." Option (a) is cleaner.

### A4: §7's "held under its bar by one non-generalizing seed" fails counterfactual arithmetic
**Disposition:** CONCEDE + FIX (framing — kill the causal "because"; the boundary claim survives)
**Response.** Confirmed arithmetically from the raw. S5's decisive n_h=4
triad, crosscheck far64: seed0=0.80, seed1=0.00, seed2=0.65; own ceilings
1.0/0.45/1.0; group far=0.4833, ceiling=0.8167, bar=0.735 (=0.9×0.8167).
The "because one seed" attribution is false under every counterfactual:
drop seed1 → mean far of the two siblings = (0.80+0.65)/2 = **0.725 <
0.735** (and against a 2-seed 90%-bar of 0.90 it fails by far more);
replace seed1 with a seed at 100% of its own ceiling (0.45) → mean far =
1.9/3 = **0.633 < 0.735**. The siblings individually reach only 80% and
65% of their own ceilings, both under the 90% criterion, so the group
misses its bar *independently of the outlier*. Crucially this makes the
boundary claim **stronger**, not weaker — S5 is a robust, lens-
independent model failure — but the causal sentence in the exact section
whose job is to prove no spin must be rewritten.
**Supporting evidence.** `crosscheck_lens_verdict_output.json`:
S5/arm3/nh4 rows (0.8/0.0/0.65, ceilings 1.0/0.45/1.0) and
`verdict_crosscheck_lens.per_group.S5` (ceiling 0.8167, far_bar 0.735, far
0.4833, arm2 far 0.0, far_clears False) — all re-read this session.
**What goes in the paper if this defense is accepted.** Rewrite the §7
sentence and the `tab:remetric` caption: "S5's decisive triad misses its
90%-of-ceiling bar (0.483 vs 0.735) lens-independently: one seed is a
total generalization failure at low training loss (0.00 both lenses), and
even its two siblings reach only 80% and 65% of their own ceilings —
below the 90% criterion — so the group fails with or without the outlier."
Delete "held under its bar by one non-generalizing seed" from the caption.

### A5: The opening prevalence claim is disclaimed by the paper's own limitations, and "six" has no stated selection rule
**Disposition:** CONCEDE + FIX (framing — three edits)
**Response.** The tension is real and internal. §1 opens "A
measurement-heavy empirical program breaks its instruments more often
than its models" — a prevalence claim with an implied 6-vs-2 denominator —
while §8 states "six incidents support no prevalence claim." Both cannot
stand as written. The unstated counting rule is the deeper issue: the
discipline as described triggers on *apparent model failure*, so
instrument defects that *flatter* the model are structurally absent — and
the draft's own W5 ("its expected-null had been read as confirmation for a
round") is exactly such a flattering defect, caught only incidentally and
not counted as an incident. The Limitations paragraph names one bias
direction (defects everything missed) but not this selection asymmetry. I
concede all three sub-points; the fixes are prose.
**Supporting evidence.** `sections/01_intro.tex:4` (opening) vs
`sections/08_related_limits.tex:24` ("no prevalence claim");
`sections/04_case_wronglayer.tex:48-52` (W5 as a false-confirmation
defect, not a failure). The program record's §2.29 budget-breaker is a
further eval-side instrument defect in the same window, uncounted.
**What goes in the paper if this defense is accepted.** (i) Replace the
opening with a program-scoped statement: "In one program's five-day
window, six apparent headline model failures traced to instruments and two
to models." (ii) Print the incident selection rule ("a pre-registered
endpoint or eval round read as a model verdict") so W5 and §2.29-class
defects are visibly out of scope by rule, not by omission. (iii) Add one
Limitations sentence naming the flattering-defect blind spot: "defects
that flatter the model are under-represented, because the discipline
triggers on apparent failure; five of the six fixes moved results in the
program-favorable direction."

### A6: "One and the same discipline" is overstated — rules were added mid-catalogue
**Disposition:** CONCEDE + FIX (framing — rescope; the substance survives intact)
**Response.** The draft contradicts its own "fixed"/"one and the same"
framing. §4 says Rule 3's positive-control clause exists *because* Case II
misread an expected-null ("hence Rule 3's positive-control clause"), and
the graded-crosscheck of Rule 2 was constructed as the B1 fix. So at least
two clauses postdate the incidents they were built from; Case II was not
caught by the full five-rule discipline, it *contributed* a rule. This is
a better story than the one the abstract sells ("the discipline is the
distillate of the catalogue"), and the attacker's own concession holds:
the full discipline *was* in force before the final, decisive Case III, so
"Case III was adjudicated by all five rules" is true and untouched. I
concede the overstatement because the paper's own text falsifies "fixed."
**Supporting evidence.** `sections/04_case_wronglayer.tex:51-52`;
`sections/02_discipline.tex` Rule 2/Rule 3 text; brief B1 row ("§1.26 fix
map"). No raw contradicts this — it is a self-consistency catch.
**What goes in the paper if this defense is accepted.** Change the
abstract's "Each incident was caught by one and the same discipline" →
"the discipline was assembled across the catalogue and in full force
before the final, decisive incident," and add a per-rule "in force since"
note to §2 (Rule 3's positive-control clause: since Case II; Rule 2's
graded crosscheck: since the covariance incident). The teeth and the
pre-registered switching rule are unaffected.

### A7: Case II is a probing-pitfalls case study that cites zero probing-methodology literature
**Disposition:** CONCEDE + FIX (framing — citations + one distinguishing paragraph)
**Response.** Confirmed against `refs.bib`: the ten non-companion entries
are schlag, nichani, silberzahn, damour, henderson, bouthillier, kapoor,
sculley, lipton — *none* is probing methodology. Case II's machinery is
the published probe-hygiene playbook item for item (shuffled-tap controls
= Hewitt & Liang control tasks/selectivity; probe-reads-zero-but-network-
performs = the probing-vs-behavior gap Belinkov surveys; causal state-
zeroing = Elazar et al. amnesic probing; planted-signal positive controls
= Adebayo et al. sanity checks), and Schaeffer et al. ("emergence is a
metric mirage") is the exact inverse thesis this paper's audience will
expect named in the first related-work paragraph. I judge this the single
most acceptance-relevant attack in the report for the target venue — more
than the CRITICAL A1 — because it is the omission a measurement/eval
reviewer treats as disqualifying. The distinction is real and defensible
(each prior work diagnoses one instrument class post hoc; this is a
serial, pre-registered, cross-instrument adjudication record with
falsifier teeth), so this costs citations and prose, not a claim.
**Supporting evidence.** `refs.bib` key list (this session). Attack §4
citation list with arXiv IDs (reproduced under New Citations below).
**What goes in the paper if this defense is accepted.** Add a 3–4-sentence
block to §8 engaging Hewitt & Liang, Elazar et al., Adebayo et al., and
Schaeffer et al., stating the serial/pre-registered/teeth distinction;
cite Hewitt & Liang at the §4 shuffled-tap control and Elazar et al. at
the §4 state-zeroing. This addition competes for space with A9 — see the
page-budget note in A16b.

### A8: The anonymized companion citation is load-bearing twice, and unverifiable
**Disposition:** PARTIAL (soften the two companion-dependent sentences; add the in-map anchor where one exists)
**Response.** The attack is right that `companion2026capacity` carries two
things a reviewer cannot check inside the paper: (i) Case I's premise (the
d96 cliff "contradicted the same architecture's established super-linear
growth" — the surprise that made the tolerance output an *anomaly*) and
(ii) B2's punchline ("a zero-padded re-run subsequently delivered it").
But I decline to escalate this to CONCEDE+FIX: anonymized companions are
standard and accepted at double-blind venues, and the *incidents
themselves* are fully evidenced in this paper's own map — only their
framing leans out. For (i) the anomaly can be anchored partly in-paper:
this paper's own unlock fit (I4) shows d96 has **no cliff** through
K/d=0.9375, which is itself the evidence that a cliff would have been
surprising. So the fix is a scoping/softening edit plus one in-map anchor,
not a new artifact.
**Supporting evidence.** `refs.bib` companion entry (anonymized,
"under review"); `sections/03_case_tolerance.tex:15` (companion cite in
the premise); `sections/09_appendix.tex:35-36` (companion cite in the B2
punchline); brief I4 row (in-map unlock fit).
**What goes in the paper if this defense is accepted.** (i) Add half a
sentence to §3 grounding the anomaly in this paper's own evidence ("the
same wide grid unlocked to no cliff through K/d=0.9375 once the tolerance
was fixed — §3 — so the apparent cliff was surprising against the
architecture's own behavior"), keeping the companion cite as corroboration
not sole support. (ii) Weaken the B2 punchline to "a corrected-target
re-run was subsequently able to deliver the test (companion work,
anonymized)," or attach the re-run's number as an in-scope measurement-fix
evidence row if the venue permits supplementary artifacts.

### A9: The first contribution (the catalogue) and every per-incident table live in the appendix
**Disposition:** CONCEDE + FIX (presentation — pull the catalogue into the body; fix a stale brief row)
**Response.** Confirmed. Contribution 1 is "the catalogue," but
`tab:catalogue` — the only single-view listing of the six incidents — is
in `sections/09_appendix.tex`, along with Fig. 2 (Case II's decisive
causal figure), `tab:walk`, `tab:taps`, `tab:remetric`, and `tab:teeth`.
A body-only read (which NeurIPS-pattern workshop reviewers are entitled
to) contains one figure and zero tables: the catalogue paper has no
catalogue in the body. Separately, brief row X2 promises "**Fig. 1** +
Appendix table" for the 62-cell grid, but no 62-cell table exists anywhere
in the draft — the map's own figure/table column is stale.
**Supporting evidence.** `sections/09_appendix.tex` (all six floats live
here); body `\ref` usage across §§1–7; brief X2 "Fig. 1 + Appendix table"
and the "Appendix (not counted): … the full 62-cell lens table" line —
neither realized.
**What goes in the paper if this defense is accepted.** Move
`tab:catalogue` (scriptsize, ~0.35pp) into the body, or compress it to a
6-row body table; keep the walk/taps/teeth detail tables in the appendix.
Fix brief row X2 to "Fig. 1" (drop the nonexistent appendix table) or
actually add the 62-cell table to the appendix. This addition trades
against A7 and the 4pp ceiling — flag for the page-budget decision.

### A10: The discipline's cost side has no denominator, and the economics generalize from one incident with a raw-less row
**Disposition:** CONCEDE + FIX (framing — scope, add the denominator or admit it, fix the map row)
**Response.** Three sub-points, all conceded. (1) *No false-alarm
denominator anywhere.* The paper advocates a policy ("suspect the
instrument first") but never says how many instrument-health adjudications
ran in the window and confirmed the instrument healthy — without it the
policy's precision is unmeasured, and six-hits-from-six is a very
different result from six-from-sixty. This is the substantive part and is
genuinely SERIOUS for a paper recommending a practice. (2) *n=1
generalization.* The GPU-hour argument is computed for Case I only yet
stated as "the economics still favor adoption" unqualified. (3) *Row I6 is
prose-backed.* Its own raw column reads "no single raw file carries all
three" — the one evidence row that violates the paper's Rule 4. I verified
the 1.09 GPU-h (poolmargin wrapper_wall_s 2608.7+1343.5 = 3952.2s =
1.098h) and the attacker verified the 6.33h; the **0.82 GPU-h repro-
instrument figure I could not source from any named raw this session
either** — so I6 is genuinely under-provenanced.
**Supporting evidence.** `sections/08_related_limits.tex:27-31`;
`poolmargin_run1.log` lines 13/26 (wrapper_wall_s 2608.7, 1343.5, GPU-pinned);
`KEY_ANCHORING_SCALING_DRAFT.md` §15.22 (22799.05s = 6.3331h per the
attacker); 0.82h unlocated in `experiment-runs/2026-07-08_c17_repro/`.
**What goes in the paper if this defense is accepted.** (1) Add the
denominator if records support it ("the health checklist ran on every
harvest in the window, N=…; it escalated K times, all confirmed"); if they
don't, say so in Limitations rather than imply precision. (2) Scope: "in
the one incident we costed, the two diagnostic instruments cost under 2
GPU-hours against 6.33 of training." (3) Fix map row I6 to name the three
raw sources (per-cell wall_s JSONs; poolmargin_run1.log; the c17 repro
log) and either source or drop the 0.82.

### A11: Evidence-map self-compliance nits — a number cited to artifacts that don't contain it, and a mis-described raw directory
**Disposition:** CONCEDE + FIX (framing — map hygiene)
**Response.** Both confirmed. (a) §4's transformer baseline recall
**0.0295** is `final_rung1_accuracy = 0.029541015625` in the calib3 JSON —
which brief row W1 does **not** cite; W1 points at
`tap_localization_SUMMARY.json` (no transformer field) and the three
diagnosis JSONs, whose nearest transformer quantity is 0.0304 (the LM-head
route = W3's number, not the recall). A verifier following the map cannot
find 0.0295. (b) Row I1 calls its directory "(12 new-cell JSONs)"; it
holds 16 (12 new + 3 reused K69 + one contingency seed). These are pure
map-hygiene fixes and change no claim.
**Supporting evidence.** This session: `h2h_calib3/results/h2h_calib
_transformer_task1_calib_primary_K32_auxrev2.json` →
`final_rung1_accuracy = 0.029541`; `diagnosis_transformer.json` q_pos
route = 0.03040 (not the recall).
**What goes in the paper if this defense is accepted.** Add the calib3
JSON (with md5) to brief row W1 as the source of 0.0295; reword row I1's
parenthetical to "16 JSONs on disk: 12 new per §15.22's cell list plus 3
reused K69 and one contingency seed." No draft-text change needed.

### A12: "Closed-form ridge regression, the global optimum for the linear probe class" — it is the optimum of the ridge objective
**Disposition:** CONCEDE + FIX (framing — one clause)
**Response.** Technically correct catch. Ridge at λ≈100 is the global
optimum of the *penalized* objective, not of the unregularized linear
probe class, so the phrase as written invites a "the λ shrank the cosines"
rebuttal. The paper's own raws already kill that rebuttal (variant (iv)
recovers rf@0.9=0.674 with the same rig where signal is linearly present,
and pinned SGD/MLP probes reproduce 0.0), but the sentence should not
assert a false optimality. This is a MINOR precision fix.
**Supporting evidence.** `sections/04_case_wronglayer.tex:18`; `tab:taps`
row (iv) 0.674; brief W2 (SGD/MLP reproduce 0.0).
**What goes in the paper if this defense is accepted.** In
`sections/04_case_wronglayer.tex` replace "the global optimum for the
linear probe class" with "(λ swept; the same rig recovers rf@0.9=0.674 at
the post-nonlinearity tap, ruling out over-regularization)," or drop
"global optimum" and lean on the three-probe convergence already stated.

### A13: The falsifier runs on 3 of 13 checkpoints (selection unstated); the 0.5 threshold is descriptive
**Disposition:** PARTIAL (add the selection sentence and a *correct* robustness clause — the attacker's proposed clause is false)
**Response.** The valid core: the draft never states how the three teeth
checkpoints (A6-nh4-s0, S5-nh4-s0, S4-nh2-s2) were chosen, and the 0.5
disagreement cut and the loss<0.02 convergence cut are post-hoc
descriptive choices presented without a sensitivity note. One clause each
fixes it. **But I overturn the attacker's proposed remedy**: it claims the
13-cell count is "identical for any cut in [0.3, 0.7]" with "minimum
nonzero disagreement among the 13 = 0.65, maximum among the rest ≈0.1."
The raw says otherwise — the counts are **16 / 14 / 13 / 12 / 11** at cuts
0.3 / 0.4 / 0.5 / 0.6 / 0.7, the min gap among the ≥0.5 group is **0.55**
(not 0.65), and the max gap below is **0.4** (not 0.1). So the count is
*not* threshold-invariant and that sentence must not be pasted in. What
*is* robust and true: **every cell with a disagreement above 0.3 is a
converged cell (16/16 at cut 0.3; 13/13 at cut 0.5)** — the qualitative
signature "large disagreement ⟺ converged" is threshold-insensitive even
though the exact count is not.
**Supporting evidence.** This session on `crosscheck_lens_verdict_output.json`:
gap>0 distribution = {0.15(loss 0.074, NOT converged), 0.35, 0.35, 0.40,
0.55, 0.65, 0.80, 0.85, 0.90×2, 0.95×5, 1.00×2}; at cut 0.3 → 16 cells all
loss<0.02; at cut 0.5 → 13 cells all loss<0.02. §2.31a is where the
teeth-checkpoint selection was pre-registered.
**What goes in the paper if this defense is accepted.** Two clauses: (1)
"the three teeth checkpoints (pre-registered, §2.31a) span three groups
and include the two §5 exemplars"; (2) "the signature is threshold-robust:
every cell with a lens disagreement above 0.3 is a converged cell (16/16),
the same qualitative split the 0.5 cut reports." Do **not** claim a
constant count.

### A14: "Not 'no signal anywhere'" mis-describes the primary verdict, and the "different reason" is authorial gloss over an identical machine string
**Disposition:** CONCEDE + FIX (framing — two edits, both in §5 and one in the catalogue)
**Response.** Both halves confirmed from the raws. (a) Under the *primary*
lens S3 cleared its bar: far_recovered_frac_90 = 0.54 vs far_bar = 0.459,
far_clears = True. So the primary verdict was **not** "no signal
anywhere" — it read signal at S3 (and near-miss at A5); the FALSIFY
trigger was specifically the two decisive groups S5/A6. The draft's
"not 'no signal anywhere' but 'two specific failures'" mis-states the
primary as a blanket null. (b) The primary and crosscheck verdict `reason`
strings are **byte-identical** ("the contender does NOT measurably
separate from Arm 2 at EITHER S5 or A6 …"). So the "more informative
reason" is the authors' reading of the per-cell failure *geometry* (real
and well-evidenced: A6-nh2 0.00×5 both lenses vs S5's near-miss), not
anything the verdict machinery emitted. Both phrases need editing.
**Supporting evidence.** This session: `stage2_harvest_report.json`
(md5 7dddd19b…) primary S3 far 0.54 / bar 0.459 / clears True;
`stage2_harvest_report.json` `verdict.reason` == `crosscheck_lens_verdict
_output.json` `verdict_crosscheck_lens.reason` (string-equal True).
"no signal anywhere" appears twice: `sections/05_case_gauge.tex:66` and
the catalogue row `sections/09_appendix.tex:84`.
**What goes in the paper if this defense is accepted.** Replace "not 'no
signal anywhere' but 'two specific … failures'" with a faithful
description: "the corrected endpoint keeps the same FALSIFY verdict and
the same machine-emitted reason; what the fix changes is the *readable
geometry* — the two decisive groups (S5 near-miss, A6 non-convergent) fail
lens-independently, no longer masked by a primary that also zeroed the
succeeding groups." Fix the catalogue row's "with no signal anywhere"
likewise.

### A15: `tab:remetric`'s "separates" column is unintelligible without the (unstated) separation rule
**Disposition:** CONCEDE + FIX (presentation — one caption clause)
**Response.** Confirmed and faithful to the pre-registration, just
undocumented. The table shows S3 (far 0.900 vs baseline 0.150) →
"separates: no" while S4 (1.000 vs 0.000) → "yes," which reads as a
contradiction until you know the rule: separation requires the contender
to clear its bar **and** the baseline to collapse below 50% of its own
ceiling. For S3 the baseline far 0.15 exceeds its 0.125 collapse bar, so
no collapse, so no separation. The verdict raw carries exactly this
(`separates_from_arm2: False` for S3 with `arm2_collapses: False`). A
one-clause caption addition removes the apparent contradiction.
**Supporting evidence.** `verdict_crosscheck_lens.per_group.S3`
(`arm2_far 0.15, arm2_bar 0.125, arm2_collapses False, separates False`)
vs S4 (`arm2_collapses True, separates True`), this session.
**What goes in the paper if this defense is accepted.** Add to the
`tab:remetric` caption: "separation requires the contender to clear its
bar AND the Arm-2 baseline to collapse below 50% of its own ceiling; S3
clears its bar but its baseline does not collapse (0.15 > 0.125), hence
no separation."

### A16: Abstract frames all six as "a trained model appeared to fail" — incident 4 was synthetic; and the venue is still an assumption
**Disposition:** PARTIAL (concede (a); defend (b) as a correctly-flagged process item, not a paper defect)
**Response.** (a) Conceded: B1's failing "model" is a synthetic *perfect*
injection in an acceptance test — no trained model appeared to fail — so
the abstract's uniform "a trained model appeared to fail" and the intro's
"six apparent headline model failures" over-cover it by one. Cheap to fix.
(b) I defend this half: the venue's non-existence is not a defect *in the
paper* — it is an external dependency the paper's own Stage-0 process
already flags explicitly (venue-requirements.md: workshop list 404, "every
format field … is a working assumption," NeurIPS-2025 kit marked
UNVERIFIED). Flagging an unresolved assumption per the documented rule is
the process working, not a weakness a reviewer can act on. It resolves on
the Jul-11 list drop. The only real action item is the coupling with A7+A9:
the body is exactly 4.0pp with zero slack, so the citation block and the
catalogue-table move cannot both land without either a confirmed higher
limit or a cut.
**Supporting evidence.** `sections/00_abstract.tex:6-7` ("a trained model
appeared to fail"); `sections/09_appendix.tex:5` ("injected a synthetic
*perfect* model"); venue-requirements.md status block (404, working
assumptions).
**What goes in the paper if this defense is accepted.** (a) Abstract:
"a model — in one case a synthetic oracle — appeared to fail," and intro:
"six apparent failures at the model side of the pipeline." (b) No paper-
text change; on Jul-11 run the venue-requirements refresh, confirm the
page limit, and budget the A7 + A9 additions against it.

## 3. New citations found during defense

All required by A7; the attacker supplied arXiv IDs, which must be
re-verified against the arXiv export API before insertion (the refs.bib
header asserts every arXiv-bearing entry was so verified — the new ones
must meet the same bar). Load-bearing first:

- **Schaeffer, Miranda & Koyejo, "Are Emergent Abilities of LLMs a
  Mirage?," NeurIPS 2023, arXiv:2304.15004** — the highest-profile
  "apparent model property is a metric artifact" result; this paper's
  thesis inverted. The comparison a measurement-workshop reviewer expects
  named first. **Highest priority.**
- **Hewitt & Liang, "Designing and Interpreting Probes with Control
  Tasks," EMNLP 2019, arXiv:1909.03368** — control tasks / selectivity =
  Case II's shuffled-tap controls. Cite at the §4 control.
- **Elazar, Ravfogel, Jacovi & Goldberg, "Amnesic Probing," TACL 2021,
  arXiv:2006.00995** — causal removal vs decodability = Case II's
  state-zeroing localization. Cite at the §4 zeroing.
- **Adebayo et al., "Sanity Checks for Saliency Maps," NeurIPS 2018,
  arXiv:1810.03292** — instrument falsification via randomization = the
  direct ancestor of Rule 3's teeth.
- **Belinkov, "Probing Classifiers: Promises, Shortcomings, and
  Advances," Computational Linguistics 48(1) 2022, arXiv:2102.12452** —
  survey naming the probe-vs-behavior dissociation Case II reports.
- **Kapoor & Narayanan, "Leakage and the Reproducibility Crisis," Patterns
  2023, arXiv:2207.07048** — leakage taxonomy; supports the crosscheck's
  fit/eval-split leakage-guard (tiebreak ground 1), and shares a first
  author with the already-cited REFORMS, so it is nearly free to add.
- *(Optional)* **Northcutt, Athalye & Mueller, "Pervasive Label Errors,"
  NeurIPS 2021 D&B, arXiv:2103.14749** — broken measurement substrate
  flipping verdicts; benchmark-side sibling of the broken-lens catalogue.
- *(Optional)* **Voita & Titov, "Information-Theoretic Probing with MDL,"
  EMNLP 2020, arXiv:2003.12298** — preempts a "why rf@0.9 at all" review
  with one §4 clause.

## 4. Attack ordering note

The attacker's raw-artifact discipline is sound; almost every catch
reproduces. My severity disagreements:

- **A1 (CRITICAL) — I would call it SERIOUS on validity, but I do not
  strongly contest the CRITICAL label.** The fix is a one-character edit
  (30→25) and the qualitative claim is untouched, so no conclusion is at
  risk. What justifies the high rating is *thesis-specific*: this paper's
  entire credibility is "we re-verified every number against raws," and A1
  is a number the authors demonstrably did *not* re-verify (copied from an
  internally-inconsistent design-doc). A reviewer who finds it discounts
  the central methodological claim. So: trivial to fix, but rightly
  high-severity for *this* paper. The most valuable follow-through is the
  verify-vs-raws sweep over all count claims, which A2/A11/A14 show is
  where the draft's own verification was weakest.

- **A7 (SERIOUS) — under-weighted; this is the most acceptance-critical
  attack in the report, above A1.** A probing-pitfalls case study that
  cites none of the probing-methodology canon is the kind of omission a
  measurement/eval reviewer rejects on ("does not engage the field"),
  independent of whether the numbers are right. The numbers being right
  (which they are) does not save the paper from the missing-context
  rejection. Treat A7 as co-critical with A1.

- **A3 (SERIOUS) — the weakest of the SERIOUS tier, arguably MINOR.** It
  lands on a "briefly" incident (§6), the confounding number (0.046) is
  *already disclosed* in the appendix, and the fix is a one-clause
  attribution correction. Its weight comes only from the irony (a
  misattribution in the don't-misattribute incident), which is real but
  reputational, not substantive.

- **A13 — the attacker's own remedy is factually wrong.** Flagged in the
  A13 entry: do not add the "count identical for any cut in [0.3,0.7]"
  clause; the raw count is 16/14/13/12/11. The correct robustness claim is
  about the *pattern* (disagreement>0.3 ⟺ converged, 16/16), not the
  count. This is the one place the rebuttal must correct, not just relay,
  the attacker.

- **A16 split.** (a) is a fair MINOR concession; (b) is not a paper defect
  at all (a correctly-flagged external assumption) and should not count
  against the draft.

**Weaknesses the attacker missed (surfaced for a better paper):**

1. **The "two genuine model failures" are both inside Case III (A6 and
   S5).** Contribution 3 and the abstract present them as a program-level
   property of the discipline ("two genuine model failures survived every
   instrument fix"), but both come from the single Case III re-metric —
   the boundary evidence is narrower than the framing implies. Add a
   scoping clause: the anti-laundering demonstration rests on one
   incident's re-read, corroborated by Case I's two withheld findings.

2. **The "broken primary" narrative resolves to the *identical* endpoint
   verdict and byte-identical reason string (A14 extended).** The flagship
   case study concludes the primary lens was "broken on converged cells,"
   yet the corrected lens reaches the same FALSIFY with the same
   machine-emitted reason. A sharp reviewer will ask what the broken lens
   actually cost. The honest answer — it changes the *per-cell/per-group
   geometry interpretation*, not the endpoint — should be stated plainly
   rather than left for the reviewer to infer, or the word "broken" will
   read as overstated.

3. **The falsifiability headline generalizes from one falsifier
   instance.** "The discipline is itself falsifiable" (abstract/thesis) is
   demonstrated only for Rule 3's teeth on Case III (three checkpoints).
   The other four rules are not shown to be falsifiable. Either scope the
   claim ("the decisive adjudication carried a pre-registered falsifier")
   or note that the falsifiability property is demonstrated for the
   crosscheck rule specifically. Overlaps A13 but is a distinct framing
   point.
