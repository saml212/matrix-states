# Defense Report — Round 1 (Stage 02)

Paper: "The Rank the Task Demands: A Causal Rank Law for Matrix Memories
Trained on Group Composition" (NeurReps 2026 EA draft, `papers/neurreps-ea/`).

I re-derived every number the attack report cites directly from the raw
JSONs, `readout.py`, `group_word_encoder.py`, and
`CAPABILITY_SEPARATION_DESIGN.md`, independent of the attacker's own
computations. All of it checked out. I am not being generous here: this
is one of the stronger attack reports I could construct evidence
against, and most of it survives contact with the raw artifacts.

---

## 1. Summary for the rebuttal agent

**0 DEFEND, 3 PARTIAL (A2, A7, A8), 5 CONCEDE+FIX (A1, A3, A4, A5, A6).**

The paper is submittable after fixes, and every fix is a **framing fix** —
none requires new compute or new seeds before submission. That is the
single most important finding of this defense: A1 (the CRITICAL) is real
and re-verified independently (the max cosine of any rank-$(\dmin{-}1)$
matrix against the rank-$\dmin$, unit-singular-value target is exactly
$\sqrt{(\dmin{-}1)/\dmin}$, below 0.9 at all five groups — I re-derived
this from Von Neumann's trace inequality plus Cauchy–Schwarz, not just
from the design record's pre-computed numbers, and confirmed the model's
own `force_rank_k` path truncates a $d_{\mathrm{state}}\times
d_{\mathrm{state}}$ matrix to global rank $k$ before the readout
restricts it to the $\dmin$-dim subspace, so the bound applies exactly as
claimed) — but it is fixable by demoting necessity from "causal SGD
discovery" to "geometrically forced floor," and by moving the razor's
actual evidentiary weight onto two things the tautology does *not*
touch: (a) sufficiency at $k=\dmin$ (not geometrically guaranteed, the
real causal claim), and (b) how close the below-$\dmin$ cells land to
their own forced ceiling (76–95%, mean 88%, a genuine and previously
unstated finding about how well SGD trains under the cap). A3, A4, A5,
and part of A6 are documented, computed, and sitting in the design record
already — restoring compressed disclosures, not new experiments. A2, A7,
A8 need wording, not retraction.

**Most important fixes, in priority order:**
1. Rewrite the necessity leg (abstract, intro, §5, fig captions) — A1.
   Submission blocker if left as-is; not compute-blocked.
2. Add the S3 per-seed 2/4 + bar-sensitivity disclosure (one footnote or
   appendix line) — A3.
3. Reword "held-out depths to 21" to disclose the $h \bmod K = 5$
   equivalence and reframe as a composition-stability probe — A4.
4. Add a gate1a per-cell table for S3/S5 to the appendix and soften
   "anchor-relative and unaffected" — A5.
5. Fix the Barrington citation precision and add 1–2 short Related Work
   sentences (Chughtai et al. is the one the attacker missed and is the
   closest prior work) — A6.
6. Soften "returns to unconstrained-anchor levels" and disclose the S4/S5
   anchor-inversion numbers with a candidate mechanism — A2.
7. One clause on why 0.8, not 0.9 or the observed value, is the M1 bar
   — A8.
8. Real anonymized code link (or explicit release commitment) — A7.

None of these touch the paper's two claims that are actually clean: the
$K$-pair binding foundation (Fact 1 + the $d{=}8,K{=}4$ causal step) and
the marquee M1 observational/TOST result, both of which I independently
re-verified against the raw JSONs and found exactly as reported.

---

## 2. Defenses

### A1: The razor's necessity leg (N1, N2) is a mathematical tautology of the 0.9 threshold, not a causal SGD finding

**Disposition:** CONCEDE + FIX (framing fix, submission blocker if left
unaddressed, but does not require new evidence)

**Response.** The attack is correct, and I verified it two ways
independent of the attacker's own derivation. First, analytically: for a
target $T$ with $\dmin$ tied unit singular values, Von Neumann's trace
inequality gives $\langle X, T\rangle_F \le \sum_i \sigma_i(X)\sigma_i(T)$
for any $X$, and maximizing this over rank-$\le k$ $X$ at fixed
$\|X\|_F$ (by Cauchy–Schwarz, spreading $X$'s $k$ singular values evenly
across $T$'s $k$ largest, all equal to 1) gives $\max \cos(X,T) =
\sqrt{k/\dmin}$ — this is a supremum over *all* rank-$\le k$ matrices, not
just a constructed example, so no rank-$(\dmin{-}1)$ matrix, however
trained, can exceed it. Second, computationally: I confirmed
`force_rank_k` truncates the model's full $d_{\mathrm{state}} \times
d_{\mathrm{state}}$ output to global rank $k$
(`group_word_encoder.py:119-122`, `truncate_to_rank`) *before*
`readout.py`'s subspace restriction $A(w) = U^\top Z(w) U$, so the
restricted operator scored against $\rho_{\mathrm{eval}}$ is itself
rank-$\le k$, and the bound transfers exactly. I computed the ceiling for
all five groups myself: $0.7071/0.8165/0.8165/0.8660/0.8944$ for
$\dmin=2,3,3,4,5$ — every one below the paper's 0.9 cutoff, matching the
attacker's numbers to four decimals. And the design record itself
pre-computed this before the wave launched: `CAPABILITY_SEPARATION_DESIGN.md:5869-5871`,
"Oracle ceilings for the harvest: k≥d_min exact (1.0), k=d_min−1 bounded
≤0.894," used explicitly only to distinguish the repaired wave's
signature from the old ambient-identity-tax defect's $\sqrt{k/d_{\mathrm{state}}}$
signature — never flagged as a caveat on what the necessity zero can
prove. This is exactly what it looks like: a broken positive control that
the team's own records show they computed and then didn't connect to the
headline claim.

I do not think this is fatal to the paper, and I do not think it needs a
retraction. The paper's own binding-task foundation (§3, Fact 1) already
treats necessity as *provable*, not a discovered SGD behavior — the
causal contribution there is that SGD *reaches* the provable bound, not
that the bound exists. The group razor can be recast on exactly the same
footing: necessity below $\dmin$ is representation-theoretically forced
given the 0.9 threshold (provable, like Fact 1), and the empirical
content moves to two places the tautology does not touch: (i) the
below-$\dmin$ cells' *continuous* cosines land at 76–95% of their own
forced ceiling (I computed this precisely: S3 86.3%, S4 91.2%, A5 94.9%,
S5 75.7%, A6 93.5%, mean 88.3% — an unstated finding that models trained
at $k=\dmin{-}1$ get most but not all of the way to the achievable
optimum, with S5 a real outlier worth flagging, tying into A5), verifying
models trained near the rank-constrained ceiling rather than collapsing;
and (ii) sufficiency at $k=\dmin$ is *not* geometrically forced — a priori
nothing guarantees SGD finds a rank-exactly-$\dmin$ solution when capped
there, and the fact that it does, clearing the anchor-relative bar in
all five groups, is the actual causal finding. The M1 result (SGD
recruiting $\approx\dmin$ *without* a cap, i.e. not over-recruiting
despite two spare dimensions) is untouched by this attack and reinforces
the same story from the unconstrained side.

**Supporting evidence.**
- `matrix-thinking/capability_separation/group_word_encoder.py:119-122`
  (`force_rank_k` truncates global rank before subspace restriction).
- `matrix-thinking/capability_separation/readout.py:128-164`
  (`degauge_and_score` scores the restricted, rank-capped operator).
- Independent recomputation of ceilings: $\sqrt{1/2}=0.7071$,
  $\sqrt{2/3}=0.8165$ (×2), $\sqrt{3/4}=0.8660$, $\sqrt{4/5}=0.8944$.
- `experiment-runs/2026-07-09_m3fix_harvest/zero_pad__*__k_dmin_minus_1__seed0.json`,
  `crosscheck_mean_cos` re-pulled directly: 0.6102/0.7446/0.7752/0.6553/0.8364
  — I computed % of ceiling myself (86.3/91.2/94.9/75.7/93.5%, mean
  88.3%), a number not currently in the paper.
- `CAPABILITY_SEPARATION_DESIGN.md:5869-5871` (the pre-registered oracle
  ceiling, computed before launch, used only as a bug-distinguishing
  check).
- Von Neumann's trace inequality (classical; e.g. via Eckart–Young,
  Psychometrika 1936, the attacker's own cited source) as the general
  form of the bound the design record's unit test special-cases.

**What goes in the paper if this defense is accepted.**
- Abstract: replace "one rank below, exact recovery is 0.000 everywhere
  ... while at $\dmin$ recovery returns to unconstrained-anchor levels"
  with something like: "a pre-registered force-rank razor is exact in
  both directions: one rank below $\dmin$, recovery is capped at the
  geometrically forced ceiling and never crosses the 0.9 threshold (a
  consequence of the threshold and the target's tied spectrum, not an
  SGD collapse — cells land at 76–95% of their own ceiling); at $\dmin$
  — not geometrically guaranteed — recovery clears the anchor-relative
  bar in all five groups."
- §5 (`sections/05_causal_razor.tex`): replace "Necessity is exact and
  noiseless: $k=\dmin{-}1$ yields $\recninety=0.000$ ... show the zero is
  a threshold phenomenon, not a training collapse" with an explicit
  statement of the analytic bound (one sentence with the derivation
  sketch and the five ceiling numbers), followed by the 76–95%-of-ceiling
  finding, then pivot the "the pre-registered causal criterion is met"
  sentence to lean on sufficiency, not necessity.
- Figure 1/2 captions: "exactly 0.000 one rank below $\dmin$ everywhere"
  should note this is the analytically forced floor, not an
  emergent zero; keep "anchor-class at $\dmin$" as the causal read.
- Intro: soften "$\dmin{-}1$ is fatal, $\dmin$ suffices" to something
  like "$\dmin{-}1$ is analytically incapable of exact recovery under
  this threshold; $\dmin$ suffices, which is not guaranteed a priori."

---

### A2: n=1-per-cell statistics + anchor inversion undermine "returns to unconstrained-anchor levels"

**Disposition:** PARTIAL

**Response.** The numbers are correct — I re-pulled them directly:
S4 anchor 0.65 vs. $k{=}\dmin$ 0.80 (+0.15); S5 anchor 0.50 vs.
$k{=}\dmin$ 0.60 (+0.10) — and I agree the framing "recovery returns to
unconstrained-anchor levels" oversells the anchor as a ceiling when it is
demonstrably not one in 2 of 5 groups, on exactly the two marquee cells.
But I don't think this is a CONCEDE+FIX on the paper's substance: the
sufficiency verdict is measured against $0.9\times$anchor, not "beats or
matches anchor," and clearing that bar is the pre-registered criterion —
the inversion doesn't change which side of the bar any cell lands on. It
is a real statistical-fragility point about *how decisive* the margins
look (S4's +0.215, S5's +0.15 over the bar), given n=1.

I looked for a mechanism, since the attack explicitly invites one rather
than a bare assertion. There is a plausible one already latent in the
paper's own design: $d_{\mathrm{state}} = \dmin + 2$ for every arm, and
the fixed-wave target is zero-padded (`target_padding="zero"`), so the
*unconstrained* anchor must learn to drive two full ambient dimensions to
zero on top of learning the $\dmin$-dim signal, while the $k=\dmin$ arm
is structurally incapable of putting any energy in those directions at
all — the cap acts as an inductive bias that removes work the
unconstrained arm still has to do, under the *same* fixed step budget
(steps are pinned per group, not per arm — confirmed against
`CAPABILITY_SEPARATION_DESIGN.md:5969-5978`). This is consistent with
(and probably the same underlying cause as) A5's soft-convergence finding
for S4/S5's neighboring groups. I flag this as a hypothesis, not a
proven mechanism — it would need the anchor's own training curve for
the two ambient directions to confirm, which is a real experiment, not a
framing fix, and I would not block submission on running it.

**Supporting evidence.** `experiment-runs/2026-07-09_m3fix_harvest/zero_pad__S4__unconstrained__seed0.json`
(0.65) vs. `zero_pad__S4__k_dmin__seed0.json` (0.80);
`zero_pad__S5__unconstrained__seed0.json` (0.50) vs.
`zero_pad__S5__k_dmin__seed0.json` (0.60) — both re-pulled directly,
matching Table 2 exactly. `sections/02_setup.tex`: "$d_{\mathrm{state}} =
\dmin+2$, leaving spare dimensions so over-recruitment is expressible"
and "the target embeds the reference block-diagonally
($\rho_G(\cdot)\oplus 0$ in the repaired wave)" — confirms the zero-pad
mechanism candidate.

**What goes in the paper if this defense is accepted.** Reword "recovery
returns to unconstrained-anchor levels" to precise bar language
("clears the pre-registered $0.9\times$anchor bar") everywhere it
appears (abstract, §5, fig captions). Add one Limitations clause
disclosing the S4/S5 anchor-inversion numbers and the ambient-dimension
hypothesis, explicitly flagged as unconfirmed. Optionally (camera-ready,
not a blocker): 2–3 more seeds on the S4/A5/S5 marquee cells to make the
margins n>1.

---

### A3: S3's "confirmation" (N3) is fragile to a bar-choice the paper doesn't disclose

**Disposition:** CONCEDE + FIX (framing fix — the numbers already exist)

**Response.** Both facts check out exactly, straight from
`CAPABILITY_SEPARATION_DESIGN.md:6080-6183` (§1.36a), which I read in
full rather than trusting the attacker's excerpt. Per-seed:
$k=\dmin$ clears its own seed's $0.9\times$anchor bar in seeds 1 (+0.010)
and 3 (+0.110) only; misses in seed 0 (−0.045, the original marginality
trigger) and seed 2 (−0.120) — 2/4, exactly as attacked. And the
self-referential bar ($0.9\times$ the seed-mean anchor $= 0.9\times
0.6375 = 0.574$) does flip the verdict against the seed-mean $k=\dmin$
value of 0.5625 (−0.011, FAIL). The important thing I want to be honest
about: this is not a hidden fact the team is trying to bury — §1.36a's
own text discloses *both* numbers in essentially the same words the
attack uses ("Disclosed secondary read (does not override the primary
verdict above)... clears the OWN seed's bar in only 2/4 seeds... would
put the seed-mean k=d_min... marginally BELOW that self-referential bar
... this is exactly why the pre-registered comparison uses the fixed
§1.36 literal, not a self-referential recompute, and is disclosed here
rather than silently chosen post hoc"). The design record did the right
thing; the paper's 4pp compression is what dropped it. That is a real
gap between what the lab knows and what a NeurReps reviewer would see,
and it's squarely in the gauntlet's job to catch a compression that loses
a caveat this material.

**Supporting evidence.** `CAPABILITY_SEPARATION_DESIGN.md:6122-6132`
(the seed-mean table, all four rows independently re-derivable),
`CAPABILITY_SEPARATION_DESIGN.md:6172-6183` (§1.36a's own disclosed
secondary read, matching the attack's numbers exactly, including the
"driven by anchor noise... consistent with §1.33's own flag of S3 as the
noisiest group" framing).

**What goes in the paper if this defense is accepted.** One footnote or
appendix sentence attached to the current $^{\ast}$ marker in Table 2 /
§5: "Per-seed, $k=\dmin$ clears its own seed's bar in 2/4 seeds
(driven by anchor noise, S3's own anchor ranging 0.55–0.80); the
self-referential bar ($0.9\times$ seed-mean anchor) narrowly fails
(0.563 vs.\ 0.574) — the fixed pre-registered literal is used to avoid
laundering that noise into the threshold itself." This fits in the
budget the appendix already has (per `brief.md`'s page table, Appendix
A/B have slack) and requires no new computation.

---

### A4: N9's "held-out depths to 21" claim is confounded by cycle periodicity — depth 21 is mathematically identical to depth 5

**Disposition:** CONCEDE + FIX (framing fix)

**Response.** I pulled `M3_held_out` from all five seed JSONs myself:
every seed's `"21"` entry carries `"effective_hop": 5`, and $21 \bmod 8 =
5$ exactly, confirming the attack's core fact. But I want to be precise
about what this does and doesn't undermine, because the source draft
(`matrix-thinking/submissions/neurips-ws-2026/sections/04_task_e.tex`)
already gets most of this right and the EA draft's one-line compression
is what loses it. The base draft's own footnote explains *why* a single
Hamiltonian $K$-cycle was used (to avoid the *worse* problem — general
permutations decomposing into short disjoint cycles that collapse
held-out hops into trivial/identity queries, "up to 100% collapse
measured at small $K$") — but that footnote does not itself state the
$h \bmod K$ equivalence for depth 21, and the base draft's own headline
framing is explicitly *not* "generalizes to a deeper target": its
§\ref{sec:taske-headline}/§\ref{sec:taske-depth} argue depth 21 is a
*spectral-exactness / composition-stability* probe ("iteration depth is
a sharper spectral-exactness probe than any single-hop cosine
tolerance"), motivated by the fact that a rank-starved ($k{=}7{=}K{-}1$)
operator is fine through $h{=}7$ (rec@0.9 $=0.881$) but collapses by
$h{=}21$ (rec@0.9 $=0.060$) — i.e. the base draft's own argument for why
depth 21 matters is explicitly about numerical/spectral robustness under
repeated self-application, not about reaching a new group-theoretic
target. That framing survives this attack intact. What does not survive
is the EA's compressed one-liner in §3 (`03_binding.tex`): "the same
operator composes exactly ($Z^h$, held-out depths to 21) in four of five
seeds" — stripped of all the base draft's careful qualification, this
reads to a NeurReps reviewer as "generalizes to a novel deep target,"
which it is not: seed 0's own numbers show $h{=}5$ recovers only
partially (rec@0.9 $=0.303$) while $h{=}21$ — the *same* group-theoretic
target — reads $0.0001$, a genuine and large gap the compressed sentence
doesn't explain and the un-compressed base draft does (repeated
self-application amplifies the same underlying imperfection).

**Supporting evidence.**
`experiment-runs/2026-07-02_task_e_40k/task_e_40k/t1_matrix_permutation_K8_frN_s{0..4}.json`,
`M3_held_out["21"].effective_hop = 5` in all 5 seeds, independently
re-pulled. Seed 0: $h{=}5$ mean_cos $0.856$/rec@0.9 $0.303$ vs.\ $h{=}21$
mean_cos $0.486$/rec@0.9 $0.0001$ — confirms the ~2500× gap the attack
cites is real and is the "still-transitioning" seed the base draft
already flags. Seeds 1–4: $h{=}5$ and $h{=}21$ both read rec@0.9
$\approx 1.0$ (e.g. seed 1: $1.000$/$1.000$) — for the *converged*
seeds, the claim "exact through 21 self-applications" is true and
non-trivial (it is a stronger statement about the trained operator's
numerical exactness than $h{=}5$ alone, precisely because 21
self-applications of an imperfect operator would ordinarily amplify
error, and here it doesn't). `matrix-thinking/submissions/neurips-ws-2026/sections/04_task_e.tex:14-20`
(the periodicity-motivation footnote) and `:99-107` (the
spectral-exactness-probe framing, already correct).

**What goes in the paper if this defense is accepted.** In
`sections/03_binding.tex`, replace "held-out depths to 21" with
something like: "the operator composes exactly through 21 sequential
self-applications in four of five seeds (nominal depth 21 is
periodicity-equivalent to depth 5 under the 8-cycle target, so this
tests numerical exactness under repeated composition, not a distinct
group-theoretic target)." This is a one-clause fix, not a rewrite, and
it can cite `liu2023shortcuts` (see A6) in the same breath as the
natural citation for why this distinction matters.

---

### A5: The soft-convergence disclosure in Limitations understates how much it affects the two groups it names

**Disposition:** CONCEDE + FIX (framing fix — restore already-computed numbers)

**Response.** I pulled gate1a directly from `CAPABILITY_SEPARATION_DESIGN.md:6060-6066`
(not just the harvest JSONs) and the attacker's numbers are exactly what
the design record itself reports: "gate-1(a) (min L∈[2..5] cos ≥ 0.92):
all 10 k=d_min−1 cells fail it... S3's four variant-A cells (anchor
min_val 0.9143) and S5's anchor/k_dmin/k_dmin+1 (0.876-0.879) sit just
under the bar." I'll note something the attacker didn't: the design
record's own phrase "sit just under the bar" is itself a bit generous
for S5 — 0.876–0.879 against a 0.92 bar is a 4.1–4.4-point shortfall,
not "just under" in the way S3's 0.9143 (a 0.6-point shortfall) is. So if
anything the source disclosure slightly undersells the S5 gap too, and
the EA's further-compressed "two groups... soft convergence (disclosed)"
compresses it again.

That said, I don't think "the razor reading is anchor-relative and
unaffected" is false — it's under-qualified. The razor's *decisive
necessity* leg is unaffected because (per A1) it reads 0.000 regardless
of convergence quality; that argument holds regardless of gate1a. The
*sufficiency* leg is where convergence quality could matter, since it's
compared against the (possibly under-converged) anchor — which is
exactly A2's anchor-inversion concern for these same two groups. I
recommend reading A2 and A5 together: once A1's fix demotes "returns to
ceiling" language, and A2's fix discloses the anchor-inversion numbers,
most of A5's remaining bite is the missing gate1a table itself, which is
cheap to add.

**Supporting evidence.** `CAPABILITY_SEPARATION_DESIGN.md:6060-6066`
(verbatim gate1a disclosure, matching the attack's numbers to three
decimals). `CAPABILITY_SEPARATION_DESIGN.md:6172-6178` (§1.36a's
separate gate1a table for the S3 extension: "unconstrained clears at
seed 2 only (0.9325); k=d_min clears at seed 3 only (0.9217)" —
confirms the mixed-across-seeds picture the attack describes).

**What goes in the paper if this defense is accepted.** Add a small
appendix table (Appendix A or B, both have declared slack per
`brief.md`'s page-budget table) with per-cell gate1a `min_val`/pass-fail
for S3's 4 cells and S5's 4 cells. Reword the Limitations sentence from
"two groups at the shortest pinned budget show soft convergence
(disclosed; the razor reading is anchor-relative and unaffected)" to
something scoped like "...soft convergence (disclosed, appendix table);
the necessity leg is unaffected by construction (App.~\ref{...}), the
sufficiency leg for these two groups should be read as
directionally-consistent-under-disclosed-soft-convergence rather than
folded into the unqualified 5/5 count." No new experiments required to
state this; re-running S5 at a longer budget is a real camera-ready
strengthening, not a blocker.

---

### A6: Missing citations — direct competitors on the exact mechanism this paper studies

**Disposition:** CONCEDE + FIX (mixed: one MUST-CITE precision fix is
essentially free; the rest is SHOULD-CITE, budget-permitting)

**Response.** I checked all three of the attacker's candidates plus one
they missed, weighing this against `brief.md`'s stated ~0.30-page
Related Work budget (already engaging 5 papers) and the 4pp hard limit.

- **Barrington & Thérien (JACM 1988) — MUST-CITE, effectively free.**
  The current single citation (`barrington1989`) is Barrington's own
  1989 result establishing $\mathrm{NC}^1 = $ width-5 branching programs
  via $S_5$ specifically. The paper's actual claim — that non-solvable
  groups' word problems are $\mathrm{NC}^1$-complete *in general*,
  invoked across all three non-solvable groups ($A_5, S_5, A_6$), not
  just $S_5$ — is the Barrington–Thérien classification theorem, not
  Barrington 1989 alone. This is a precision fix to an existing
  citation (add one `\citep{}` key), not new prose — free.
- **Chughtai, Chan & Nanda, "A Toy Model of Universality: Reverse
  Engineering How Networks Learn Group Operations," ICML 2023,
  arXiv:2302.03025 — MUST-CITE, the attacker missed this one.** This is
  the closest prior work to the paper's actual thesis: it studies how
  small networks learn group multiplication via representation theory
  (irreps), which is exactly this paper's framing (minimal faithful
  representation dimension as the geometric currency). A NeurReps
  reviewer versed in this literature would flag its absence before any
  of the attacker's three. One sentence.
- **Liu et al., "Transformers Learn Shortcuts to Automata," ICLR 2023,
  arXiv:2210.10749 — SHOULD-CITE, but effectively free because it
  belongs in A4's fix (§3), not Related Work (§6).** It's the natural
  citation for why the depth-21 reframing matters (shortcut circuits vs.
  genuine sequential composition) — bundling it into A4's one-clause fix
  costs nothing extra against §6's budget.
- **Delétang et al., "Neural Networks and the Chomsky Hierarchy," ICLR
  2023, arXiv:2207.02098 — SHOULD-CITE, budget-permitting.** Its Cycle
  Navigation task is the closest existing benchmark precedent
  (cyclic-group word problem, length generalization). Genuinely useful
  but cuttable if the other three don't leave room for a fourth
  sentence in 0.30 pages.

**Supporting evidence.** All four papers verified as real with correct
venues/arXiv IDs against my training knowledge (I did not have live web
access in this pass; recommend a final citation-existence check before
submission, standard practice). None of the four currently appear in
`refs.bib` (grepped directly — confirmed absent).

**What goes in the paper if this defense is accepted.** `refs.bib`: add
four entries (`barringtontherien1988`, `chughtai2023universality`,
`liu2023shortcuts`, `deletang2023chomsky`). `02_setup.tex`: change
`\citep{barrington1989}` to `\citep{barrington1989,barringtontherien1988}`.
`06_related.tex`: one sentence distinguishing Chughtai et al. by name
(MUST), one more distinguishing Delétang et al. if space allows
(SHOULD, cuttable). `03_binding.tex`: fold in Liu et al. as part of A4's
fix (MUST, but zero marginal Related-Work cost).

---

### A7: No code/data release for review undercuts verifiability of exactly the pipeline this report found an issue in

**Disposition:** PARTIAL

**Response.** The factual premise is correct — I confirmed the anon
build's code link is the literal placeholder
`https://anonymous.4open.science/` (`brief.md`, Anonymization surface
section), and `VENUE_REQUIREMENTS.md:24` does say "manuscript, data and
code anonymized during review," which is instructions for *how* to
anonymize code if it accompanies a submission, not a mandate that code
must be included — EA tracks at this and comparable workshops routinely
accept without released code, and I don't think this "undercuts the
paper" as strongly as the attack frames it, since A1 was found by
reading the *design record*, not the training code, and would be equally
findable by a reviewer with only the paper's math in hand once the fix
above lands. So I'm not defending the placeholder as ideal, but I am
pushing back on "undercuts verifiability" as an indictment — the fixed
paper's necessity claim will be a closed-form derivation any reviewer
can check without any code at all.

**Supporting evidence.** `papers/neurreps-ea/brief.md` Anonymization
surface section (placeholder link, verified verbatim);
`papers/neurreps-ea/VENUE_REQUIREMENTS.md:24` (anonymization instruction,
not a code-mandate).

**What goes in the paper if this defense is accepted.** Swap the
placeholder for a real anonymized repo snapshot (readout/degauging code
only, no training infra needed) before submission if time allows; if
not, this is a legitimate camera-ready deferral, not a submission
blocker — MINOR severity is right.

---

### A8: The reported exact-null p-value uses a looser threshold than the observed statistic

**Disposition:** PARTIAL

**Response.** I independently enumerated all $5! = 120$ assignments of
$\dmin$ values to a fixed rank ordering: $8/120 = 6.67\%$ achieve
$\rho \ge 0.8$ (matches the paper exactly) and $2/120 = 1.67\%$ achieve
$\rho \ge 0.9747$ (the true observed max), confirming the attack's
recomputation to the decimal. But I want to push back on the implication
that 0.8 is an arbitrary or unexplained choice: `CAPABILITY_SEPARATION_DESIGN.md:2732`
records that $\rho \ge 0.9$ was *evaluated and explicitly declined*
before the wave ran, specifically because the achievable $\rho$ values
under this design's tie structure are discrete and cliff-shaped —
"the next-highest achievable ρ after the perfect-ordering maximum is
0.8721, an ~0.10 cliff, so a 0.9 bar would fail on any single
ordinary-noise misordering, undermining M1's corroborating role." 0.8
sits deliberately below that cliff, as a genuinely reasoned
pre-registered choice, not a post-hoc loosening. That reasoning is real
and good — it just isn't in the paper, which is a fair and cheap thing
to fix.

**Supporting evidence.** Independent enumeration (Python, exact,
$5!=120$): $8/120=6.667\%$, $2/120=1.667\%$, matching both the paper and
the attack. `CAPABILITY_SEPARATION_DESIGN.md:2732`: "ρ≥0.9 evaluated and
EXPLICITLY DECLINED... achievable but brittle... a 0.9 bar would fail on
any single ordinary-noise misordering, undermining M1's corroborating
role"; `CAPABILITY_SEPARATION_DESIGN.md:529-534` (the four discrete
achievable $\rho$ levels: 0.9747, 0.8721, 0.8208, 0.7182 — confirms the
cliff structure is real, not asserted).

**What goes in the paper if this defense is accepted.** One clause after
the $P(\rho\ge0.8)$ sentence in `04_ranklaw_observed.tex`: "(0.8, not the
observed value, is the pre-registered bar, chosen below the next
achievable level at 0.87 to avoid brittleness to a single misordering
given the design's discrete achievable-$\rho$ structure)."

---

## 3. New citations found during defense

- **Barrington, D.A.M. & Thérien, D., "Finite Monoids and the Fine
  Structure of $\mathrm{NC}^1$," JACM 35(4), 1988.** Precision fix to
  the existing `barrington1989` citation for the general
  solvable/non-solvable word-problem dichotomy (A6).
- **Chughtai, B., Chan, L. & Nanda, N., "A Toy Model of Universality:
  Reverse Engineering How Networks Learn Group Operations," ICML 2023,
  arXiv:2302.03025.** The closest prior work to this paper's own thesis
  (representation theory of learned group operations); the attacker
  missed this one and it is arguably more central than any of their
  three candidates (A6).
- **Liu, B., Ash, J., Goel, S., Krishnamurthy, A. & Zhang, C.,
  "Transformers Learn Shortcuts to Automata," ICLR 2023,
  arXiv:2210.10749.** Natural citation for the A4 fix (shortcut circuits
  vs. genuine composition) — attacker's find, confirmed real.
- **Delétang, G. et al., "Neural Networks and the Chomsky Hierarchy,"
  ICLR 2023, arXiv:2207.02098.** Closest existing benchmark precedent
  (Cycle Navigation); attacker's find, confirmed real, SHOULD-CITE if
  space allows.

I did not have live web access during this defense pass; all four are
verified against training knowledge (titles, authors, venues, arXiv IDs
all consistent and specific enough that I'd be surprised if any were
fabricated) — recommend one final existence/ID check before the actual
submission, standard practice regardless of this gauntlet.

---

## 4. Attack ordering note

The attacker's severities are basically right, with two notes:

- **A1 is correctly CRITICAL** and I'd resist any temptation to downgrade
  it — it's the paper's headline sentence in the abstract, and the
  tautology is exact, not approximate (the ceiling is strictly below 0.9
  at all five groups, no near-misses). It is fixable without new
  compute, which softens its practical severity but not its rating.
- **A5 is correctly SERIOUS but is substantially entangled with A1 and
  A2, not independent of them.** Once A1's fix demotes "returns to
  ceiling" language and A2's fix discloses the anchor-inversion numbers,
  most of what A5 is protecting against (an over-strong "unaffected"
  claim) is already absorbed by those two fixes; what's left of A5 on
  its own is "the appendix should have a gate1a table," which is a small
  ask. I would not escalate A5 to CRITICAL, but I'd flag it as the one
  SERIOUS attack whose fix is mostly free-riding on A1+A2's fixes rather
  than needing independent text.
- **A6 undersells itself slightly** by not finding Chughtai et al. 2023,
  which is a stronger miss than any of the three papers the attacker did
  find, given the paper's own framing (representation theory of what
  gradient descent recruits) is closer to Chughtai's territory than to
  the automata/benchmark papers the attacker cites. Same severity rating
  (SERIOUS/missing-citation) is fine; I'd just make sure the fix
  prioritizes Chughtai first if space is the binding constraint.

**One thing the attacker didn't raise that I want to flag explicitly, not
as a numbered attack but as a process risk:** the Related Work section
self-cites `\citep{larson2026gradient}`, and that bib entry's `author`
field is `{Larson, Sam}` — the real author name, verbatim
(`refs.bib:2`). `brief.md`'s own anonymization surface section commits
to zero matches of the token `larson` anywhere in `main-anon.pdf`'s
source closure. No `main-anon.tex` exists yet in this tree (only
`main.tex`, the single-blind placeholder-author build; the `Makefile`'s
`anon` target references a file that hasn't been created), so this is
not a live defect in the current build — but it is a landmine for
whoever builds the double-blind version next: a rendered bibliography
entry reading "Larson, Sam. The Gradient Does Not See Rank..." would
deanonymize the paper via self-citation, and the anonymization grep gate
brief.md describes would need to specifically catch this (self-citations
typically need a separate "Anonymous, Under Review" bib entry for the
double-blind build, not just a grep-and-hope). Worth a line in whoever's
task list builds `main-anon.tex`.
