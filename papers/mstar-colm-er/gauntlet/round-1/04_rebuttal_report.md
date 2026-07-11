# Rebuttal report — round 1 — mstar-colm-er

## 0. Top-line: is any CRITICAL still open?

**No CRITICAL remains open, conditional on FIX-1 being applied in full.**
Both A1 and A2 are adjudicated **DEFENSE INSUFFICIENT as originally
written** (the defense did not disprove either attack — it correctly
conceded both on the facts), but a single fix (FIX-1, below) rescopes the
one paragraph both attacks target and, once applied, drops both below
CRITICAL to a disclosed-limitation level. FIX-1 has two file edits
(`sections/02_setup.tex` and `sections/03_recall.tex`) plus a new
appendix figure; applying only one half re-opens the other attack, so
treat it as a single atomic change. I independently re-verified the
factual premises of every attack against the raw JSONs and the design
registry (not just re-reading the attack/defense reports) and found no
place where either report mis-stated a number; my adjudications below
change severities and fix *content* in a few places, not underlying
facts.

Two things I found independently, not flagged by either the attacker or
the defender, that materially change the shape of the fixes:

1. **The defense's proposed A2 fix would cite exact numbers
   (`recovered_frac@0.9 = 1.000000` / `0.000000`) from the design
   registry's K-simultaneous-bindings diagnostic, but no archived,
   md5-asserted JSON artifact backs those numbers under
   `experiment-runs/`** — I grepped for it and found none; the result
   exists only as a table embedded in the internal markdown design doc,
   computed in a "design-review scratchpad" that was never archived as a
   pipeline source. Quoting those specific decimals in the paper would
   violate the registered honesty pin *"every number keeps its
   evidence-row comment"* (there is no evidence row for them) and would
   silently create a second, weaker evidentiary standard alongside
   C1–C10. FIX-1 below resolves this by using the underlying argument as
   **self-contained mathematical reasoning from the ablation's own stated
   update equation** (no citation, no new evidence row needed, because it
   is a structural fact about $s_t \odot q_t$, not a measured number) —
   this is a strictly stronger fix than quoting the registry's numbers,
   and it is genuinely free.
2. **Neither defense-proposed fix for A2 or A6 may cite
   `HEAD_TO_HEAD_DEMO_DESIGN.md` by name or section number in the paper
   text.** It is an internal, non-public project document; citing it in
   a double-blind workshop submission is both unciteable (reviewers have
   no access) and a deanonymization risk (it names internal repo/tooling
   structure). Both fixes below are written as self-contained prose
   instead.

**Disposition counts:** 4 DEFENSE INSUFFICIENT (A1, A2, A3, A4) — all
resolved by a fix below; 4 DEFENSE VALID BUT EDIT (A5, A6, A7, A8) — all
resolved by a one-paragraph disclosure edit. 0 DEFENSE VALID (clean
disproof) on any attack; 0 PARTIAL survives after the fix list (see
residual risk, §4).

**The three fixes that carry most of the weight:** FIX-1 (CRITICAL,
resolves A1+A2 — rescopes the paper's one interpretive/mechanistic claim
and adds a training-curve figure, zero new compute), FIX-2 (SERIOUS,
resolves A3 — a two-line `zorder`/`alpha` change to the money figure),
FIX-3 (SERIOUS, resolves A4 — a statistical-framing correction using
numbers already in the draft). FIX-4/5/6 are one-paragraph disclosure
edits.

---

## 1. Ordered fix list

### FIX-1: Rescope the recall section's "licensed reading," disclose the additive control's capacity/write-conditioning confound, and add a training-loss-trajectory figure

**Severity:** CRITICAL
**Resolves:** A1, A2
**File(s):** `sections/02_setup.tex`, `sections/03_recall.tex`,
`sections/10_appendix.tex`, `figures/figure-gen.py`, `refs.bib`
(auto-regenerates `figures/tables_generated.tex` and produces
`figures/fig3_traincurve.pdf` — do not hand-edit the generated table
file)

This is one atomic fix in five parts. Apply all five; applying only the
`02_setup.tex` edit re-opens A1's overclaim, and applying only the
`03_recall.tex` edit re-opens A2's undisclosed confound.

#### 1a. `sections/02_setup.tex` — rename and disclose the additive control

**Location:** the "Flat-vector recurrence" paragraph in "Three arms,
matched budget."

**Before:**
```
\textbf{Flat-vector recurrence (the structure control).} Identical
embedding table, output head, and feed-forward blocks; only the mixer
changes, to a gated elementwise vector recurrence
$s_t = s_{t-1} \odot g_t + v_t$ with read $o_t = s_t \odot q_t$, at matched
total parameter count via mixer width and $d_{\mathrm{state}} = 64$ pinned
equal to the contender's. This arm removes the multiplicative
outer-product update while preserving everything else, rather than
reshaping the same operator, since any $d^2$-dimensional vector can encode
a $d \times d$ matrix and structure matters only if the operations use it.
Its state is 512 bytes, reported for completeness. % <!-- evidence: C10 -->
```

**After:**
```
\textbf{Flat-vector recurrence (the additive control).} Identical
embedding table, output head, and feed-forward blocks; only the mixer
changes, to a gated elementwise vector recurrence
$s_t = s_{t-1} \odot g_t + v_t$ with read $o_t = s_t \odot q_t$, at matched
total parameter count via mixer width and $d_{\mathrm{state}} = 64$ pinned
equal to the contender's. This arm removes the multiplicative
outer-product update while preserving everything else, rather than
reshaping the same operator, since any $d^2$-dimensional vector can encode
a $d \times d$ matrix and structure matters only if the operations use it.
This control differs from the contender on three axes, not one: mixer
recurrence, raw state capacity (512 bytes versus 32{,}768,
% <!-- evidence: C10 -->
reported for completeness), and write conditioning -- its update has no
dependence on the key $k_t$ at all, so it cannot perform content-addressed
binding under any training outcome, unlike vector-native binding schemes
that use a key-dependent operation such as circular convolution
\citep{plate1995holographic} or tensor-product binding
\citep{smolensky1990tensor}. Section~\ref{sec:recall} states which
reading of the resulting comparison the capacity and write-conditioning
gaps license.
```

#### 1b. `sections/03_recall.tex` — rescope the "licensed reading" paragraph

**Location:** the paragraph beginning "Two readings of this table are
licensed."

**Before:**
```
Two readings of this table are licensed, and one is not. Licensed: the
outer-product state update is doing task-critical work its
parameter-matched additive counterpart does not do, since the flat-vector
arm differs only in the mixer recurrence. Also licensed, with its
registered phrasing: the transformer baseline is \emph{non-competitive at
matched params/tokens}; at this scale and budget it does not learn the
task at all, which is itself the axis-2 datum, claimed only at this
matched training budget and never extrapolated to differently-scaled or
differently-trained attention models. Not licensed: any claim that a
transformer \emph{cannot} perform this task. Attention solves associative
recall well in other regimes \citep{arora2023zoology,
nichani2024understanding}; what the matched budget shows is that under
this task distribution, tokenizer, parameter count, and step count, the
attention arm did not get there while the fast-weight arm reached ceiling
in every seed.
```

**After:**
```
Three readings of this table are licensed; two are not. Licensed: under
this shared training budget, the matrix fast-weight update reaches
ceiling while the additive control's training loss does not move past its
first-500-step value for the remaining 97.5\% of the run
(Appendix~\ref{app:traincurve}); learning rate, warmup, and
auxiliary-loss weighting were shared across three architecturally
distinct mixers, not independently tuned per arm, so this paper does not
claim the additive update is \emph{architecturally incapable} of the
task, only that it does not reach it under this matched budget. Licensed,
and narrower than a "matrix beats vector" claim: because the additive
control also holds $64\times$ less raw state and a write with no
dependence on the query key (Section~\ref{sec:setup}), the comparison
supports \emph{a matrix state with a multiplicative, key-conditioned
write outperforms a smaller, non-key-conditioned vector state under a
shared training budget}; capacity and write-conditioning are not
disentangled by this ablation, and no resized- or key-conditioned-vector
control was run. Also licensed, with its registered phrasing: the
transformer baseline is \emph{non-competitive at matched params/tokens};
at this scale and budget it does not learn the task at all, which is
itself the axis-2 datum, claimed only at this matched training budget and
never extrapolated to differently-scaled or differently-trained attention
models. Not licensed: any claim that a transformer \emph{cannot} perform
this task, or that outer-product structure specifically, rather than
capacity and key-conditioning together, is what the additive control's
failure isolates. Attention solves associative recall well in other
regimes \citep{arora2023zoology, nichani2024understanding}; what the
matched budget shows is that under this task distribution, tokenizer,
parameter count, and step count, the attention arm did not get there
while the fast-weight arm reached ceiling in every seed.
```

Note the deliberate absence of a specific "$K$-bindings recover
$1.000000$ vs.\ $0.000000$" number: the underlying argument for why
enlarging the additive control's state would not close the gap is stated
qualitatively via the write rule already given in the paper
($s_t \odot q_t$ is a per-coordinate scalar operation, one degree of
freedom per coordinate, independent of $d$) rather than by citing the
un-archived registry diagnostic — do not add that citation or those
decimals; see §0 above.

#### 1c. New appendix figure — training-loss trajectories

**File:** `figures/figure-gen.py`. Add a new figure function, call it
from `main()`, using the same three already-md5-asserted sources already
loaded for `n_params` (`train_contender`, `train_ablation`,
`train_transformer` — no new `DATA_SOURCES`/`SOURCE_MD5` entries, no new
evidence row required, this is C9's existing artifact set):

```python
def fig3_traincurve(repo, out_dir, plt):
    """Training-loss trajectories, all three arms, task1 seed 0 (A1 fix).
    Same archived configs already used for the parameter-count row (C9)."""
    arms = [("contender", "matrix fast-weight", STYLE["ink"], "-"),
            ("ablation", "flat-vector recurrence", STYLE["accent"], "--"),
            ("transformer", "transformer", STYLE["gray"], ":")]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for a, label, color, ls in arms:
        curve = _load(repo, f"train_{a}")["curve"]
        steps = [row["step"] for row in curve]
        losses = [row["train_loss"] for row in curve]
        ax.plot(steps, losses, color=color, lw=1.3, ls=ls, label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("train loss")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_traincurve.pdf"))
    plt.close(fig)
```

Add `fig3_traincurve(args.repo, args.out, plt)` next to the
`fig1_horizon`/`fig2_szero` calls in `main()`.

**File:** `sections/10_appendix.tex`. Insert the new section between the
end of the `app:taps` table (`\end{table}` after `\label{tab:taps}`) and
`\section{Reproducibility of every number}`:

```latex
\section{Training-loss trajectories, all three arms}
\label{app:traincurve}

\begin{figure}[H]
\centering
\includegraphics[width=0.62\linewidth]{figures/fig3_traincurve.pdf}
\caption{Training loss (task1, seed 0) over the full 20{,}000-step run,
all three arms, from the same archived configs used for the parameter
count in Section~\ref{sec:setup}. The flat-vector and transformer arms
drop from initialization to $\approx$7.7--7.8 within the first 500 of
20{,}000 steps and do not move further; the matrix fast-weight arm keeps
falling for the entire run, reaching 1.38. Learning rate
($3\times10^{-4}$), warmup, and auxiliary-loss weighting were shared
across the three architecturally distinct mixers and were not
independently tuned per arm; a per-architecture learning-rate
sensitivity sweep was not run. The claim in Section~\ref{sec:recall} is
scoped to this shared training budget accordingly.} % <!-- evidence: C9 -->
\label{fig:traincurve}
\end{figure}
```

#### 1d. `refs.bib` — add the two vector-binding citations

**Location:** anywhere in the file (alphabetical/thematic grouping not
enforced elsewhere in this file).

```bibtex
@article{plate1995holographic,
  title   = {Holographic Reduced Representations},
  author  = {Plate, Tony A.},
  journal = {IEEE Transactions on Neural Networks},
  volume  = {6},
  number  = {3},
  pages   = {623--641},
  year    = {1995}
}

@article{smolensky1990tensor,
  title   = {Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems},
  author  = {Smolensky, Paul},
  journal = {Artificial Intelligence},
  volume  = {46},
  number  = {1-2},
  pages   = {159--216},
  year    = {1990}
}
```

`refs.bib`'s own header comment states every entry was checked against
the arXiv API or CrossRef before this gauntlet round; these two are
neither, so run the same CrossRef check the file already did for
Schmidhuber 1992 (DOI lookup by title/author) before submission and add
the DOI to each entry, matching the file's existing convention.

#### 1e. Regenerate and re-render

Run `figure-gen.py`, confirm it produces `fig3_traincurve.pdf` with no
md5 errors, and re-check the total page count against the 10-page hard
cap (the brief records 2.75pp of slack; this fix adds roughly
0.2–0.3 pages of figure plus a longer paragraph in Section 3 — should fit,
but verify, don't assume).

**Why this resolves A1 and A2.** A1's core objection was that the
paper's "licensed reading" implies the additive update is
architecturally incapable of the task when the evidence (flat loss for
97.5% of the run, no gradient-health artifact, shared untuned
hyperparameters across structurally different mixers) supports only a
narrower, budget-scoped claim. The rewrite states exactly that scope and
adds the standard figure a reviewer would expect (zero new compute — the
`curve` field was already sitting in an already-archived, already-verified
source). A2's core objection was that "differs only in the mixer
recurrence" is false on the paper's own numbers (64$\times$ capacity gap,
zero key-conditioning in the write) and that the "licensed" sentence
overclaims what a confounded ablation supports. The rewrite discloses
both axes explicitly, narrows the claim to what is actually isolated
(capacity + key-conditioning jointly, not "matrix structure" alone), and
gives a citable, self-contained (non-registry) reason why size alone is
not the whole story, without introducing any new unevidenced number.

---

### FIX-2: Make the capped-KV cross markers visible in the money figure

**Severity:** SERIOUS
**Resolves:** A3
**File(s):** `figures/figure-gen.py`
**Location:** `fig1_horizon`, the "Transformer uncapped" and "Transformer
capped" plotting blocks (currently lines ~133–151).

**Before:**
```python
    # Transformer uncapped, per seed.
    unc = verdict["transformer_uncapped_refs_acc_A"]
    for s in range(3):
        ys = [unc[h][s] for h, _ in horizons]
        ax.plot(xs, ys, color=STYLE["accent"], lw=1.4, ls="--", marker="s",
                ms=4, zorder=4,
                label="transformer, uncapped KV cache" if s == 0 else None)

    # Transformer capped, every M in the walk grid, every seed: gray crosses.
    capped = verdict["capped_transformer_acc_A"]
    first = True
    for m_key, by_h in capped.items():
        for h, t in horizons:
            for s in range(3):
                ax.plot(t, by_h[h][s], color=STYLE["gray"], marker="x", ms=4,
                        ls="none", zorder=3,
                        label=(r"transformer, KV capped at $M\times$32,768 B, "
                               r"$M\in\{2,\ldots,32\}$") if first else None)
                first = False
```

**After:**
```python
    # Transformer uncapped, per seed. alpha<1 so the capped-M crosses
    # (plotted above, higher zorder) remain visible through the dashed
    # line where they overlap in the near-chance band.
    unc = verdict["transformer_uncapped_refs_acc_A"]
    for s in range(3):
        ys = [unc[h][s] for h, _ in horizons]
        ax.plot(xs, ys, color=STYLE["accent"], lw=1.4, ls="--", marker="s",
                ms=4, zorder=4, alpha=0.6,
                label="transformer, uncapped KV cache" if s == 0 else None)

    # Transformer capped, every M in the walk grid, every seed: gray crosses.
    # zorder=6, above both the dashed uncapped line (4) and the solid
    # contender line (5) -- round-1 gauntlet (A3) found these markers
    # rendering invisible beneath the dashed line at the previous zorder=3.
    capped = verdict["capped_transformer_acc_A"]
    first = True
    for m_key, by_h in capped.items():
        for h, t in horizons:
            for s in range(3):
                ax.plot(t, by_h[h][s], color=STYLE["gray"], marker="x", ms=5,
                        mew=1.2, ls="none", zorder=6,
                        label=(r"transformer, KV capped at $M\times$32,768 B, "
                               r"$M\in\{2,\ldots,32\}$") if first else None)
                first = False
```

Regenerate (`figure-gen.py`), then **visually re-inspect**
`fig1_horizon.pdf` (e.g. render to PNG at 300dpi and view it) before
trusting this fixed — this is exactly the class of defect ("code plots it
but the render doesn't show it") that a diff review would miss. If the
crosses are still visually indistinguishable from the dashed line at this
zoom level, add a zoomed inset axis over the $y\in[0,0.06]$ band
(`ax.inset_axes`) as a second pass; not required if the zorder/alpha
change alone makes the crosses legible.

**Why this resolves A3.** The attack and defense agree the crosses are
real data occluded by draw order, not missing data. Reordering the
`zorder` and lightening the occluding line is the mechanical fix both
reports converge on; no data or caption changes are needed.

---

### FIX-3: Correct the seed-1 $S_1$-zeroing statistical framing

**Severity:** SERIOUS
**Resolves:** A4
**File(s):** `sections/05_mechanism.tex`
**Location:** the "One instrument note" sentence at the end of the first
paragraph of "The first block's state is causally necessary; the second
block's is inert."

**Before:**
```
One instrument note: at a seed reading exactly 1.0 intact, the binomial
band used for the ``unchanged'' check degenerates to zero width, so the
0.0051 dip at seed 1 under $S_1$-zeroing is disclosed as an edge case of
the check rather than adjudicated by it. % <!-- evidence: C5 -->
```

**After:**
```
One instrument note: at seed 1 ($\mathrm{acc\_intact} = 1.0$),
$S_1$-zeroing produces a small, reproducible change on 21 of 4{,}096
queries (0.51\%, $\Delta = 0.0051$). % <!-- evidence: C5 -->
The binomial standard-error band used elsewhere in this check assumes a
sampling process and is ill-posed at $p=1$ (it evaluates to zero width);
because both the intact and $S_1$-zeroed readings are deterministic
argmax decodes of the same frozen checkpoint on the same fixed query set,
this is not measurement noise but a real, deterministic behavior change
under the intervention, so the band is not used to adjudicate this cell.
The effect is confined to $S_1$, the block Figure~\ref{fig:szero} shows is
causally inert for the $S_0$-necessity result, and does not affect it;
whether the 21 queries reflect a small, consistent causal contribution
from $S_1$ or near-tied logits is not established by this
instrument. % <!-- evidence: C5 -->
```

**Why this resolves A4.** The attacker's statistical point — that a
zero-width binomial band on a deterministic, paired comparison means any
nonzero delta is a real, repeatable effect, not an artifact the band
should have caught — is correct, and the raw JSON's own `unchanged: false`
field already agrees. The rewrite reports the raw count plainly, corrects
the causal framing (the band is ill-posed, not "protecting" against
noise), and states clearly what is and is not established, without
touching the $S_0$-necessity headline, which this attack never disputed.

---

### FIX-4: Scope the constant-memory claim to the tested horizon band

**Severity:** SERIOUS
**Resolves:** A5
**File(s):** `sections/08_limitations.tex`
**Location:** end of the (single) Limitations paragraph.

**Before (final sentence):**
```
Horizon evaluation extends context
with filler continuation, not with distractor bindings, which is the
easier long-context condition and is disclosed as such.
```

**After:**
```
Horizon evaluation extends context
with filler continuation, not with distractor bindings, which is the
easier long-context condition and is disclosed as such. The longest
tested context is 1{,}798 tokens, eight times the binding
span: % <!-- evidence: C3 -->
this paper demonstrates the constant-memory \emph{mechanism} as a
small-scale existence result, not a claim that it holds at the context
lengths (tens of thousands to millions of tokens) where KV-cache cost is
a practical deployment constraint.
```

**Why this resolves A5.** The gap the attacker identifies (motivating
framing at deployment scale; testing at $\leq$8$\times$ the binding span)
is real and the existing Limitations scale caveat is written around model
size, not context length. This sentence closes that specific gap in the
place a reviewer checks first, using language already consistent with the
Conclusion's existing "small-scale existence result" framing (so it
reads as reinforcement, not a new admission).

---

### FIX-5: Ground the blank-out guarantee in a checked mechanism, described in-line

**Severity:** MINOR (downgraded from the attacker's SERIOUS — see §3)
**Resolves:** A6
**File(s):** `sections/05_mechanism.tex`
**Location:** opening paragraph of Section 5.

**Before:**
```
A separation on accuracy alone leaves open where the winning arm keeps the
bindings. We intervene directly: after the bind phase, the recurrent
states are frozen, one block's state is replaced with zeros, and the query
continuation runs as a pure function of the remaining state and the query
tokens (the decoder reads only the state, never the raw bind tokens, and a
blank-out test on the continuation path enforces this by construction).
```

**After:**
```
A separation on accuracy alone leaves open where the winning arm keeps the
bindings. We intervene directly: after the bind phase, the recurrent
states are frozen, one block's state is replaced with zeros, and the query
continuation runs as a pure function of the remaining state and the query
tokens. The decoder reads only the state, never the raw bind tokens, and
this is checked rather than assumed: a fresh model instance given only
the cached state (no raw bind-phase tokens) reproduces bit-identical
continuation logits to the original run, closing the channel by which a
hidden cache of the raw tokens could otherwise leak into the query pass.
```

**Why this resolves A6.** The attack's concern was that "enforces this by
construction" reads as an architectural assertion the paper never shows
was checked, unlike every other claim in the draft. The underlying check
is real (independently confirmed against the design registry: a
fresh-instance, state-only continuation producing bit-identical logits),
but it is a pass/fail structural guarantee, not a quoted number, so it
does not need (and per Appendix C's own scoping should not manufacture) a
numbered evidence row the way C1–C10 do. Describing the mechanism inline,
without an external citation to a non-public document, both answers the
attack and avoids the anonymization/citability problem flagged in §0.

---

### FIX-6: Disclose the seed-count rationale and the single-decision-rule scope

**Severity:** MINOR
**Resolves:** A7, A8
**File(s):** `sections/02_setup.tex`
**Location:** end of "Analysis protocol, frozen before the runs."

**Before (final sentence of the subsection):**
```
Evaluation
uses a pinned held-out query set of 4{,}096 queries over 128 episodes per
cell, % <!-- evidence: C3 -->
identical across arms and seeds.
```

**After:**
```
Evaluation
uses a pinned held-out query set of 4{,}096 queries over 128 episodes per
cell, % <!-- evidence: C3 -->
identical across arms and seeds. Three seeds is the registered count for
the primary comparison; the protocol includes a seed-extension trigger
for any seed whose paired gap approaches the 0.30 margin, and none did
(Table~\ref{tab:main}), % <!-- evidence: C2 -->
so the count was not extended for this comparison. The multi-hop
variant's later nine-seed table (Section~\ref{sec:scope}) reflects a
different, seed-variance-driven trigger under the same protocol, not a
change in standard. This protocol treats the primary contender-versus-
additive-control comparison as the single pre-registered decision rule;
the remaining reported intervals in this paper (the per-$M$
capped-budget gaps, the pooled task2 interval, per-seed bar comparisons)
are descriptive and diagnostic rather than additional hypothesis tests,
so no multiple-comparison correction is applied to them.
```

(Note: "additive control" matches the FIX-1 rename; if FIX-1 is applied
first, use "additive control" here for consistency. If FIX-6 is applied
before FIX-1 for some reason, use "flat-vector arm" instead and update
this sentence when FIX-1 lands.)

**Why this resolves A7 and A8.** Both attacks concede no conclusion is
threatened; both ask for a one-paragraph acknowledgment of standard
statistical caveats (thin-by-convention seed count for the headline
comparison; no multiple-comparison correction across the paper's many
diagnostic intervals). This paragraph states both, in the section a
statistically careful reviewer checks first (Analysis Protocol), using
only numbers already established elsewhere in the draft.

---

## 2. Verdict table

| Attack | Severity (attack) | Defense disposition | Final verdict | Fix ID |
|---|---|---|---|---|
| A1 — flatlined baseline losses, no gradient-health artifact | CRITICAL | CONCEDE + FIX | DEFENSE INSUFFICIENT (resolved by fix) | FIX-1 |
| A2 — additive-control ablation confounds capacity (64×) and write key-conditioning | CRITICAL | CONCEDE + FIX | DEFENSE INSUFFICIENT (resolved by fix) | FIX-1 |
| A3 — money figure's capped-KV crosses occluded by draw order | SERIOUS | CONCEDE + FIX | DEFENSE INSUFFICIENT (resolved by fix) | FIX-2 |
| A4 — seed-1 $S_1$-zeroing statistical framing backwards | SERIOUS | CONCEDE + FIX | DEFENSE INSUFFICIENT (resolved by fix) | FIX-3 |
| A5 — motivating deployment-scale framing tested 3–4 orders of magnitude below deployment context lengths | SERIOUS | PARTIAL | DEFENSE VALID BUT EDIT | FIX-4 |
| A6 — blank-out guarantee asserted without a quotable reported result | SERIOUS (attacker); MINOR (adjudicated, §3) | PARTIAL | DEFENSE VALID BUT EDIT | FIX-5 |
| A7 — three seeds thin for the primary comparison | MINOR | PARTIAL | DEFENSE VALID BUT EDIT | FIX-6 |
| A8 — no multiple-comparison correction across pre-registered tests | MINOR | PARTIAL | DEFENSE VALID BUT EDIT | FIX-6 |

**Disposition counts:** DEFENSE VALID: 0. DEFENSE VALID BUT EDIT: 4 (A5,
A6, A7, A8). DEFENSE INSUFFICIENT: 4 (A1, A2, A3, A4) — all four resolved
below CRITICAL/SERIOUS by an in-scope fix using only existing evidence.
PARTIAL — ATTACK SURVIVES IN REDUCED FORM: 0.

---

## 3. Adjustments to the attacker's/defender's severity ratings

- **A6 downgraded SERIOUS → MINOR.** The defense's re-tracing of this
  claim into the design registry (`§1.3.1.3` blank-out test, `R5-F4`
  fresh-instance companion) checks out — I independently grepped the
  registry and confirmed the same two citations the defense found, and
  confirmed (via a separate grep across all three relevant
  `experiment-runs/` directories for "blank") that no archived JSON
  backs it with a quotable number, also as the defense found. This is a
  true, verified, non-numerical structural claim written without its
  citation — a hygiene gap, not a validity threat, and the attacker's own
  text concedes the paper's Appendix C promise (about numerical claims)
  technically does not cover it. I agree with the defense's downgrade.
- **A1 and A2 confirmed CRITICAL** on independent review of the same
  nine training JSONs (all `curve` fields and all `grad_ratio_at_tap_step500`
  fields, not a sample) and the `02_setup.tex`/`03_recall.tex` text —
  both attacks are exactly as damaging as rated, because both point at a
  gap between what the comparison structurally supports and what the
  paper's prose claims is "licensed." I found one additional problem
  neither attacker nor defender raised (§0, item 1): the defense's
  proposed A2 fix would have introduced a new quoted number with no
  archived evidence row, which is itself close to the kind of defect A6
  flags elsewhere in the paper. FIX-1 avoids this.
- **A5 kept SERIOUS**, matching the attacker; the defense's "borderline
  SERIOUS/MINOR" framing understates how load-bearing the intro's
  deployment-scale framing is — it is literally the paper's motivating
  stakes, so a reviewer noticing the horizon gap has grounds to discount
  the whole introduction's framing, not just one section, absent the
  fix.
- **A3, A4, A7, A8** — no adjustment; attacker and defender agree, and I
  found nothing in the raws that changes either the facts or the
  severity.

---

## 4. Residual risk after all fixes are applied

**Workshop-survivable (acceptable to submit with, disclosed):**

- **A1 residual:** even after FIX-1, the paper does not determine whether
  the additive control and transformer would learn the task under a
  properly-tuned per-architecture LR/warmup — it now honestly says it
  doesn't know, rather than implying it does. A reviewer who wants a
  stronger causal claim (not just a budget-scoped one) is a legitimate
  ask for the next revision (a cheap same-day per-arch LR sweep, flagged
  by the defense as a camera-ready strengthening, not a submission
  blocker) but is not a validity threat to the narrower claim now made.
- **A2 residual:** the write-key-conditioning axis stays genuinely
  undisentangled from the capacity axis — FIX-1 discloses this
  explicitly rather than closing it (no HRR-style key-conditioned vector
  control was run, and none is available before the deadline). The
  rescoped claim (matrix + key-conditioned write beats a smaller,
  non-key-conditioned vector under a shared budget) is fully supported;
  the stronger claim the paper no longer makes (matrix structure alone
  is necessary) would require that new ablation.
- **A4 residual:** whether the 21/4,096 seed-1 flip reflects a genuine
  small $S_1$ causal contribution or near-tied-logit noise is disclosed
  as unresolved. It does not touch the $S_0$-necessity headline.
- **A5, A6, A7, A8 residuals:** none beyond what the disclosure sentences
  already state; these were low-severity to begin with.

**Conference-blocking risk if this paper moves beyond a non-archival
workshop:** A1 and A2's residuals above (no per-architecture LR
sensitivity check; no key-conditioned vector control) would very likely
be pushed harder by a full-conference reviewer expecting a stronger
causal isolation of "why the outer-product update wins." Both are
disclosed as open, not closed, in this fix pass, which is appropriate for
a non-archival workshop submission but would need real follow-up
experiments (flagged, not run, per the no-new-evidence constraint on this
gauntlet round) before a flagship/full-paper submission of the same
result.

**Process note for future rounds:** if either follow-up experiment above
is later run (per-arch LR sweep, or a key-conditioned vector control),
treat the resulting claims as new and re-enter the full gauntlet on them
— do not fold a new number into FIX-1's language after the fact without
a fresh attack pass, especially given what §0 found about the previous
attempt to shortcut this with an un-archived number.

---

## 5. Re-run scope for the writer

FIX-1 rescopes a central interpretive claim (contribution #1's mechanistic
reading), so per the gauntlet's re-run rule, the following must re-enter
attack/defense/rebuttal, not just get edited and shipped:

- `sections/02_setup.tex` (the additive-control paragraph, edited)
- `sections/03_recall.tex` (the licensed-reading paragraph, edited)
- `sections/10_appendix.tex` (new `app:traincurve` section)
- `figures/figure-gen.py` and the new `fig3_traincurve.pdf`
- `refs.bib` (two new entries)
- **Check for consistency, no edit expected:** `sections/00_abstract.tex`
  and the contribution-bullet list in `sections/01_intro.tex` — neither
  currently states the "outer-product... only differs" language FIX-1
  rewrites, so they should not need changes, but re-verify they don't
  contradict the now-narrower Section 3 claim once it's rewritten.

FIX-2 (figure) does not need a full attack/defense re-run (it changes no
claim, only a rendering bug), but it **does** need the mechanical
re-render-and-visually-inspect step specified in FIX-2 before submission
— this exact category of bug (code is correct, render isn't) is easy to
mis-verify by re-reading the diff instead of looking at the output.

FIX-3, FIX-4, FIX-5, FIX-6 are text-only disclosure edits with no new
claims beyond what the raw JSONs already support; a light consistency
check (do the new sentences use numbers that match the tables) is
sufficient, not a full re-attack.

---

## 6. New citations

**MUST-CITE** (load-bearing for FIX-1, blocks the fix if omitted):

- Plate, T. A. (1995), "Holographic Reduced Representations," IEEE
  Transactions on Neural Networks, 6(3):623–641.
- Smolensky, P. (1990), "Tensor Product Variable Binding and the
  Representation of Symbolic Structures in Connectionist Systems,"
  Artificial Intelligence, 46(1-2):159–216.

**SHOULD-CITE** (strengthens Related Work if the 0.5pp page budget
allows; not required by any fix above):

- Olsson, C. et al. (2022), "In-context Learning and Induction Heads,"
  Anthropic Transformer Circuits Thread. Context for why a 2-layer
  transformer's complete failure at this task class is a surprising
  result worth a sentence in Related Work; not load-bearing since FIX-1's
  text does not depend on it.
- Yang, S. et al., "Gated Delta Networks," arXiv:2412.06464. Nearest
  follow-on to the contender's own delta-rule mechanism; current Related
  Work stops at the 2024 parallelization paper.
- Behrouz, A., Zhong, P., Mirrokni, V., "Titans: Learning to Memorize at
  Test Time," arXiv:2501.00663. Same framing (fixed-size memory vs.
  growing KV cache) as this paper's core motivation; a workshop reviewer
  in this space would likely expect it distinguished from.
- Arora, S. et al., "Based," arXiv:2402.18668. Direct follow-up to the
  already-cited Zoology paper, same authors, on the recall/efficiency
  tradeoff.
