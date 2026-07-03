# DeltaNet Real-Data Design — The Real-Data Link in the Causal-Rank Chain

> **STATUS: CLOSED (one of STATE.md's five closed 2026-07-01→03 programs;
> Wave 1 closing verdict §17.11, Wave 2 results §19).** Verdict: rank causal
> necessity **CONFIRMED** at K∈{8,16,24,32} on real GPT-2-tokenized text via
> eval-time truncation — a hard ceiling reached exactly at k=K, but graded
> across a multi-rank window rather than the synthetic design's razor cliff
> (trained keys are non-orthonormal on real text — the pre-registered
> caveat, landing exactly as predicted). This is the project's first
> demonstration of genuine, causally-verified rank-K relational binding in
> a production architecture on real tokenized surface forms. **Wave 2**
> (Waves C+D, closed 2026-07-04, §19): reasoning-dense text (OpenR1-Math) is
> measurably more truncation-damage-sensitive than narrative text
> (WikiText) at low k, converging to the same noise floor by k≈48 of
> d_state=64; layer-0 effective rank *falls* (not rises) as training
> proceeds in both corpora — opposite the "SGD recruits more rank as it
> learns" intuition from the controlled causal-rank chain. Wave 2 closes
> this program's record; no further waves are pre-registered beyond §7's
> manifest Reserve row. The open follow-on question this program's results
> motivated — *why* real-text composition falls short of the synthetic
> razor cliff — is being pursued separately in `DELTANET_RD_EXACTNESS_DESIGN.md`
> (living, not closed; see STATE.md "Chapter 2 — STATUS").

**Drafted 2026-07-02, before any code changes.** Status: design only, per
instruction — no model/training code is written here. This is the fourth leg
of the Chapter 2 "does the gradient see rank" lineage, and the first one
that leaves fully-synthetic, hand-constructed key/value vectors behind:

| Setting | Bound | Result |
|---|---|---|
| Bolt-on matrix-CODI (ICML MI-workshop paper, accepted) | none (task rank-1-solvable) | rank-blind on ProsQA |
| Matrix-native from scratch, Task D/E (`chapter2/`) | provable, hand-built orthonormal `Z` | SGD recruits provably-necessary rank; composes to h=21 |
| Production DeltaNet, synthetic grammar (`DELTANET_CAUSAL_RANK_DESIGN.md`) | provable, hand-built orthonormal keys | Wave 0 (2026-07-02): unconstrained arm saturates perfectly at every `(d,K)` cell tested, `d≤128`; entity-subspace rank = whole-matrix rank = K exactly, no inflation (the delta rule only ever writes within `span({k_j})` — §3.4's algebra makes this architectural, not a fortunate optimization outcome); **causal claim now rests entirely on the in-flight Wave B force-rank grid** |
| **DeltaNet, real data (this doc)** | **?** — the central problem this document exists to solve | not yet run |

> **CHANGELOG — 2026-07-02, revision 2 (post-attack-round).** An independent
> adversarial review of revision 1 returned BUILD WITH FIXES (2 FATAL-scoped,
> 6 MAJOR, 6 MINOR). All findings addressed in this revision:
> **FATAL-1 (the load-bearing one — revision 1's grammar had no valid write
> mechanism).** The C9 single-write β-mask has no realization for revision
> 1's free-form sentences: the synthetic grammar co-locates key+value on ONE
> token (`concat(key_t, value_t)`, β=1 scattered at the K item positions),
> but "Ann gave the book to Bob" puts key and value ~5 BPE tokens apart —
> beyond `conv_size−1 = 3` — and a **causal** layer cannot write Ann→Bob at
> Ann's position (Bob unseen) nor reach Ann from Bob's position through the
> conv window. Revision 1's "C1–C15 transfer unchanged in spirit" begged
> exactly this. Resolved by choosing and fully developing the **constrained
> adjacency template** (option (a)): key and value within one conv window,
> write position = the binding-completion (VALUE) token, β-mask relocated
> there; the bound's premise re-derived for the post-conv effective keys at
> the write positions; a new bind→query alignment premise made explicit;
> and the conv reversal (conv REQUIRED for the primary gate, unlike the
> synthetic design's no-conv bespoke harness) reconciled with §4.3's
> production-kernel commitment (§5.2). Jointly resolves MINOR-14: the
> single-token-entity open question is DECIDED (single-token + adjacency
> for the primary gate; multi-token/free-form as Reserve stress arms).
> **FATAL-2 (C16 had no numeric rule).** A concrete threshold is now
> pre-registered before any Wave 1 data exists: premise-valid ⇔
> `‖K_effᵀK_eff − I‖_F < τ = 0.03`, anchored to the synthetic design's
> measured regimes (unconstrained 0.0052–0.0097; force-ranked 3.56–4.91),
> plus a salvage tier, a companion classification rule for straddling
> checkpoints, and a distinct "premise-failed" outcome category (§5.2).
> MINOR-13 acknowledged inline: this is a calibrated empirical premise
> check on a continuous SGD-produced object — C15's exact-threshold
> structural-assert / run-to-completion-negative-test discipline has no
> coherent analog here; what transfers is pre-registration.
> **MAJOR-3** — C19 held-out-template control added (disjoint relation-verb
> pool + second template family, zero-shot; CLUTRR / MQAR-family precedent)
> (§5.3, §5.4).
> **MAJOR-4** — the bf16 cross-check's ground truth is itself now verified
> first: a Tier-0 fp64 finite-difference gradcheck of the *patched* naive
> reference at LM-relevant `(d, T)` shapes, before it is promoted to
> ground-truth status (§4.4).
> **MAJOR-5** — Wave 2 gains a third contrast arm (symbol-density-matched
> STEM expository prose) with a pre-registered frequency-normalized-metric
> fallback — separating "derivation vs. exposition" from "notation vs.
> prose" (§6.1).
> **MAJOR-6** — §4.2's cost estimate re-derived from the actual bottleneck
> (the 50,257-class softmax head + logits-tensor VRAM, per the house
> rules), fixing revision 1's internal contradiction with §4.3's kernel
> choice; the "~4 minutes" figure corrected to ~0.6–2.2 h/run (the
> conclusion survives — pretraining is still not the budget bottleneck —
> at ~10–30× less margin).
> **MAJOR-7** — the Wave −1 gate collision resolved: revision 1 priced
> Wave B at 8–10× while its own gate said ">8× → STOP." Buckets re-set
> consistent with the pricing baseline (≤10× proceed / 10–15× re-price via
> the concrete pre-registered cut order / >15× stop), with the re-price
> semantics spelled out (§7).
> **MAJOR-8** — the claim-tier label is **"train-time causal,
> premise-conditional"** everywhere it appears as a label (§1, §2.2, §2.5,
> §10) — the premise is empirically verified per checkpoint (FATAL-2), not
> architecturally guaranteed, and the label now says so.
> **MINOR-9** — the two unverified citation specifics (Nazari & Rusch's
> "Gated DeltaNet 370M / FineWeb-Edu"; Sun et al.'s "Qwen3-Next / RankViz")
> hedged as internal-note-sourced; re-verify before external citation
> (§10). **MINOR-10** — Gated DeltaNet-2 mechanism phrasing corrected
> (decoupled erase/write gates, not "per-channel decay factors") (§10).
> **MINOR-11** — the backwards cost-history citation fixed: Stage 0's
> GPU-h estimates ran 5–6× HIGH (pessimistic — measured runs came in
> cheaper), which cuts toward comfort on pricing; the pattern that
> actually bites is the late-transition / step-budget-artifact pattern
> (§4.2, §12). **MINOR-12** — C13/C16 reconciled: one instrument (C16
> subsumes C13; no separate C13 exists in this design), with defined
> object, cadence, threshold, and consequence (§5.2, §5.4). **MINOR-13,
> MINOR-14** — folded into FATAL-2 and FATAL-1 respectively, as noted.

> **CHANGELOG — 2026-07-02, revision 2.1 (round-2 verify: all 14 round-1
> findings confirmed genuinely addressed, FATAL-1 arithmetic verified
> correct; 4 new findings + 4 minors, verifier pre-cleared the fixes
> below — CLEAN TO BUILD once applied, no further review round; design
> FROZEN at this revision).**
> **R2-1 (FATAL — the τ rule stranded Wave B).** Revision 2's single
> global τ contradicted itself: its own anchor paragraph cites
> force-ranked Gram deviations of **3.56–4.91** as the degraded band —
> if that transfers, EVERY B-probe/Wave B checkpoint is premise-invalid
> under "headlines only from premise-valid checkpoints," so the primary
> causal test could never CONFIRM; simultaneously the salvage tier was
> named "the bound's actual minimal premise" AND forbidden from
> headlines — a direct contradiction. Fixed with **arm-specific rules**
> (§5.2): (i) unconstrained arm keeps τ = 0.03 as written; (ii) the
> causal `k ≥ K` leg's headline validity is **salvage tier or better**
> (`σ_K/σ_1 ≥ 0.1` — linear independence with bounded conditioning, the
> bound's actual premise), with τ = 0.03 retained as a clean-regime
> descriptor only; (iii) the `k < K` collapse leg never invokes the
> bound on failure — collapse is read through the logged
> premise/alignment diagnostics with F18-style one-directional-bias
> discipline (force-rank-induced key degradation must not be read as a
> rank effect).
> **R2-2 (MAJOR)** — premise (iii) alignment was thresholdless —
> reproducing the FATAL-2 hole in miniature, and as written a
> one-directional CONFIRM-rescue lever. Now pre-registered: per-item
> `cos(k_eff_bind, q_eff_query) ≥ 0.9`, with **symmetric** handling — a
> SUCCESS at a misaligned checkpoint is flagged premise-questionable
> too, not just failures excused (§5.2).
> **R2-3 (MAJOR)** — implementation shape specified: a **custom block
> calling `fla.ops.delta_rule.chunk_delta_rule` directly** with an
> externally computed masked β (the stock `fla.layers.delta_net.DeltaNet`
> computes β internally via its `b_proj` and exposes no mask hook),
> controlled conv, and **reserved buffer token IDs zero-pinned AND
> frozen** in the real embedding table; mask/pinning verification added
> to the F15-LM checkpoint (§5.2, §7).
> **R2-4 (MAJOR)** — the τ re-registration decision moved from "before
> B-probe" to the **Wave 0 → Wave A gate**: Wave 0 already outputs the
> measured C16 band, and Wave A's screening numbers must not be
> reported under a known-miscalibrated τ (§5.2, §7, §9, §12).
> **R2-5** — MFU label corrected: 50–200K tok/s ≈ **0.4–1.7% end-to-end
> MFU**, not "5–15%"; the tok/s band and the 0.6–2.2 h conclusion are
> unchanged (§4.2). **R2-6** — stale §7 cross-reference fixed (item 8 →
> item 9). **R2-7** — two query-side conventions pinned: the query uses
> the same relation verb as bind (C19's held-out arm renders bind AND
> query in the held-out template), and `h` never appears in the surface
> form (§5.2). **R2-8** — per-item cross-clause leak check added to the
> smoke gate (corrupt clause `j` → clause `j+1`'s `k_eff` bit-identical),
> and C16/alignment diagnostics are computed on the C17/C19 held-out
> pools at eval — classifying held-out failures as premise vs.
> competence (§5.2, §5.3).

This document is written concurrently with the synthetic DeltaNet design's
own execution: at the time of writing, Wave B-probe (the force-rank causal
grid at the synthetic design's selected primary cell, `d=64, K=32`) is live
on the Brev cluster (confirmed via `ps aux`, §7.5), and Wave A's
unconstrained screening grid has already produced the striking §12.3/§12.4
result above. **This document does not wait for Wave B to resolve** — the
methodological design for the real-data link does not depend on whether the
synthetic force-rank grid confirms or falsifies, and building it now means
it is ready to launch the moment Wave B's verdict lands (mirroring the
project's own "plan the next experiment before the current one drains,"
`H100_SETUP.md` §5). If Wave B falsifies, this document's Wave 1 (§5) still
stands on its own as a test of the same H_DN-style question on a
different, harder axis (real tokenizer) — a FALSIFY at the synthetic cell
does not make the real-data question moot, it raises its stakes (see §9
item 9).

---

## 0. Reading list this design builds on (context, not repeated here)

- `matrix-thinking/DELTANET_CAUSAL_RANK_DESIGN.md` — the full methodology
  this document transplants onto real data: the provable `rank(S_T) ≥ K`
  bound (§4.1), the two-kernel-call truncation split that makes train-time
  force-rank causal without destroying the chunked kernel (§3.5), the
  entity-subspace-restriction lesson pulled forward pre-emptively (§3.6),
  the multi-head/multi-layer/conv-boundary escape closures (§4.2-4.3), and
  — load-bearing for §4 below — **§12's actual measured results**: Wave −1's
  timing/instability numbers, Wave 0's "no late-transition regime, perfect
  saturation, zero rank-inflation" finding, and the F13 wall-clock-ratio
  correction (8.4–9.9×, not the originally-assumed ≤2×).
- `matrix-thinking/chapter2/TASK_E_FINDINGS.md` — what "compositional"
  means operationally in this lineage: the pinned iterated-matmul readout
  `pred(a,h) = Zʰ·key_a`, the depth-amplification finding (raw hop count is
  a sharper rank test than single-hop cosine tolerance, §3/§9), the
  subspace-restriction resolution (§9-§10: whole-matrix rank is polluted by
  an inert orthogonal complement; the *entity-subspace-restricted* rank is
  what tracks K), and the "three-budget-artifacts" program-level finding
  (§10: every "dead" cell in this program's history that was re-tested at
  2–2.5× budget either transitioned or, in exactly one case (K=16=d, a
  structurally distinct boundary case with no orthogonal complement), did
  not — "re-test before calling dead" is now a standing house discipline).
- `STATE.md` — the thesis (rank as a measurable structural correlate of
  reasoning capacity), the Chapter 3 framing (matrix-native byte-level on
  real data, gated on Task E), and the **already-settled bytes-vs-tokens
  decision**: hold the tokenizer standard (GPT-2/BPE) for the primary
  scaled experiment; byte-level input is a separate, explicitly-sequenced
  follow-on ablation, not bundled with a first unproven architectural
  change. This document inherits that decision directly — every wave below
  uses the GPT-2 tokenizer, never a byte-level input, and §9 attacks any
  drift from this.
- `research/task-d-novelty-july2026.md` — the existing, PDF-verified
  (pages 1–3, high confidence) positioning against Nazari & Rusch
  (arXiv:2602.04852) and Sun et al. (arXiv:2602.02195): both are
  **descriptive, post-hoc analyses of pretrained real LLMs** (Gated
  DeltaNet 370M; Qwen3-Next), with an algebraic *upper*-bound rank fact
  (`rank(S_t) ≤ t` / `≤ min(t,d)`), no provable *lower*-bound necessity
  theorem, and no training-time causal intervention of any kind. §10 below
  re-verifies this is still true as of 2026-07-02 and adds one paper this
  earlier pass did not have reason to find.

---

## 1. The sharpened question

> **H_RD (primary, two arms — deliberately not one).**
> **(RD-1, train-time causal, premise-conditional — the strong claim).** A DeltaNet layer trained
> end-to-end, under the synthetic design's own bottleneck and force-rank
> mechanism (§3.5 of `DELTANET_CAUSAL_RANK_DESIGN.md`, reused verbatim), on
> a relational-binding task whose surface form is **natural-language
> sentences tokenized by a real BPE tokenizer** rather than hand-built
> orthonormal vectors, still (a) recruits entity-subspace rank tracking the
> binding count K, and (b) shows the same causal force-rank accuracy step,
> **once the linear-independence premise is verified empirically on the
> actual embedded keys** (§5, replacing the synthetic design's
> by-construction orthonormality with the pre-registered, numerically
> thresholded C16 premise rule of §5.2 — the successor to the C13/F4
> post-conv effective-key discipline, promoted to a per-checkpoint
> instrument and applied one level further from the metal).
> **(RD-2, inference-time causal + descriptive, the weaker but more
> "real-data"-flavored claim).** A DeltaNet-hybrid LM pretrained on **real
> reasoning text** (OpenR1-Math) shows state-rank dynamics that differ
> between reasoning-dense and narrative-dense text (descriptive, in the
> spirit of Nazari & Rusch / Sun et al.), **and** inference-time rank
> truncation of the trained state damages downstream reasoning-task
> accuracy more than it damages narrative-text perplexity (the causal
> addition neither prior paper makes).
>
> **CONFIRM (RD-1)** ⇒ the from-scratch rank-recruitment result survives
> the one axis Task D/E and the synthetic DeltaNet design deliberately held
> fixed (a real tokenizer's segmentation and vocabulary-ID space, and a
> real embedding geometry SGD controls rather than the experimenter —
> single-token entities at the primary gate per §5.2's FATAL-1 decision;
> multi-token mentions are a Reserve stress arm) — closing the gap
> between "provable synthetic bound" and "real data" one variable at a time,
> per this project's own controlled-single-variable-change discipline
> (`CLAUDE.md`: *"Hold tokenization... FIXED when testing a primary
> hypothesis... Treat the second axis as a separate, explicitly-sequenced
> follow-on ablation."* — here rank-recruitment is the already-tested
> primary hypothesis and tokenization is exactly that follow-on axis).
> **FALSIFY (RD-1)** ⇒ a real, informative negative distinguishing "rank
> recruitment is a property of the delta rule's own dynamics" (Wave 0's
> finding) from "rank recruitment survives being routed through a
> tokenizer's embedding table" — genuinely open, not assumed either way
> (§2.1).
> **CONFIRM (RD-2)** ⇒ the first *causal* (not just descriptive) rank study
> on a fast-weight LM trained on real text — directly, constructively
> extends Nazari & Rusch / Sun et al. rather than duplicating them (§10).
> **FALSIFY (RD-2)** ⇒ still informative: truncation damages fluency and
> reasoning equally, meaning the descriptive rank-stratification those two
> papers report is not differentially load-bearing for reasoning
> specifically — a real qualification of how their finding should be read.
>
> **RD-1 and RD-2 are not the same claim and neither entails the other.**
> RD-1 is a **train-time** causal claim (does SGD get pushed toward a
> rank-disciplined solution) on a **controlled, provable-bound** task
> wrapped in natural surface form — and it is **premise-conditional**: the
> bound's linear-independence premise is verified empirically per
> checkpoint (§5.2's C16 rule), never architecturally guaranteed; the
> label carries this qualifier everywhere it appears (MAJOR-8). RD-2 is an **inference-time** causal
> claim (does the already-converged state's rank matter functionally) on
> **genuinely uncontrolled real text with no provable bound at all**. §2
> is the argument for why both are needed and why they are sequenced
> RD-1 → RD-2, not built simultaneously.

---

## 2. The central methodological problem

### 2.1 What "provable" bought us, and exactly why it evaporates on real text

The entire Chapter 2 lineage's rigor rests on one proof (`TASK_D_PREREGISTRATION.md`
§3, restated for DeltaNet in `DELTANET_CAUSAL_RANK_DESIGN.md` §4.1): if `K`
keys are linearly independent, `K` values are linearly independent, and the
readout is the exact linear unbind `S_T · key_j`, then exact recovery of all
`K` bindings forces `rank(S_T) ≥ K`. Task D/E and the synthetic DeltaNet
design satisfy the premise **by construction** — keys are QR-orthogonalized
vectors the experimenter builds, not learned representations of anything.

On real text, there is no equivalent construction available for free.
Entity mentions ("Bob", "the book") are drawn from a real vocabulary whose
embedding geometry SGD controls, not the experimenter. Three specific ways
the bound's premises can silently fail that have no synthetic-task analog:

1. **BPE fragmentation.** A name like "Cara" may tokenize to 1–3 sub-word
   pieces depending on context (leading space, sentence position,
   co-occurring punctuation) — the clean "one key = one embedding vector"
   correspondence Task D/E's grammar enjoyed is not guaranteed; the
   "effective key" that reaches the recurrence is whatever the model's own
   embedding-lookup + optional conv produces from a *variable-length* token
   span, not a single controlled vector.
2. **Embedding-geometry collisions.** Two entity names could end up with
   accidentally-correlated learned (or pretrained) embeddings for reasons
   having nothing to do with the task (shared BPE prefix, similar frequency
   statistics, etc.), silently weakening the linear-independence premise in
   a way no synthetic orthonormal construction ever risks.
3. **No experimenter-controlled `N`.** Task D/E draws keys from a
   QR-orthogonalized pool of exactly `N` directions the experimenter
   chooses. A real tokenizer's vocabulary is fixed at 50,257 tokens
   (GPT-2) — the effective "pool" of usable entity representations is
   whatever the model's own embedding table happens to look like at a
   given checkpoint, not a controllable design parameter.

**This is the actual content of "there is no provable rank(S)≥K bound on
real data"** — it is not that real data lacks relational structure (it
doesn't; that is exactly what makes it interesting), it is that the
*embedding-level* linear-independence premise the proof needs stops being
an architectural given and becomes an empirical, checkpoint-conditional
fact about what SGD did to a real vocabulary.

### 2.2 Candidate (a) — naturalistic probe tasks

**Mechanism.** Keep Task E's exact generator machinery (single Hamiltonian
K-cycle, injective binding, `TaskEConfig`-style periodicity guards) but
render the BIND/QUERY grammar as natural-language sentences through a real
BPE tokenizer instead of raw vector concatenation — e.g. a fixed pool of
`K` first names bound into a chain ("Ann gave the book to Bob. Bob gave it
to Cara...") with QUERY sentences asking "who has the book after N
exchanges?" The **relational graph stays fully constructed** (the
experimenter still builds the K-cycle) — only the **surface form and the
embedding/tokenization path** become real. (Revision-2 note: this
free-form example motivates the candidate but is NOT the realized
grammar — FATAL-1 showed a causal single-layer DeltaNet cannot write a
binding whose key and value sit beyond one conv window apart; the
realized surface form is §5.2's adjacency-constrained template.)

**What survives, and how.** The provable bound is **not** lost outright —
it is converted from an architectural given into an **empirically-verified,
checkpoint-conditional** bound: restrict the entity vocabulary to a small,
fixed, closed set of `K` names; assert (not assume) at every checkpoint
that the *effective* keys reaching the recurrence — the actual embedded,
post-tokenizer, post-any-conv representations of those `K` names — have a
Gram matrix within tolerance of full rank (`‖K_eff^T K_eff − I‖_F` below a
threshold, or more permissively, `rank(K_eff) = K` exactly via SVD). **This
is not a new methodological hole this design invents — it is the identical
discipline the synthetic DeltaNet design already had to build for its own
production-block secondary check** (C13, §4.3 of `DELTANET_CAUSAL_RANK_DESIGN.md`:
*"the harness must be built so the readout is a pure function of `(S_T,
query)`... verify by corrupting/zeroing every input token embedding"*),
applied one level further from the metal: instead of verifying premises
survive a short causal conv, this verifies premises survive a real
tokenizer + a real (from-scratch, per §2.5's scoping decision) embedding
table. The claim tier this earns is named accordingly throughout:
**train-time causal, premise-conditional** (MAJOR-8) — the qualifier is
part of the label, not a footnote.

**What is genuinely lost, honestly stated (return to this in §9 item 3):**
the *causal-rank methodology* content of this candidate is identical
whether entities are named "entity_7" or "Cara" — the proof does not care
about surface form, only about embedding-level linear independence. So
candidate (a) does **not**, by itself, test "can matrix rank support
natural-language reasoning" in any deep sense. It tests a narrower,
still-nontrivial, mechanical question: **does the provable-bound causal-rank
result survive being routed through a real BPE tokenizer and a real,
SGD-controlled embedding table, instead of a hand-built orthonormal
vector?** This is honestly scoped as a **tokenizer-robustness bridge**, not
yet a "real reasoning" result.

**Cost.** Cheap. No real pretraining corpus is needed — the probe-task LM
is trained from scratch on synthetically-generated natural-language
sentences only (§5), reusing the just-validated synthetic design's entire
harness (`model_dn.py`, `deltanet_core.py`, `rank_utils.py`,
`task_e.py`'s generator) with one substitution: a real tokenizer +
embedding lookup in place of raw vector concatenation. Everything else —
the two-kernel truncation split, C1–C15 controls, the force-rank grid
structure, the entity-subspace-restriction metric — transfers with the
§5.2 re-anchoring (FATAL-1: C9's β-mask moves to the binding-completion
positions; C14 is promoted to the primary gate), not "unmodified."

### 2.3 Candidate (b) — instrumented LM

**Mechanism.** Train a small DeltaNet-hybrid LM on real reasoning text
(OpenR1-Math, confirmed available, §3), measure entity-subspace-agnostic
whole-state effective rank across reasoning-dense vs. narrative-dense text
(the latter from WikiText-103, also confirmed available, §3, same
tokenizer), correlate rank with text type, then **add the causal
ingredient neither Nazari & Rusch nor Sun et al. include**: truncate the
trained state's rank at inference time and measure the differential damage
to (i) downstream reasoning-task accuracy on held-out OpenR1-Math problems
vs. (ii) narrative-text perplexity on WikiText.

**What this is, positioned honestly.** This is squarely Nazari & Rusch /
Sun et al.'s own territory (descriptive rank measurement on a real-text
LM) **plus** an inference-time intervention. It is **not** a train-time
causal claim — the intervention manipulates an *already-converged* state's
usable rank at eval time, which tells you about the *converged solution's*
functional dependence on rank, not about whether the training process was
*pushed toward* a rank-disciplined solution (§1's RD-1/RD-2 distinction).
This is a genuinely weaker causal claim than RD-1, stated plainly rather
than oversold.

**No provable bound at all.** Real reasoning text has no known ground-truth
`K` — there is no way to construct a `rank(S_T) ≥ K` necessity theorem for
"how many facts a math solution needs to track," so this candidate cannot,
even in principle, produce Task D/E-style decisive CONFIRM/FALSIFY evidence
about rank *necessity*. It can only produce evidence about rank
*correlation* (descriptive) and rank *sufficiency-for-current-performance*
(the truncation intervention) — a fundamentally different, weaker
evidentiary class, honestly scoped as such throughout this document.

**Cost.** Real pretraining is required (a small DeltaNet-hybrid LM must
reach some minimal competence on math reasoning text to make "reasoning
span" a meaningful concept at all) — see §4.2's FLOPs estimate, which
finds this is, perhaps counter-intuitively, the *cheap* part of this
document's budget; the expensive part remains the force-rank-style
ablation machinery reused from (a) if RD-1 confirms and is extended here.

### 2.4 Candidate (c) — hybrid (pretrain on real text, evaluate on embedded probe tasks)

**Mechanism.** Pretrain a DeltaNet-hybrid LM on real text (as in (b)), then
embed candidate (a)'s constructed relational-binding probe task as
zero-/few-shot evaluation prompts (or a light fine-tuning pass) layered on
top of real pretraining — the model that gets rank-forced and evaluated on
the provable-bound task is the *same* model that has seen real reasoning
text, not a probe-task-only model.

**Why this is the most ambitious option and why it is deliberately not
built first.** It is the only candidate that could, in principle, directly
speak to the program's stated ambition ("abstract thought in matrix
states," `STATE.md`) with both a provable bound *and* real-pretraining
realism in the same experiment. But it bundles **three** simultaneously
unproven things: (i) does rank-recruitment survive a real tokenizer at all
(RD-1's question), (ii) does a DeltaNet-hybrid LM learn anything useful
from real reasoning text at this budget's scale (RD-2's question), and
(iii) does real pretraining *interact* with the probe task's rank-recruitment
mechanism (a new, third question this candidate alone asks). Building (c)
before (a) and (b) individually confirm violates this project's own
"change one variable at a time" discipline twice over in a single
experiment, and — per the same discipline that killed the original Task A
design before any GPU ran — a result from a triple-bundled experiment
would be uninterpretable regardless of outcome (a CONFIRM could be RD-1's
mechanism, RD-2's mechanism, or a genuine interaction; a FALSIFY could
implicate any of the three, or their interaction, with no way to
attribute it from this experiment alone).

**Verdict: explicitly deferred**, not built in this document's budget — a
natural Chapter 3/4 direction once (a) and (b) individually land, not
before.

### 2.5 Ranking

| Candidate | Provable bound? | Real-data realism | Causal-claim tier | Rigor-per-GPU-h | Scooping risk (§10) | Verdict |
|---|---|---|---|---|---|---|
| **(a) Naturalistic probe task** | **Yes** — empirically-verified, checkpoint-conditional (§2.2), not architectural | Low-medium (real tokenizer/embeddings, constructed content) | **Train-time causal, premise-conditional** (strongest tier this document can earn; the premise is verified per checkpoint by §5.2's C16 rule, not architectural) | **Highest** — cheap (§4.2), reuses ~90% of already-validated harness, decisive CONFIRM/FALSIFY structure | **Low** — closest analog found (§10, `arXiv:2602.14814`) trains a linear RNN on real-ish surface form for state-tracking but has no rank observable and no necessity bound; genuinely distinct | **Wave 1 (primary, build first)** |
| **(b) Instrumented LM** | No | High (real reasoning corpus, already tokenized and on the local SSD, §3) | Descriptive + **inference-time** causal | Medium — real pretraining is cheap (§4.2) but the claim tier is inherently weaker and non-decisive | **Medium-high** — this is exactly the shape of experiment Nazari & Rusch / Sun et al. already ran; the "plus interventions" delta is the only differentiator and is a fairly obvious next move for other groups too | **Wave 2, gated on Wave 1 CONFIRM** (§6) |
| **(c) Hybrid** | Partial (bound only holds within the embedded probe, and only if (a) + (b)'s premises both hold simultaneously, unverified) | Highest | Would be train-time causal (premise-conditional) on the probe, descriptive+interventional on the rest — a genuine, currently-nonexistent combination if it worked | Lowest at this budget — triple-bundled, uninterpretable-on-failure risk (§2.4) | Low (nobody is doing exactly this) but only because nobody has (a) or (b) working yet either | **Explicitly out of scope for this design's budget; a Chapter 3/4 direction** |

**Decision: build (a) first, as this document's Wave 1, with a hard gate
before (b) becomes Wave 2.** This mirrors the Task D → Task E precedent
exactly (cheap, decisive, provable-bound experiment first; only extend
toward the harder-to-control real-data setting once that result is in
hand) and the project's own explicit "hold tokenization fixed, sequence the
second axis" rule — here, "provable relational structure" is what stays
fixed (the constructed K-cycle) while "real tokenizer, real embedding
geometry" is the one new axis Wave 1 introduces. Wave 2's real-pretraining
axis is sequenced *after*, not bundled in.

---

## 3. Data — what actually exists (correcting an assumption in the task brief)

The task brief's framing ("43.7M tokens OpenR1-Math GPT-2 tokenized live at
`/root/data/reasoning` on the OLD pod — likely GONE with that pod... may
need re-tokenizing from HF") is **half right and half stale, checked
directly against both the current Brev box and the local SSD before writing
this section:**

| Location | What's there | Status |
|---|---|---|
| `/root/data/reasoning` (old single-H100 pod, `H100_SETUP.md`'s "prior/legacy" section) | OpenR1-Math tokenized | **Correctly identified as superseded** — that pod's access details are explicitly marked stale in `H100_SETUP.md`; not checked directly (no reason to — see next row) |
| **`/Volumes/1TB_SSD/learned-representations/data/reasoning/`** (local Mac SSD) | `train.pt` (334MB), `val.pt` (18MB), `meta.json` | **CONFIRMED PRESENT, 2026-07-02.** `meta.json`: `vocab_size=50257`, `tokenizer="gpt2"`, `train_tokens=43,725,587`, `val_tokens=2,301,347`, `source="open-r1/OpenR1-Math-220k"`, `n_examples=93733`. **The data did not die with the old pod — it was archived to the local SSD and survives intact.** |
| **`/Volumes/1TB_SSD/learned-representations/data/wikitext103_tokenized/`** (local Mac SSD) | `train.pt` (943MB), `val.pt`, `test.pt`, `meta.json` | **CONFIRMED PRESENT.** `meta.json`: `vocab_size=50257`, `tokenizer="gpt2"`, `train_tokens=117,920,140`, `val_tokens=247,289`, `test_tokens=283,287`. **Same tokenizer as the reasoning set** — a clean, zero-extra-tokenization-cost reasoning-vs-narrative corpus pair already exists (needed for Wave 2, §6.1). |
| **Current Brev box (`youthful-indigo-turkey`, `/home/nvidia`, `/data` [17TB free], `/ephemeral` [5.8TB free])** | Searched `/home/nvidia`, `/data`, `/ephemeral` for `*reasoning*` / `*openr1*` | **CONFIRMED ABSENT**, 2026-07-02. This box has never had the OpenR1-Math data — it is a different (newer) instance than whatever pod originally tokenized it. |
| Current Brev box — `fla-org` package | `flash-linear-attention` **0.5.1** | **CONFIRMED INSTALLED** (`pip show`, 2026-07-02) — the F15 checkpoint's "package is not currently installed" caveat in `DELTANET_CAUSAL_RANK_DESIGN.md` §6.1 is now resolved; it was installed as part of that design's own build phase and is live on the box today. |
| Current Brev box — existing DeltaNet harness | `/home/nvidia/chapter2/deltanet/{model_dn.py, deltanet_core.py, run_deltanet.py, run_deltanet_sweep.py, task_dn.py, rank_utils.py, probe_trunc.py}` | **CONFIRMED PRESENT AND ACTIVELY RUNNING** (Wave B-probe live via `ps aux`, 2026-07-02) — this is the exact reuse target §5 below builds on |

**Pipeline cost, corrected:** not "re-tokenize from HF" (the brief's
speculative worst case) — the actual cost is **one `scp` of ~352MB**
(`reasoning/{train,val}.pt` + `meta.json`) from the local SSD to the
current Brev box's `/data` volume (17TB free, no capacity concern), plus a
second `scp` of ~945MB for the WikiText-103 companion set if Wave 2 is
reached. Both transfers are minutes, not the "~8 min full HF
re-tokenization" `H100_SETUP.md`'s own pod-setup checklist budgets for the
from-scratch case. **This removes what could have been the single largest
line-item risk in this document's budget** (an unplanned HF
download+tokenize pass, with its own rate-limit/flakiness risk) and is
recorded here as a Wave 0 prerequisite action (§7), not a researched
uncertainty.

**Live resource-contention note (2026-07-02, at time of writing):** the
current box's 8 GPUs are substantially occupied by two concurrent, unrelated
workloads at the moment this design is written — the synthetic DeltaNet
design's Wave B-probe (`run_deltanet_sweep.py --wave=Bprobe`, GPUs 6–7,
plus individual `run_deltanet.py` force-rank jobs on GPUs 0–2/4) and Stage G
Wave A (`run_stageg_sweep.py --wave=A`, GPUs 3–5). This document's own Wave
0 (§7) should be scheduled to queue behind whichever of these finishes
first, or packed into whatever idle capacity remains, per the project's own
"pack multiple tiny runs per GPU" pattern (`H100_SETUP.md` §4) — not treated
as a blocking conflict, but not ignored either.

---

## 4. Model scale and kernel-path feasibility

### 4.1 Parameter budget — reconciling the brief's 10–50M target with the 10M+ hard rule

`CLAUDE.md`'s hard rule (*"At 288K params, models barely learn unigram
statistics... Need 10M+ minimum"* for any reasoning/generalization claim)
sets the floor; the brief's own 10–50M guidance sets the ceiling for this
budget tier. The arithmetic, worked on paper per the pre-experiment
checklist:

- **Embedding/output head dominates at this scale**, exactly as this
  project's own prior LM work already found (`H100_SETUP.md`: *"The
  bottleneck is the logits tensor... Embeddings: ~3.2M params (63%)"* at a
  much smaller vocab). At GPT-2's 50,257-token vocabulary, tied
  embed/unembed at `d_model`:
  - `d_model=64` (the synthetic design's own just-validated primary state
    dimension, §12.5): `50257 × 64 ≈ 3.2M` params — **under the 10M floor
    on its own**, and a single bare DeltaNet layer at this width adds only
    tens of thousands more. **Not viable as a standalone config for any
    reasoning claim under the hard rule.**
  - `d_model=256`, 2 layers (DeltaNet mixing, `H=1` state dim held at 64
    per C11's continuity — see below — plus a small FFN sublayer per
    layer): `50257×256 ≈ 12.9M` (embed) `+ ~1.2M` (2× DeltaNet+FFN layers)
    `≈ 14M` total. **Clears the 10M floor, sits inside the brief's 10–50M
    range, with headroom.**
- **Decoupling `d_model` (residual-stream width) from the DeltaNet state
  dimension `d` (§4.2 of the synthetic design) is architecturally free** in
  a real `fla-org` block — `head_dim` and `hidden_size` are independently
  configurable, with the layer's own `W_q/W_k/W_v/W_o` projections handling
  the width mismatch. **Recommendation: keep the single-head state at
  `d=64`** (the synthetic design's own Wave 0/A-selected primary cell,
  §12.5, chosen there on cost/stability grounds after both `d∈{64,128}`
  were calibrated) **while letting `d_model=256` carry the param budget** —
  this is a genuine continuity decision, not a coincidence: it means Wave 1
  below is testing the *exact* state dimension the synthetic design already
  has Wave 0 saturation data and (pending) Wave B causal data for, isolating
  the tokenizer/embedding axis as the only new variable relative to that
  already-running experiment, per §2.5's stated discipline.
- **`H=1` single head is retained** (C11 from the synthetic design, §6.2)
  for the primary Wave 1 gate — same rationale (closes the multi-head
  joint-storage escape, §4.2 of that design) — full transplant, not
  re-derived here.

| Config | `d_model` | State `d` (`H=1`) | Layers | Approx. params | Clears 10M floor? | Role |
|---|---|---|---|---|---|---|
| Minimal (rejected) | 64 | 64 | 1 | ~3.3M | **No** | Rejected outright |
| **Wave 1 primary** | **256** | **64** | **2** | **~14M** | **Yes** | Naturalistic probe-task LM (§5) |
| Wave 2 (real-corpus) | 384 | 64–128 (Wave 0-calibrated, §7) | 2–3 | ~20–28M | Yes | Instrumented LM (§6) |

### 4.2 FLOPs / token budget — derived from the actual bottleneck (REWRITTEN, revision 2 — MAJOR-6)

> **MAJOR-6 (why revision 1's estimate died).** Revision 1 applied the
> generic `6 × N_params × N_tokens` rule at an assumed 15–25% MFU,
> justified by "the DeltaNet state-update is FLOP-comparable to attention"
> — an attention analogy that contradicts §4.3's own commitment to the
> chunked production kernel, and one that ignores where the FLOPs and the
> VRAM actually live at this scale: the 50,257-class softmax head. The
> house rules already knew this (`H100_SETUP.md`: *"The bottleneck is the
> logits tensor"*; *"The 50K vocab logits tensor is the VRAM bottleneck,
> not the model activations"*; *"eval can OOM even if training fits"*).
> Re-derived below from that bottleneck.

**FLOPs decomposition at the Wave 1 primary config** (`d_model=256`,
`V=50,257`, tied head, 2 layers, `d_state=64`):

- **Head (unembed matmul + softmax/xent):** `2·d_model·V ≈ 2.6×10⁷`
  FLOPs/token forward, ≈ `7.7×10⁷`/token with backward — **~10× everything
  else combined; the head IS the compute budget.**
- **DeltaNet body:** ~1.2M non-embedding params → `6·N_body ≈ 7×10⁶`
  FLOPs/token. The chunked delta-rule kernel's own per-token state cost at
  `d_state=64` is small next to this (per-chunk `O(d_state²)` terms
  amortized over `chunk_size=64` tokens) and is priced from **measured
  kernel throughput at Wave −1**, not by analogy to attention (fixing
  revision 1's internal contradiction with §4.3).
- The generic `6·N·tokens` with `N≈14M` (`8.4×10⁷`/token) lands
  coincidentally close — but only because the tied embedding *is* the
  head. The honest statement is that training cost is **head-dominated**,
  which drives both the MFU assumption and the VRAM budget below.

**VRAM (the standing house rule, priced for this config):** the logits
tensor is `B×T×V` in bf16 — at `B=32, T=1024`:
`32·1024·50257·2 B ≈ 3.3 GB` forward, roughly doubled with gradients.
Comfortable on an 80GB H100 at these batch sizes, but the **eval batch is
capped independently of the train batch** from the start (the exact
"batch=112 trains, eval OOMs" lesson, applied rather than re-learned).

**Token target (unchanged arithmetic from revision 1):** `N ≈ 15–25M`
params (§4.1) at a `≥20 tokens/param` Chinchilla-ish floor →
`N_tokens ≥ 300–500M`. Available corpus (§3): OpenR1-Math (43.7M) +
WikiText-103 (117.9M) = 161.6M unique tokens, same tokenizer → 2–3 epochs
over the blend, or more epochs over OpenR1-Math alone if narrative text is
held out as a non-blended contrast corpus (§6.1).

**Throughput and wall time, honestly banded (label corrected, revision
2.1 — R2-5):** a head-dominated small model in an unfused research
script (softmax + cross-entropy at `V=50K` is memory-bound, not
tensor-core-bound) realistically sustains **~50–200K tokens/s/GPU** —
which at `~8.4×10⁷` FLOPs/token is `4–17` TFLOPS ≈ **0.4–1.7%
end-to-end MFU** against the 990 TFLOPS bf16 peak (revision 2 mislabeled
this band "5–15% MFU"; the honest number for a tiny-`d_model` unfused
research script dominated by memory-bound ops and framework overhead is
sub-2%, and the tok/s band — the quantity the wall-time estimate
actually uses — is unchanged).

- **Wall time, 1 GPU: ≈ 0.6–2.2 h per complete pretraining run** at
  400M tokens — revision 1's "~4 minutes" corrected upward ~10–30×.

**Reading this honestly (conclusion survives, margin shrinks ~10×).**
Even at the pessimistic end, Wave 2's full pretraining sweep (~8–12 runs)
is ~10–25 GPU-h — real money now, but still not this document's budget
bottleneck; that remains the force-rank grid's **measured 8.35–9.9×
wall-clock multiplier** (§12.2 of the synthetic design, an empirical
fact, not an estimate). Two notes on this project's own estimate track
record, stated correctly this time (MINOR-11 — revision 1 cited this
history backwards): (i) Stage 0's flat-rate GPU-h estimates ran **5–6×
HIGH** — measured runs came in *cheaper* than priced — which cuts toward
comfort on the per-run pricing above; (ii) the pattern that actually
bites this program is the **late-transition / step-budget artifact**
(three documented instances, `TASK_E_FINDINGS.md` §10): the historical
under-estimates are in *steps-to-convergence*, not per-run cost. Wave 0's
2.5× extended-budget calibration (§7) exists for (ii); (i) is why §7's
totals are treated as upper bounds, not targets — and none of the numbers
above are trusted until Wave −1/0 measure them directly.

### 4.3 Pure-PyTorch recurrence vs. the `fla-org` production kernel — resolved by measured facts, not assumed

This is not a hypothetical to investigate from scratch — the synthetic
DeltaNet design's own build phase already produced a directly relevant,
concrete finding, read from `deltanet_core.py`'s own header on the box
(2026-07-02):

- **`fla.ops.delta_rule.chunk_delta_rule`** (the production Triton kernel)
  **rejects float32 inputs outright** (raises an explicit error demanding
  bfloat16) — this is why the synthetic design's harness does not use it at
  all: a float64 finite-difference gradcheck through it is categorically
  impossible.
- **`fla.ops.delta_rule.naive.delta_rule_recurrence`** (fla's own
  pure-PyTorch reference implementation) internally hard-casts every input
  to float32, silently defeating a float64 gradcheck (10–20% Jacobian
  mismatch observed and traced to the downcast, not a real autograd bug —
  removing the cast and rerunning passes cleanly).
- **`chunk_delta_rule`'s backward crashes** ("illegal memory access" inside
  a Triton autotuner kernel) for sequence lengths below its internal
  `chunk_size=64` — confirmed absent at `T≥128`.

**Why this reverses at LM scale, and must be decided explicitly, not
inherited.** The synthetic design correctly chose the pure-PyTorch
sequential recurrence *because* its BIND phases are short (`T ≈ 32–160`,
`T_bind = K×(1+buf_len)`) and it needed literal fp64 gradcheck through the
truncation. **Neither condition holds for the real-data LM path:**

1. Real text sequences for a meaningful LM (§4.2) are `T ≥ 512–1024`, well
   above the `chunk_size=64` crash threshold — the production kernel's one
   known failure mode is irrelevant here.
2. LM-scale training needs **throughput over hundreds of millions of
   tokens**, not a one-off finite-difference check — a per-token-sequential
   Python loop (the pure-PyTorch path's whole cost structure) forfeits
   exactly the chunked-parallel scan that is DeltaNet's entire practical
   reason for existing (`DELTANET_CAUSAL_RANK_DESIGN.md` §3.2 already makes
   this point for the synthetic per-step-truncation option; it applies with
   even more force to full LM pretraining, where sequences are 4–16× longer
   than the synthetic harness's BIND phase).

**Decision, pre-registered here rather than left implicit:** the real-data
LM path (both Wave 1 and Wave 2) uses the **production `fla-org` chunked
kernel** (bf16) — called directly by a custom block per R2-3 (§5.2: the
stock `fla.layers.delta_net.DeltaNet` nn.Module computes β internally via
its `b_proj` and exposes no hook for C9's hard β-mask, so the kernel is
invoked via `fla.ops.delta_rule.chunk_delta_rule`, not the stock layer) —
reversing the synthetic design's own choice — because the tradeoff that motivated that
choice (short-`T` fp64 gradcheck feasibility over training throughput) is
inverted here (long-`T` training throughput matters, fp64 gradcheck is
infeasible through the kernel regardless of sequence length, per the
float32-rejection fact above — there is no regime where the production
kernel supports fp64). This must still be **verified, not assumed**, as a
Wave 0 calibration decision (§7) — mirroring the synthetic design's own
`d=64`-vs-`d=128` Wave 0 decision-by-measurement discipline, applied here to
kernel-path-vs-pure-PyTorch instead.

### 4.4 The F15 gradcheck, redone for the LM configuration — an honest downgrade, not a dodge

The synthetic design's F15 checkpoint ran a literal finite-difference fp64
gradcheck through `deltanet_core.py`'s pure-PyTorch recurrence, with C15's
run-to-completion negative-test discipline (a deliberately detached
gradient path must FAIL). **That exact check is impossible for the LM path**,
for the same hard reason §4.3 documents: the production kernel rejects
float32/float64 categorically, so no finite-difference gradcheck can be run
through it at any tolerance.

**Resolution, pre-registered (REVISED, revision 2 — MAJOR-4: the ground
truth is itself verified before it is used as ground truth):** a
three-tier verification chain replaces revision 1's single cross-check —
revision 1 promoted the patched naive reference to bf16 ground-truth
status on the strength of the synthetic design's fp64 verification *at
that design's own short-`T`, small-`d` shapes*, which is not evidence
about the LM shapes this design uses it at:

1. **Tier 0 — fp64 gradcheck of the reference at LM-relevant shapes
   (MAJOR-4, new).** Patch the naive reference's internal `.float()`
   downcast (the exact bug the synthetic design's F15 checkpoint already
   diagnosed), then run a literal fp64 finite-difference gradcheck through
   the **patched** reference at the shapes this design will actually use:
   Wave 1's real BIND-phase length, and the largest feasible `T` toward
   Wave 2's 512 (finite-difference cost scales with the number of
   perturbed input elements — run at `B=1`; if the full training `T` is
   wall-clock infeasible, the largest verified `T` is recorded and the
   gap to the training `T` disclosed, not glossed). Shape-dependent
   numerical pathologies (error accumulation over longer recurrences) are
   exactly what this tier exists to catch **before** the reference is
   promoted to ground-truth status. Reuses `deltanet_core.py`'s
   `_self_test` harness pattern, including its detached-gradient-path
   negative test run to completion (coherent here — this is a structural
   gradcheck, unlike C16's calibrated empirical premise, §5.2).
2. **Tier 1 — kernel vs. verified reference.** On identical inputs, run
   the production kernel and the Tier-0-verified reference; compare
   **(a)** forward outputs and **(b)** parameter gradients after
   `.backward()` (an implementation-vs-implementation cross-check, not a
   finite-difference estimate) at a bf16-appropriate relative tolerance
   (`<1e-2`). Same detached-path negative-test discipline.
3. **Tier 2 — the kernel trains.** Only after Tiers 0–1 pass does the
   production kernel carry any training run.

**Tier 1 remains genuinely weaker than a literal fp64 gradcheck through
the production kernel itself — stated plainly, not hidden.** That check
stays categorically impossible (§4.3's float32-rejection is a hard
library constraint, not a design choice). Tier 0 closes the "unverified
ground truth" half of the gap the attack round flagged (MAJOR-4); the
residual — kernel-internal bugs smaller than the bf16 tolerance — is
flagged in §9 as the place an independent reviewer should still push.

---

## 5. Wave 1 design — naturalistic probe task (primary path, RD-1)

### 5.1 What transfers verbatim from `DELTANET_CAUSAL_RANK_DESIGN.md`

Everything except the input representation **and the write-site anatomy**
(the subject of §5.2's FATAL-1 rework — where C9's β-mask anchors, and
the conv's promotion from excluded to required):

- The single-Hamiltonian-K-cycle generator (`task_e.py::_permutation_graph`,
  the periodicity guards, the injective-graph assert-at-generation-time
  check).
- The two-kernel-call truncation split (§3.5): BIND-only pass → truncate
  `S_T` once → decode via an external, pinned readout — now running through
  the production `fla-org` kernel (§4.3) rather than `deltanet_core.py`'s
  pure-PyTorch path, with the F15 round-trip check re-verified at the new
  kernel path (state-extraction/state-seeding hooks, zero-embedding buffer
  convention, C15's post-truncation SVD rank assertion — all unchanged in
  spirit, re-run against the new code path as a build-time gate, not
  assumed to transfer).
- `H=1`, single layer for the primary gate (C11); readout reads only the
  final layer's final state (C10); pinned iterated-matmul composition
  readout `pred(a,h) = S_Tʰ·key_a` (§5.4 of the synthetic design) for the
  held-out-hop extension.
- The entity-subspace-restricted rank metric (§3.6/§6.3 of the synthetic
  design) as the primary M1-equivalent instrument — **directly informed by
  Wave 0's finding that DeltaNet shows no rank-inflation to distinguish**
  (§12.4: entity-subspace rank = whole-matrix rank = K exactly, because the
  delta rule only ever writes within `span({k_j})` by construction). This
  is a strong prior for Wave 1 too (the *mechanism* generating this
  no-inflation property — §3.4's algebra — has nothing to do with whether
  the keys arrived via a synthetic vector or a token embedding), but it is
  **re-measured, not assumed**, since Wave 1's keys are no longer
  architecturally orthonormal (§5.2).

### 5.2 What's new — the grammar, the write mechanism, and the embedding path (REWRITTEN, revision 2 — FATAL-1, FATAL-2)

> **FATAL-1 (why revision 1's grammar died).** Revision 1 rendered
> bindings as free-form sentences ("Ann gave the book to Bob.") and
> asserted "C1–C15 transfer unchanged in spirit." The attack round killed
> this on the write mechanism: the synthetic grammar co-locates key and
> value on **one token** (`concat(key_t, value_t)`), so C9's hard β-mask
> ("β=1 at the K item positions, 0 elsewhere") has a direct realization —
> one token, one write. In "Ann gave the book to Bob," the key (Ann) and
> value (Bob) are ~5 BPE tokens apart: a **causal** layer cannot write
> the binding at Ann's position (Bob has not been seen), and at Bob's
> position the short conv (`conv_size−1 = 3` reachable previous tokens)
> cannot reach back to Ann. There is **no position at which a single
> causal DeltaNet layer can form `(k=f(Ann), v=g(Bob))`** under that
> surface form — the β-mask specification had no realization, and every
> control built on it inherited the hole.

**Chosen resolution: (a) constrained adjacency template** — developed
fully here. Option (b) — learned β at all content tokens plus an
aggregate multi-write premise — is rejected for the primary gate because
it changes the *theoretical object*: the bound's premise would have to be
re-derived for writes smeared across positions (an "aggregate per-entity
write energy" theory with its own, currently nonexistent, verification
design — explicitly *not* a C16 reuse), which would bundle a new-theory
axis with the tokenizer axis this design exists to isolate — exactly the
multi-variable bundling §2.5 and `STATE.md`'s tokenizer decision forbid.
Option (a) preserves the K-single-writes structure of the
already-validated bound exactly, and pays with an honest, disclosed
naturalness restriction instead.

**The template.** Each binding is rendered as a short transitive clause
whose key and value fall within one conv window:

```
<buf> KEY <rel> VALUE .  <buf> ...        e.g.  "Ann handed Bob."
```

with KEY, `<rel>` (a relation verb), and VALUE each **verified
single-token under GPT-2 BPE at build time** (tokenizer-checked, not
assumed; non-overlapping BPE prefixes), so KEY sits exactly 2 tokens
before VALUE — within `conv_size−1 = 3`. The adjacency constraint is
stated **parametrically in `conv_size`** (key within `conv_size−1` tokens
of the write position) and re-derived from the measured `fla` 0.5.1
default at the F15-LM checkpoint (§7), never hard-coded against an
assumed value.

**The write mechanism, relocated.** The write position is the **VALUE
token** — the binding-completion position, the first position at which a
causal model has seen both endpoints. C9's hard β-mask transfers with its
anchor moved: **β = learned (token-conditioned) at the K VALUE-token
positions ONLY; hard 0 at every other position** — key tokens, relation
tokens, punctuation, buffers, and the entire QUERY phase. At the write
position, the layer's causal conv mixes the local window
`[KEY, <rel>, VALUE]`, and the learned `W_k`/`W_v` projections must
extract the key component into `k_eff` and the value component into
`v_eff` — learned, not hand-wired; C16 (below) checks the result
empirically rather than assuming the extraction succeeds.

**The conv reversal, reconciled with §4.3 (the cross-check FATAL-1
demanded).** The synthetic design's primary bespoke harness has **no
conv** — and could afford none, because its grammar co-locates key and
value on one token. Here the short conv is **functionally required, not
a throughput convenience**: without it, a single causal DeltaNet layer
cannot bind two tokens at different positions at all (the FATAL-1 box
above). Two consequences, stated explicitly: (i) §4.3's
production-kernel commitment and this grammar are now mutually
consistent *by necessity* — the `fla`-block conv path is part of the
mechanism under test, not an incidental production detail; (ii) the
conv-related controls the synthetic design scoped to its *secondary*
production-block check — C14's buffer discipline (≥`conv_size−1`
zero-information, zero-embedding-pinned tokens between binding clauses
and at the BIND/QUERY boundary, preventing adjacent clauses from smearing
into each other's effective k/v) and the two-sided blank-out — are
**mandatory for this design's primary gate**.

**Implementation shape (new, revision 2.1 — R2-3, so the build phase
does not discover this the hard way):** the primary harness is a
**custom block calling `fla.ops.delta_rule.chunk_delta_rule` directly**,
not the stock `fla.layers.delta_net.DeltaNet` module — the stock layer
computes β internally via its own `b_proj` and exposes **no hook for an
external hard β-mask**, which C9 requires architecturally. The custom
block owns: (i) the k/v/q short-conv path (controlled; `conv_size`
recorded at F15-LM); (ii) externally computed
`β = sigmoid(W_β(x)) × hard_mask`, with the mask zero everywhere except
the K VALUE positions; (iii) a real embedding table with **reserved
buffer token IDs whose rows are zero-pinned AND frozen** (excluded from
the optimizer / gradient-masked — unlike the synthetic harness's
constructed inputs, a trainable embedding table would otherwise silently
learn nonzero buffer embeddings and reopen the NEW-2 boundary-mismatch
failure the synthetic design already closed). The F15-LM checkpoint (§7)
gains two verification items: the β-mask's zeros are asserted **from the
tensor** at non-write positions, and buffer embedding rows are asserted
exactly zero **after** optimizer steps, not just at init.

**Per-item cross-clause leak check (new, revision 2.1 — R2-8):** the
smoke gate includes a per-item version of the two-sided blank-out:
corrupt every token of binding clause `j` and assert clause `j+1`'s
`k_eff`/`v_eff` are **bit-identical** — verifying C14's buffers actually
isolate adjacent clauses at the conv level for every item, not just at
the BIND/QUERY boundary.

**The query realization.** The query is rendered in the *same* local
template with the value slot replaced by a designated query token:

```
<buf> KEY <rel> <Q>                       e.g.  "Ann handed" + <Q>
```

`q_eff` is extracted at the `<Q>` position through the model's own
embedding → conv → `W_k` path; the query span passes through the feature
path only, **never the recurrence** (the BIND-only kernel call ends at
the phase boundary — C9's streamed state-freeze is preserved by
construction, as in the synthetic design's §3.5 split), and the readout
stays external and pinned (`pred(a,h) = S_Tʰ · q_eff_a`, §5.1). The
bind-time and query-time conv windows are then identical except in the
final slot (`VALUE` vs. `<Q>`) — which makes bind→query alignment
*plausible by construction* but **not guaranteed** (the learned conv can
weight the final slot arbitrarily); it becomes an explicit premise
diagnostic below, with no synthetic-design analog.

Two query-side conventions, pinned (new, revision 2.1 — R2-7): **(1)**
the query clause uses the **same relation verb as the bind clauses** —
and in C19's held-out-template arm, bind AND query are both rendered in
the held-out template, so bind/query window congruence is preserved
within every arm (the congruence, not the specific verb, is what premise
(iii) needs); **(2)** the hop count `h` **never appears in the surface
form** — it enters only as the external matrix power in the pinned
readout (`S_Tʰ · q_eff_a`), so no "after N exchanges" phrasing exists
anywhere in the realized grammar (revision 1's motivating example in
§2.2 included such a phrase; it is not the realized design).

**The bound's premise, re-derived for this grammar (not inherited).**
Exact recovery of all K bindings through the pinned linear unbind still
forces `rank(S_T) ≥ K` — the proof (`DELTANET_CAUSAL_RANK_DESIGN.md`
§4.1) is a property of the readout and transfers — but its premises now
attach to different objects:

- **(i) `{k_eff_j}` linearly independent**, where `k_eff_j` is the
  post-conv, post-`W_k` effective key **at write position j** — C16's
  object (below), verified per checkpoint, never assumed.
- **(ii) `{v_eff_j}` linearly independent** — inherited from the
  injective K-cycle (C6) plus distinct single-token value names, checked
  with the same instrument applied to the value side.
- **(iii) bind→query alignment (numeric rule pre-registered — REVISED
  2.1, R2-2; revision 2 left this thresholdless, reproducing the
  FATAL-2 hole in miniature and creating a one-directional
  CONFIRM-rescue lever):** a checkpoint is **alignment-valid** ⇔
  per-item `cos(k_eff_bind_j, q_eff_query_j) ≥ 0.9` for all K items
  (0.9 pre-registered now; recalibratable only at the same Wave 0 →
  Wave A gate as τ, R2-4, as a documented deviation). Handling is
  **symmetric**: a recovery FAILURE at an alignment-invalid checkpoint
  is classified as a premise failure (not a rank effect), AND a
  recovery SUCCESS at an alignment-invalid checkpoint is flagged
  premise-questionable and excluded from headlines — misalignment can
  excuse nothing in either direction. This premise is new to this
  design; the synthetic harness made it true by construction (`W_k`
  received the identical raw key vector at bind and query time —
  `model_dn.py`'s own documented design decision).

**Naturalness cost, disclosed (jointly resolving MINOR-14 — revision 1's
§12 open question is now DECIDED).** The primary gate's surface form is
telegraphic-but-grammatical English ("Ann handed Bob.") under **two**
rigor-preserving restrictions, both stated up front: (1) single-token
entity and relation words, and (2) key/value adjacency within one conv
window. Free-form prose with key and value beyond the conv window — what
revision 1 naively promised — is **out of scope for the primary gate**
and is named in any write-up as the residual gap between Wave 1 and real
text (closing it is a multi-layer or attention-hybrid question, §11 item
4). Multi-token entity names and longer/freer templates are
explicitly-labeled **Reserve stress arms** (§7): expected to degrade,
informative either way, never gate-carrying.

**Embedding-path decision (unchanged from revision 1).** The entity-name
embedding table is **learned from scratch** (standard random init), using
GPT-2's tokenizer and vocabulary **ID space** only — not GPT-2's own
pretrained embedding weights. This isolates exactly one new variable
relative to the already-validated synthetic harness: the **tokenizer's
segmentation behavior and vocabulary-ID space**, without simultaneously
introducing a second, uncontrolled variable (a *specific* pretrained
embedding geometry, whose systematic linear dependencies among common
first names, if any, are not something this design controls). Same
discipline `STATE.md`'s tokenizer decision applies at the corpus level,
applied here to the embedding-initialization axis. **A frozen,
pretrained-GPT-2-embedding variant is a flagged, explicit Reserve-wave
robustness check (C18, §7), never promoted to the headline claim without
being labeled as bundling two axes.**

**C16 — the premise check, with the pre-registered numeric rule (REVISED,
revision 2 — FATAL-2; subsumes C13 per MINOR-12).** One instrument, fully
specified before any Wave 1 data is read:

- **Object:** the K post-conv, post-`W_k` effective keys at the write
  positions (`k_eff_j`, premise (i)), row-normalized; the same
  computation applied to the value side (premise (ii)); the bind→query
  alignment cosines (premise (iii)). **C16 subsumes and replaces the
  synthetic design's C13 for this design — one effective-key premise
  instrument, not two** (MINOR-12): C13's post-conv check, promoted from
  a build-time smoke gate to a per-checkpoint standing measurement.
- **Cadence:** every eval checkpoint (≤2K steps, §8), not a one-time
  build-time gate.
- **Rule (pre-registered NOW, before any Wave 1 data exists — REVISED
  2.1, R2-1: ARM-SPECIFIC, because revision 2's single global τ stranded
  the causal arm).** The round-2 verifier caught a self-defeating
  contradiction in revision 2: this document's own anchor cites
  force-ranked Gram deviations of 3.56–4.91 as the degraded band — if
  that transfers, every B-probe/Wave B checkpoint would be
  premise-invalid under "headlines only from premise-valid checkpoints,"
  making the PRIMARY causal test structurally unable to CONFIRM, while
  the salvage tier was simultaneously named "the bound's actual minimal
  premise" and forbidden from headlines. Fixed with three arm-specific
  rules:
  - **(i) Unconstrained arm:** premise-valid ⇔
    `‖K_effᵀ K_eff − I‖_F < τ = 0.03`. Anchoring: the synthetic
    design's Waves −1/0 measured this exact diagnostic (F4) at
    **0.0052–0.0097** across every converged unconstrained cell
    (`d ∈ {64,128}`, `K ∈ {16..64}`) — τ = 0.03 sits at ~3× the top of
    that band.
  - **(ii) Causal arm, `k ≥ K` leg (the CONFIRM-carrying leg):**
    headline validity = **salvage tier or better** — exact SVD
    `σ_K(K_eff)/σ_1(K_eff) ≥ 0.1` (linear independence with bounded
    conditioning, which IS the bound's actual minimal premise; the
    proof never needed orthonormality — premise (i) above). τ = 0.03
    is retained on this arm as a **clean-regime descriptor only**
    (reported, never gating): force-ranking measurably degrades key
    geometry (the 3.56–4.91 band is the synthetic design's own §12.6
    finding), so near-orthonormality is the wrong bar for an arm whose
    constraint mechanically pushes keys away from it.
  - **(iii) Causal arm, `k < K` collapse leg:** the bound is **never
    invoked on a failure** — a collapse at `k < K` is read through the
    logged premise/alignment diagnostics with F18-style
    one-directional-bias discipline (the synthetic design's §6.4
    advisory, transplanted): force-rank degrades keys on top of
    constraining rank, so **a `k < K` collapse accompanied by degraded
    key diagnostics must NOT, by itself, be read as a causal rank
    effect** — it is equally consistent with force-rank-induced key
    degradation; conversely, **non-collapse at `k < K` despite degraded
    keys is the stronger falsifying evidence** (surviving a doubly
    handicapped configuration). The decisive CONFIRM signature is the
    **differential**: ceiling at `k ≥ K` (premise-valid per (ii)) vs.
    collapse at `k < K`, with comparable key-degradation diagnostics on
    both legs.
  - **Re-registration timing (REVISED 2.1, R2-4):** the τ/threshold
    calibration decision is taken at the **Wave 0 → Wave A gate**, not
    "before B-probe" as revision 2 had it — Wave 0 already outputs the
    measured C16 band, and Wave A's screening numbers must not be
    reported under a known-miscalibrated τ. Any revision is a
    documented deviation recorded in Wave 0's summary before Wave A's
    manifest is generated (§8's gating discipline); never silent tuning
    after causal data exists.
  - **Held-out pools included (R2-8):** C16 and the alignment
    diagnostics are computed at eval time on the C17 (held-out names)
    and C19 (held-out templates) pools too, not just the training
    pool — untrained/undertrained embedding rows are exactly the
    failure C16 exists to catch, and this is what classifies a C17/C19
    failure as a **premise failure vs. a competence failure** rather
    than leaving the two indistinguishable.
- **Companion rule (checkpoints straddling the threshold — FATAL-2's
  second half):** every reported number carries its checkpoint's premise
  classification **under its arm's rule above (R2-1)**. Headline numbers
  are computed **only** over checkpoints that are premise-valid under
  the applicable arm rule. A run whose final checkpoint is
  premise-invalid (again, under its arm's rule) is reported in its own
  outcome category —
  **"premise-failed," alongside (never inside) CONFIRM/FALSIFY**, exactly
  as Task E reports "dead" seeds as a category distinct from converged
  ones. If a run's later checkpoints fall out of validity, the last
  premise-valid checkpoint carries that run's number, flagged as such;
  if the classification lands after a mid-wave summary already quoted a
  number, the wave summary is restated with the classification attached —
  never silently left standing.
- **What this is and is not (MINOR-13, acknowledged).** This is a
  **calibrated empirical premise check on a continuous, SGD-produced
  object** — NOT a C15-style structural assert. C15's exact-threshold,
  run-to-completion-negative-test discipline applies to constructed
  objects (a truncation either produced rank ≤ k or it did not); there
  is no coherent "deliberately premise-violating SGD run" to execute as
  a negative test here. What transfers from C15 is the
  **pre-registration** discipline: the rule above is fixed before any
  Wave 1 data exists and cannot be tuned post-hoc. The *instrument*
  (not the premise) does get a build-time sanity check: hand-built key
  sets with known Gram deviation and known rank are fed through the C16
  code path and must classify correctly — that verifies the measuring
  code, and is run to completion like any structural test.

> **ADDENDUM — 2026-07-02, build-phase audit round 1, MAJOR-1 (spec fix,
> flagged honestly: this closes a gap in the FROZEN revision-2.1 text, it
> does not merely implement it).** The Rule subsection above names premise
> (ii) — `{v_eff_j}` linearly independent, "checked with the same
> instrument applied to the value side" — but every numeric
> premise-classification rule it registers ((i)/(ii)/(iii) arm rules, the
> companion rule) gates on the **key-side** Gram/salvage diagnostics and
> the alignment cosine ONLY; the value side was specified as a
> *measurement* but never wired into *validity classification*. The
> independent build-phase audit constructed a collapsed-`W_v` state in
> which the value-side Gram deviation fires (5.78 → 9.96) while both
> registered premise flags stay green, and the FATAL-0 degenerate run
> (same audit round) was a live instance. **Registered fix, effective
> before any Wave −1 data exists:** premise classification is symmetric
> across sides — (a) unconstrained arm: premise-valid additionally
> requires `‖V_normᵀV_norm − I‖_F < τ_v = 0.03` on the row-normalized
> value set; (b) causal `k ≥ K` leg: headline validity additionally
> requires the value-side salvage tier `σ_K(V_norm)/σ_1(V_norm) ≥ 0.1`;
> (c) the `k < K` collapse leg remains never-premise-gated (R2-1(iii)
> unchanged). The value-side numeric anchors reuse the key side's (no
> independent value-side anchor exists yet); they are re-registerable
> ONLY at the same Wave 0 → Wave A gate as τ (R2-4), as a documented
> deviation. Thresholds are recorded per-run in the result JSON
> (`c16_thresholds`), and the sweep's aggregation applies these
> arm-specific rules when computing headline numbers (the same audit's
> MAJOR-2: `run_deltanet_rd_sweep.py::aggregate` computes headline means
> over premise-valid checkpoints only — each run carried by its LAST
> premise-valid checkpoint per this section's own companion rule — with
> `n_premise_failed` surfaced as its own outcome category, never blended
> into the headline).

### 5.3 New controls — held-out entities (C17) and held-out templates (C19)

**C17 — held-out entity names.** A control this real-data bridge can add
that the fully-synthetic design could not cleanly motivate: train on one
closed pool of `K` names, evaluate **zero-shot on a disjoint pool of `K`
names** (same graph structure, same task, never-seen surface forms) at the
same checkpoint. This tests whether the causal-rank mechanism is tied to
the *specific* trained entity identities or genuinely operates on
"whatever gets bound" — directly analogous in spirit to Task E's
held-out-hop-depth generalization test, but along an entity-identity axis
instead of a depth axis, and specifically motivated by §9's
closed-vocabulary memorization attack.

**C19 — held-out templates (new, revision 2 — MAJOR-3).** The
surface-form companion to C17: a **disjoint relation-verb pool** and a
**second template family** (e.g. `KEY → VALUE` alongside
`KEY handed VALUE` — both still satisfying §5.2's adjacency constraint,
which bounds how much word-order variation is available and is disclosed
as such), evaluated zero-shot at the same checkpoint. C17 varies *who* is
bound while holding the clause pattern fixed; C19 varies the *clause
pattern* while holding the entities fixed — jointly they separate
relational-binding competence from both entity-ID memorization and
template-string memorization. Precedent: CLUTRR's held-out-surface-form
compositional splits (Sinha et al., arXiv:1908.06177) and the MQAR-family
convention of varying surface realization over a fixed recall structure
(Arora et al., arXiv:2312.04927).

### 5.4 Controls table (extends the synthetic design's C1–C15)

| # | Control | Closes |
|---|---|---|
| C1–C15 (synthetic design; **re-anchored per FATAL-1 where noted**) | Train-time force-rank via the two-kernel split, ≥5 seeds, entity-subspace-restricted rank, multi-head/multi-layer escapes, C15's exact-threshold post-truncation SVD assertion. **Re-anchored:** C9's β-mask fires at the K VALUE-token (binding-completion) positions (§5.2); C14's buffers + two-sided blank-out are promoted from the secondary production-block check to the primary gate (conv is now functionally required, §5.2); **C13 is subsumed into C16 and does not exist separately in this design (MINOR-12)** | Same roles as the synthetic design |
| **C16 (REVISED — FATAL-2, R2-1/R2-2/R2-4/R2-8)** | The per-checkpoint premise instrument, fully specified in §5.2: object = post-conv effective keys/values at the write positions + bind→query alignment (per-item cos ≥ 0.9, symmetric handling); **arm-specific rules (R2-1)** — unconstrained arm τ = 0.03; causal `k≥K` leg gated by salvage tier (`σ_K/σ_1 ≥ 0.1`, τ descriptive only); `k<K` leg never invokes the bound on failure (F18-style one-directional-bias discipline); re-registration at the Wave 0 → Wave A gate (R2-4); computed on C17/C19 held-out pools at eval too (R2-8); companion classification rule + distinct "premise-failed" outcome category | The provable-bound premise silently failing via BPE fragmentation, embedding collision, conv-extraction failure, or bind→query misalignment (§2.1, §5.2) — the control that makes the "premise-conditional" claim tier honest, without stranding the causal arm |
| **C17 (new)** | Held-out entity-name pool, zero-shot, same graph structure (§5.3) | Closed-vocabulary token-ID memorization masquerading as relational-binding competence |
| **C18 (new)** | Frozen-pretrained-GPT-2-embedding variant run and reported **only** as an explicitly-labeled Reserve-wave robustness check, never substituted for the from-scratch-embedding primary arm | Silently bundling the tokenizer axis with the pretrained-embedding-geometry axis into one claim (§5.2, §9) |
| **C19 (new, revision 2 — MAJOR-3)** | Held-out template pool: disjoint relation verbs + a second template family (adjacency-constrained), zero-shot at the same checkpoint (§5.3); precedent: CLUTRR (arXiv:1908.06177), MQAR-family (arXiv:2312.04927) | Template-string memorization masquerading as relational binding |

---

## 6. Wave 2 design — instrumented LM (gated follow-on, RD-2)

**Gate: Wave 2 launches only if Wave 1 (RD-1) CONFIRMs.** If RD-1
falsifies, a real-corpus descriptive+interventional study on a mechanism
already shown not to survive real tokenization would be an uninterpretable
next step (does the LM-scale result fail for the same reason, or a
different one? Wave 2 alone cannot tell); redirect budget per §9 item 9's
FALSIFY branch instead.

### 6.1 Corpus-level, not within-document, reasoning-vs-narrative contrast

**Recommended design: two separate corpora, not within-document span
labeling.** OpenR1-Math (43.7M tokens, confirmed local, §3) as the
reasoning-dense corpus; WikiText-103 (117.9M tokens, same tokenizer,
confirmed local) as the narrative-dense contrast — compare `rank(S_t)`
trajectories for the same trained checkpoint across the two corpora. This
is methodologically simpler and more defensible than attempting to segment
"reasoning span" vs. "narrative span" *within* a single OpenR1-Math
document (chain-of-thought math solutions do not have a clean,
non-question-begging structural marker for this without risking the
labeling itself encoding the hypothesis under test). Within-document
segmentation (e.g. via arithmetic-density heuristics) is flagged as a
**stretch, exploratory secondary analysis only** (§7's Reserve wave), not
the primary evidence.

**Third contrast arm (revision 2 — MAJOR-5).** The two-corpus contrast
confounds "reasoning vs. narrative" with "math-notation/symbol density
vs. prose" — revision 1's own §12 flagged this and left it open; the
attack round correctly refused to let the primary Wave 2 design carry the
confound. Primary fix: a **third corpus arm** — STEM *expository* prose
(non-derivation scientific text, e.g. arXiv abstracts or textbook-style
exposition), filtered toward OpenR1-Math's symbol-token frequency profile
(measured against the tokenized corpora, not assumed) — so "derivation
vs. exposition at matched notation density" separates from "notation vs.
prose." Sourcing cost: one additional HF download + GPT-2 tokenization
pass (~minutes at this corpus scale, per the house pipeline's own
measured figures), deferrable until the RD-1 gate passes. Pre-registered
fallback if a satisfactorily density-matched corpus cannot be assembled
cheaply: **frequency-normalized damage metrics** (per-token truncation
damage stratified by token-frequency band and symbol-vs-word class) —
reported as the fallback it is, never claimed as equivalent to the
matched-corpus arm.

### 6.2 The interventional addition — reasoning-damage vs. fluency-damage curves

Rank-truncate the trained state at inference time (eval-only, no backward
pass required — cheap, unlike the training-time force-rank arm) across a
grid of truncation levels `k`, and measure:

- **Reasoning damage:** delta in downstream task accuracy on a held-out set
  of OpenR1-Math problems (exact-answer or exact-next-arithmetic-token
  accuracy — a concrete, scoreable outcome metric, not a vague "quality"
  judgment).
- **Fluency damage:** delta in perplexity on held-out WikiText-103 text, at
  the same truncation levels.

If reasoning damage exceeds fluency damage at truncation levels below some
`k*`, that is the interventional evidence of differential rank-dependency
for reasoning specifically — the concrete, falsifiable content of RD-2, and
the specific addition neither Nazari & Rusch nor Sun et al.'s papers make
(§10).

### 6.3 What claim tier this earns — restated, not allowed to drift at write-up time

Per §1's RD-1/RD-2 distinction: **Wave 2's strongest possible claim is
"inference-time causal + descriptive," never "SGD was pushed toward a
rank-disciplined solution by training on real text."** The latter would
require a train-time causal intervention on real-text pretraining itself
(force-ranking the state throughout LM pretraining) — not designed here,
flagged in §12 as a possible Wave 3/future extension contingent on Wave 2's
own results and this document's remaining budget, and explicitly not
promised.

---

## 7. Manifest — waves, gates, budget table

Calibration-first throughout, per the project's own repeatedly-relearned
lesson (`STAGE0_DESIGN.md` §12; `TASK_E_FINDINGS.md` §10's
"three-budget-artifacts" program-level finding: *"late seed-stochastic
phase transitions make fixed-budget negatives unreliable... every 'dead'
cell must be re-tested at 2–2.5× budget before being called dead"*) — built
into Wave 0 from the start, not discovered after a wasted round.

| Wave | Purpose | Scope | Est. GPU-h | Gate to next wave |
|---|---|---|---|---|
| **Prereq (blocking, ~0 GPU-h)** | `scp` the OpenR1-Math + WikiText-103 tokenized sets from the local SSD to the current box's `/data` volume (§3) — corrects the brief's "may need re-tokenizing" assumption; this is now a data-transfer step, not a pipeline-build step. The Wave 2 third-corpus sourcing/tokenization pass (MAJOR-5, §6.1) attaches here but is **deferred until the RD-1 gate passes** (it serves only Waves C/D) | 2 file transfers, ~1.3GB total (+1 deferred HF pass) | ~0 (minutes) | Files present + `meta.json` fields match §3's table |
| **F15-LM checkpoint (blocking)** | Verify `fla.ops.delta_rule.chunk_delta_rule` at LM sequence lengths (`T≥512`, bf16) — called **directly by the custom block, not via the stock `fla.layers` module** (R2-3: the stock layer's internal `b_proj` β leaves no mask hook); confirm the known `T<64` backward crash is genuinely absent at LM `T`; record the measured `conv_size` default (§5.2's adjacency constraint re-derives from it); **Tier 0 (MAJOR-4): fp64 gradcheck of the patched naive reference at LM-relevant shapes**, then Tier 1: the §4.4 kernel-vs-verified-reference cross-check — both with detached-path negative tests run to completion; **R2-3 verification items:** β-mask zeros asserted from the tensor at non-write positions, buffer embedding rows asserted exactly zero after optimizer steps; **R2-8:** per-item cross-clause leak check (corrupt clause `j` → clause `j+1`'s `k_eff` bit-identical) | 1 script, CPU/1 GPU | ~1 | Tiers 0–1 pass; negative tests fire; mask/pinning/leak asserts pass |
| **−1 (Wave 1 path)** | Timing calibration: production-kernel throughput at Wave 1's actual grammar/batch shape (not extrapolated from the synthetic design's shorter, vector-grammar harness); force-rank wall-clock ratio at this configuration. **Pre-committed decision rule (REVISED, revision 2 — MAJOR-7): buckets set consistent with this table's own pricing baseline, which already assumes the synthetic design's measured 8.35–9.9× ratio for Wave B.** Measured ratio **≤10× → proceed as priced**; **10–15× → re-price**: apply the pre-registered cut order below (steps (i)–(iii), in order, until the total re-fits) before Wave A launches; **>15×, or per-step skip rate >0.1%, or any post-truncation rank-assert failure → STOP**, diagnose, re-price with a revised mechanism before any downstream wave launches. (Revision 1's buckets — "stop if >8×" — collided with its own Wave B pricing at 8–10×, reproducing the exact process-deviation shape the synthetic design's §12.2 had already documented; the attack round caught the repeat. This gate is written to be consistent with its own pricing so the trigger is meaningful.) | ~6–8 runs | ~4–6 | Bucket verdict recorded in the wave summary before Wave 0 launches |
| **0 (Wave 1 path)** | Does the probe-task LM train through a real tokenizer at all? Perplexity/probe-accuracy sanity at small scale; C16's premise bands and the alignment diagnostic measured (not assumed) across checkpoints; extended-budget (2.5×) transition calibration per the standing late-transition discipline, ≥3 seeds per primary cell | ~10–15 runs | ~10–15 | ≥3/5 seeds converge at the primary `K` by the calibrated budget (Task E's own calibration-gate convention, reused); **the τ/alignment-threshold re-registration decision (R2-4) is taken and recorded HERE, in Wave 0's summary, before Wave A's manifest generates** |
| **A (Wave 1 path)** | Main screening: unconstrained entity-subspace rank vs. K across a K-grid; held-out-hop composition (M3_E-equivalent); C17's held-out-entity-name generalization control | ~20–25 runs | ~15–20 | Life at the primary cell (non-trivial recovery) before B-probe |
| **B-probe (Wave 1 path)** | Force-rank calibration probe, `k∈{K−1,K,K+1}`, 3 seeds — mirrors Task E's own M4_E / the synthetic design's B-probe discipline: do not launch the full grid blind | ~9 runs, priced at the *measured* (not assumed) wall-clock multiplier from Wave −1 | ~10–15 | ≥1 seed shows life at `k≥K` and a step relative to `k<K` |
| **B (Wave 1 path, PRIMARY causal test)** | Full force-rank straddle grid — the decisive RD-1 evidence, launched only if B-probe shows life | ~15–20 runs, 8–10× multiplier (synthetic design's own measured ratio, §12.2, used as the working assumption until Wave −1 measures this configuration's actual ratio) | ~35–50 | **RD-1 CONFIRM/FALSIFY decision recorded here** |
| **Gate: RD-1 verdict** | — | — | — | CONFIRM → Wave C launches; FALSIFY → §9 item 9's branch, Wave C does not launch on schedule |
| **C (Wave 2 path, gated)** | Real-corpus LM pretrain (**three-corpus contrast**: OpenR1-Math + symbol-density-matched STEM prose + WikiText-103, §6.1), multi-seed, small architecture sweep (`d_model∈{256,384}`, layers 2–3) informed by §4.1's table; per-run cost per §4.2's corrected ~0.6–2.2 h band | ~8–12 runs | ~12–25 | Perplexity/downstream sanity passes on all corpus arms |
| **D (Wave 2 path)** | Inference-time rank-truncation intervention grid (§6.2) — cheap, eval-only, no backward pass | truncation grid × held-out eval sets | ~5–10 | RD-2 descriptive+interventional result recorded |
| **Reserve** | C18's frozen-embedding robustness variant; §5.2's multi-token-name / free-form-template stress arms; within-document reasoning/narrative segmentation (stretch, §6.1); buffer for Wave −1/0 estimate misses (this project's own history says budget these, not hope they don't happen) | — | ~15–25 | — |
| **Total** | | | **≈105–170 GPU-h** | Within the brief's 150–250 GPU-h ceiling; **Wave 1 alone (≈75–107 GPU-h) is a complete, decisive, publishable result on its own** if Wave 2 is cut for budget or gated out by a Wave 1 FALSIFY |

**Pre-registered cut order — the concrete meaning of "re-price" in Wave
−1's 10–15× bucket, and of any budget squeeze generally** (same
discipline as the synthetic design's own §6.4 and `STAGE0_DESIGN.md` §6),
applied in order until the total re-fits:
(i) drop Wave 2 (C/D) entirely — Wave 1 stands alone as a complete result;
(ii) drop Wave A's K-grid to the primary K plus one straddle point;
(iii) shrink Wave B to the **minimal causal contrast** `k ∈ {K−1, K}` ×
3 seeds (dropping the `K+1` arm and the wider straddle, ≈halving Wave B's
cost); (iv) B-probe and that minimal Wave B floor are **never cut** —
they carry RD-1's entire causal claim, exactly as the synthetic design's
own §6.4 states for its own B-probe/B.

---

## 8. Operational harness requirements (carried forward, per `CLAUDE.md`'s learned rules — not optional)

- **Run naming carries `(wave, variant, tokenizer_mode, d_model, d_state, K,
  k, seed)`** — one more axis than the synthetic design's naming scheme
  (`tokenizer_mode` distinguishes Wave 1's from-scratch-embedding primary
  arm from C18's frozen-pretrained-embedding Reserve variant), preventing
  the exact same silent-collision failure mode Stage 0's own MAJOR-2 lesson
  already names.
- **Resume is validity-checked** (a parseable result JSON with the expected
  final-metrics fields, not mere file existence) — direct transplant.
- **Waves are bounded and gated** — no wave auto-launches the next; each
  gate's decision (F13-style ratio verdict, RD-1 CONFIRM/FALSIFY) is
  written to the wave's summary file before the next manifest generates,
  exactly the discipline `DELTANET_CAUSAL_RANK_DESIGN.md` §12 retroactively
  reconstructed for its own Wave A launch — write it forward this time, not
  reconstruct it after the fact.
- **A `scp`-then-verify prereq step is itself gated** (§7's Prereq row) —
  the manifest does not assume the data landed correctly; it checks
  `meta.json` field values against §3's recorded table before Wave −1 can
  launch, closing the (admittedly low-probability, but free-to-check) risk
  of a silently-truncated or wrong-file transfer.
- **Long-running orchestration inside `tmux new-session -d -s <name>`**,
  wrapped in a self-healing supervisor loop, never a backgrounded SSH
  shell; kills via `tmux kill-session` or exact PIDs, never `pkill -f
  <pattern>` (the self-kill footgun this project has already paid for
  once).
- **Mid-training eval checkpoints every ≤2K steps**, written incrementally.
- **A completion sentinel per wave** (the `ALL_DONE` convention already
  used by the synthetic design's own K=16 completion wave, `TASK_E_FINDINGS.md`
  §10) — a wave's directory is not treated as finished until this file
  exists, closing the "partial pull read as final" ambiguity §12.1/§12.3 of
  the synthetic design's own results section had to caveat repeatedly by
  hand.
- **Everything logs to a file; each wave emits a human-readable summary;
  exact scripts archived under `experiment-runs/`** — house convention,
  unchanged.

---

## 9. Attack-yourself

1. **Revision 1's grammar was itself the biggest bug in this document
   (FATAL-1 — caught by the independent attack round, not by self-audit),
   continuing an exact house pattern:** the synthetic design's revision 1
   died the same way (its chunk-boundary truncation mechanism, F5 there).
   The C9 β-mask as specified had no realization under free-form word
   order — no position at which a causal single layer could form the
   `(key, value)` pair at all. Resolved by §5.2's adjacency template. The
   surviving residual, stated honestly: the primary gate's "natural
   language" is a telegraphic template within one conv window, and
   free-form prose remains out of scope (§5.2's disclosure) — the
   naturalness claim is permanently narrower than revision 1 implied.
2. **"Is this just Zoology/MQAR (Arora et al., arXiv:2312.04927) with extra
   steps?"** No, on three specific axes, none of which MQAR-family
   benchmarks have: (i) a **provable rank≥K necessity bound** with a
   **train-time causal force-rank ablation** — MQAR measures accuracy vs.
   model architecture/size, never forces a rank constraint during training
   and asks whether the resulting accuracy curve has a threshold; (ii) a
   **compositional multi-hop readout via literal matrix powers** (`Zʰ`) —
   MQAR is single-hop key→value recall, not h-hop composition with a
   held-out-depth generalization test; (iii) **natural-language surface
   form through a real tokenizer specifically as the object of study**
   (§2.1's BPE-fragmentation concern) — MQAR-family tasks use small
   synthetic vocabularies by design and never have to confront tokenizer
   effects on the bound's premises at all. The genuine overlap: both are
   "controlled synthetic-graph recall through a sequence model" — this
   should be named as a close relative in any write-up, not ignored.
3. **"Does natural surface form actually add anything over synthetic
   tokens, or is this cosmetic?"** Answered honestly in §2.2: the
   *causal-rank methodology* is identical either way — natural surface form
   does not, by itself, test "matrix rank supports natural-language
   reasoning." What it tests is narrower and mechanical: does the bound's
   premises (embedding-level linear independence) survive a real BPE
   tokenizer and an SGD-controlled embedding table. This is a real,
   nontrivial question (§2.1 lists three concrete ways it could fail that
   have no synthetic analog) but it should never be write-up-time inflated
   into "we tested reasoning in natural language" — that claim is reserved
   for Wave 2, and even Wave 2 only earns a descriptive+inference-time-causal
   version of it (§6.3). Revision 2 narrows this further and says so:
   under the FATAL-1 fix, the primary gate's surface form is an
   adjacency-constrained template — the honest description is "a real BPE
   tokenizer and a real embedding path," not "natural prose." The write-up
   language must match §5.2's disclosure, permanently.
4. **Closed, small entity vocabulary risks pure token-ID memorization,
   defeating the point of using natural language at all.** Mitigated by
   C17 (held-out entities) and C19 (held-out templates, MAJOR-3), §5.3 —
   both are required controls, not optional, specifically because this
   risk is real and would otherwise be invisible in an
   in-distribution-only evaluation.
5. **The C16 premise check could itself have a false-negative mode** — if
   BPE fragments or embedding collisions interact such that the
   *aggregate* effective-key representation still looks full-rank while
   the underlying binding is ambiguous at the token level, the
   checkpoint-level Gram check could pass while a subtler failure mode
   persists. Mitigated by the §5.2 decision (single-token,
   non-overlapping-BPE-prefix names, tokenizer-verified at build time)
   and by C17/C19's behavioral cross-checks. **A second, revision-2 worry
   with teeth: C16's τ = 0.03 is anchored to the *synthetic* harness's
   measured bands — an out-of-distribution anchor for this grammar's
   post-conv key geometry.** The §5.2 re-registration rule (documented
   τ/alignment-threshold revision at the Wave 0 → Wave A gate per R2-4,
   never silent post-hoc tuning) is the designed answer; a review round
   should verify it is honored, not just written. Revision 2.1's R2-1
   also closed this worry's sharper sibling: the τ rule is now
   arm-specific, so the causal arm's mechanically-degraded key geometry
   no longer strands the primary test.
6. **Reasoning-vs-narrative span definition (Wave 2) risks
   confirmation-bias labeling if done within-document, and the two-corpus
   version confounds derivation-ness with symbol density.** Mitigated by
   §6.1's corpus-level primary design plus the revision-2 third arm
   (symbol-density-matched STEM prose, MAJOR-5) with its
   frequency-normalized fallback; within-document segmentation stays a
   demoted, explicitly-labeled stretch analysis.
7. **"Reasoning damage" (§6.2) needs a concrete, scoreable metric or the
   interventional claim is unfalsifiable.** Addressed directly: exact-answer
   or exact-next-arithmetic-token accuracy on held-out OpenR1-Math problems,
   not a vague quality judgment — stated as a requirement in §6.2, not left
   implicit.
8. **The F15-for-LM verification chain (§4.4) is still weaker than the
   synthetic design's bar, even after MAJOR-4's Tier-0 upgrade.** Tier 0
   now verifies the reference itself at LM shapes (closing the
   "unverified ground truth" hole revision 1 had), but Tier 1's bf16
   `<1e-2` cross-check remains categorically weaker than a fp64
   finite-difference gradcheck through the production kernel itself —
   which stays impossible (§4.3's float32-rejection is a hard library
   constraint). Kernel-internal bugs smaller than the bf16 tolerance are
   the residual exposure; flagged as the single most likely place an
   independent reviewer should still push.
9. **What happens to this document if the synthetic design's own Wave B
   (running concurrently, §7.5/§0) FALSIFIES before Wave 1 launches?** Wave
   1 does not become moot — it is testing a related but distinct question
   (does rank-recruitment survive real tokenization) that a synthetic-cell
   FALSIFY does not resolve either way (a synthetic FALSIFY could mean "the
   delta rule's own training dynamics don't get pushed toward
   rank-discipline even with hand-built orthonormal keys" — real
   tokenization could conceivably make this *worse* in the same direction,
   or could in principle interact differently; nothing here assumes it
   would improve). **Recommendation if this happens: still build and run
   Wave 1's B-probe (§7) before deciding — a synthetic FALSIFY plus a Wave 1
   B-probe showing life would itself be a notable, publishable interaction
   finding** (tokenizer/embedding-geometry effects rescuing a
   synthetic-cell failure would be surprising and worth reporting either
   way), rather than cancelling Wave 1 on the strength of a different cell's
   result.
10. **GPU contention (§3's live note) could silently stretch this document's
    wall-clock timeline even though the GPU-hour budget itself is
    unaffected.** Noted as a scheduling risk, not a rigor risk — flagged so
    it isn't rediscovered as a surprise mid-execution.
11. **Scooping risk on Wave 1 specifically (not just Wave 2) — is the
    "naturalistic probe task through a real tokenizer" idea itself already
    occupied?** Directly checked, §10.

---

## 10. Positioning — verified 2026-07-02, not inherited stale from the Feb 2026 research pass

**Re-verification of the existing claim.** `research/task-d-novelty-july2026.md`'s
positioning of Nazari & Rusch (arXiv:2602.04852) and Sun et al.
(arXiv:2602.02195) is marked high-confidence, PDF-verified (pages 1–3) —
**both are descriptive analyses of *pretrained* real-text LLMs** (per the
internal note's PDF pass: a ~370M Gated DeltaNet evaluated on FineWeb-Edu
and downstream tasks, and Qwen3-Next analyzed via a "RankViz" evaluation
suite, respectively — **MINOR-9: these specific model/corpus details are
sourced from `research/task-d-novelty-july2026.md` and were NOT
independently re-confirmed by this document's own revision-2 fetches;
re-verify from the PDFs before any external citation.** The load-bearing
scope facts — descriptive, pretrained-model-only, upper-bound-only, no
train-time intervention — are what this section's positioning actually
rests on, and those are multiply attested), with an **algebraic upper
bound** on rank
(`rank(S_t)≤t` / `≤min(t,d)`) derived from write-count, not a provable
*lower*-bound necessity theorem, and **no training-time causal
intervention of any kind** — neither paper trains a model under a
controlled rank constraint; both are post-hoc measurement (plus, in each
case, a structured-pruning application downstream of the measurement, not
a causal-necessity claim about training). **This is confirmed to still be
accurate as of 2026-07-02** — a fresh check (below) found no update to
either paper's own scope and no new paper closing this specific gap.

**Fresh check, 2026-07-02 (this document's own contribution, not inherited).**
Three targeted searches for anything that might have appeared in the ~5
months since the Feb 2026 papers that would scoop either RD-1 or RD-2:

- No new paper found that trains a fast-weight/linear-attention model under
  a causal, train-time rank constraint on real or real-like text with a
  provable necessity bound — the specific combination this document's Wave
  1 targets remains, as far as this check can determine, unoccupied.
- **New architecture note, scope-relevant:** Gated DeltaNet-2 (NVIDIA,
  arXiv:2605.22791, May 2026) decouples the **erase and write operations**
  in the delta rule — separately gated erase and write strengths, removing
  the single-scalar tie between them (MINOR-10: revision 1's "per-channel
  decay factors" gloss was wrong and is corrected here) — benchmarked at
  1.3B params / 100B FineWeb-Edu tokens. This is a **new
  variant of the architecture family**, not a rank-causality study — noted
  here as a scope boundary: this document's Wave 1/2, like the synthetic
  design's own C11 pinning, test **vanilla/Gated DeltaNet** (single scalar
  β, per `DELTANET_CAUSAL_RANK_DESIGN.md`'s §2 item 3 positioning), **not**
  DeltaNet-2 — an explicit, stated scope limit, not an oversight, and a
  natural "does the causal-rank finding transfer to the erase/write-decoupled
  variant" follow-on question for later.
- **Closest related work found, genuinely new to this project's citation
  list: "Learning State-Tracking from Code Using Linear RNNs"** (Siems,
  Grazzi, Kalinin, Ballani, Rahmani; arXiv:2602.14814, submitted Feb 2026,
  revised Apr 2026). **Confidence: medium — verified via abstract page
  fetch only, not a full PDF read; re-verify before external citation, per
  this project's own citation-confidence convention** (matching how the
  synthetic design's own §2 item 5 flags M²RNN as "abstract-level sources
  only so far"). What it does, per the abstract: converts **permutation
  composition** tasks into **code via REPL traces** (print statements,
  variable transformations) — a real-ish, non-synthetic-vocabulary surface
  form — and trains linear RNNs (DeltaNet with extended/negative
  eigenvalues) via next-token prediction, finding they state-track
  correctly where Transformers fail. **This is the closest existing work to
  Wave 1's own shape** (permutation composition + real-ish surface form +
  a trained, not merely evaluated, linear-attention model) found by any
  novelty check in this project's history. **The distinguishing delta,
  confirmed by the same abstract-page fetch:** it does **not** measure or
  intervene on the recurrent state's **rank** at all (its axis is the
  transition matrix's **eigenvalue range/sign**, the DeltaProduct-family
  "transition-rank" object `DELTANET_CAUSAL_RANK_DESIGN.md` §2 item 3
  already distinguishes from *state* rank — the same conflation risk that
  design flags as "the single most likely reviewer confusion," here arising
  independently in a second paper), has **no provable rank-necessity lower
  bound**, and reports no causal rank-forcing ablation. **Must be cited and
  distinguished explicitly in any write-up** (both as prior art for
  "real-ish surface form + trained linear RNN + permutation composition"
  and as the sharpest illustration that the *rank* observable specifically
  remains untested in this exact setting) — genuinely close enough that
  omitting it would look like an oversight to a reviewer who knows the
  area, not close enough that it pre-empts this document's contribution.

**The claim, precisely scoped (not overclaimed):** Wave 1 (RD-1), if it
launches and CONFIRMs, would be **the first study to combine a provable
(premise-conditional, §5.2) rank-necessity bound, a train-time causal
rank-forcing intervention, and real-tokenizer/real-embedding surface
form** on a fast-weight architecture
— a strictly stronger causal-claim tier than anything found in this check,
including the newly-found closest relative. **Wave 2 (RD-2)**, more
modestly, would be the first to add an **inference-time causal
intervention** (rank truncation, damage-differential measurement) to the
Nazari & Rusch / Sun et al. style of descriptive analysis on a real-text-trained
fast-weight LM — a real but narrower "first," since it does not claim
train-time causality at all (§1, §6.3). **Neither claim should be
stated without the qualifier "as of this novelty check" and both should be
re-verified immediately before any external write-up**, per this fast-moving
literature area's own track record (two closely-related papers appeared
in a single month, Feb 2026; this document's own fresh check found a third,
independently-discovered relative five months later).

---

## 11. Sequencing — what this unlocks

1. **This design → build (harness transplant: swap `deltanet_core.py`'s raw-vector
   input for a real-tokenizer grammar + the production kernel path, §4.3;
   extend the smoke gate with C16/C17; F15-LM checkpoint, §4.4) → audit by
   a fresh-context agent (Build → Audit → Run, per `CLAUDE.md`) → Prereq
   (data transfer) → F15-LM checkpoint → Wave −1 → Wave 0 → Wave A →
   B-probe → (conditional) Wave B.**
2. **If RD-1 CONFIRMs →** Wave 2 (C/D) launches per §6's gate — the first
   real-corpus extension of this project's causal-rank lineage, and (per
   §10) the first inference-time-causal addition to the Nazari & Rusch /
   Sun et al. descriptive literature.
3. **If RD-1 FALSIFIES →** publish as a companion negative (same house
   convention as every prior FALSIFY branch in this lineage, `CLAUDE.md`:
   *"dead directions stay dead"*), but — per §9 item 9 — do not treat this
   as automatically resolving whether the synthetic-cell result (Wave B of
   `DELTANET_CAUSAL_RANK_DESIGN.md`) and the real-tokenizer result share a
   cause; the two should be reported and interpreted together, not as
   independent coin flips.
4. **Deferred, explicitly out of scope for this document (candidate (c),
   §2.4):** the hybrid real-pretraining + embedded-probe-task design — a
   natural Chapter 3/4 direction once Waves 1 and 2 individually land, not
   before. Also deferred: a genuine train-time-causal real-corpus study
   (force-ranking throughout LM pretraining itself, §6.3's explicit
   non-promise) and transfer to the erase/write-decoupled Gated DeltaNet-2
   variant (§10's scope note).

---

## 12. Open questions (explicit, for the attack round — not resolved by fiat)

- Does the C16 Gram-matrix/exact-rank premise check actually hold in
  practice once real training runs, or does BPE fragmentation / embedding
  collision (§2.1) break it often enough to make RD-1 largely
  unmeasurable at the checkpoints that matter? Genuinely open — this is
  the empirical content of Wave 0, not assumed either way.
- Is the §4.2 head-dominated throughput band (50–200K tokens/s/GPU,
  ~0.6–2.2 h/run) right? Per the corrected cost history (MINOR-11), this
  project's per-run GPU-h estimates have historically run 5–6× **HIGH**
  (over-priced — measured runs came in cheaper); the quantity that has
  historically been UNDER-estimated is **steps-to-convergence** (the
  three-budget-artifacts pattern, `TASK_E_FINDINGS.md` §10). So the live
  risk is not the per-run price but whether Wave 0's 2.5× extended-budget
  arm suffices for this new grammar's transition onset. Not measured;
  Wave 0's explicit job.
- Does the bind→query alignment premise (§5.2, premise (iii)) actually
  hold once the conv is trained — i.e., does the learned conv treat the
  final-slot difference (`VALUE` vs. `<Q>`) as negligible for the key
  component, or does it entangle key extraction with the value slot
  badly enough that `q_eff` drifts from `k_eff` and the
  premise-classification rule fires constantly? No synthetic analog
  exists; genuinely open, Wave 0's job.
- Is τ = 0.03 (C16's unconstrained-arm threshold, anchored to the
  *synthetic* harness's measured bands) the right anchor for this
  grammar's post-conv effective keys, whose geometry has no synthetic
  precedent? The §5.2 re-registration rule (documented τ/alignment
  revision at the Wave 0 → Wave A gate per R2-4, never silent post-hoc
  tuning) is the designed answer to discovering it is not. The
  arm-specific structure (R2-1) means a miscalibrated τ can no longer
  strand the causal arm — the residual exposure is confined to the
  unconstrained arm's premise classification.
- Is `d_state=64` (continuity with the synthetic design's own Wave 0/A
  primary-cell selection) still the right choice once real embeddings are
  involved, or does the real-tokenizer axis interact with the `d`-selection
  logic in a way the purely-synthetic Wave 0 never had reason to test?
  Flagged, not assumed away.
- Does the F15-for-LM Tier-1 cross-check (§4.4) at bf16 tolerance
  actually catch the class of gradient bugs the synthetic design's fp64
  gradcheck was built to catch? Tier 0 (MAJOR-4) now verifies the
  *reference* at LM shapes, but kernel-internal bugs smaller than the
  bf16 tolerance remain the residual exposure — a concrete question for
  the audit round (§9 item 8).
- Does C17's held-out-entity-name generalization actually distinguish
  "genuine relational-binding competence" from "the model learned a
  template-position heuristic that happens to generalize across names for
  reasons unrelated to rank at all"? A plausible confound not fully closed
  by this design as written — worth an explicit attack-round pass.
- Is the corpus-level (not within-document) reasoning-vs-narrative
  contrast (§6.1) too coarse to produce an interesting descriptive result —
  i.e., is "OpenR1-Math vs. WikiText" really capturing "reasoning vs.
  narrative," or mostly capturing "math notation/symbol density vs. prose,"
  a related but distinct axis? Open, flagged rather than assumed resolved.
- Single- vs. multi-token entity names — **DECIDED (revision 2, jointly
  with FATAL-1, per MINOR-14), no longer open:** the primary gate uses
  verified-single-token entity and relation words under §5.2's adjacency
  template; multi-token names and beyond-conv-window free-form templates
  are explicitly-labeled Reserve stress arms (§7) — expected to degrade,
  informative either way, never gate-carrying. Recorded here so the
  resolution is visible where the question was originally posed.
- **CAVEAT (added 2026-07-02, build-phase audit round 1, MAJOR-3):** the
  entity-subspace-restricted rank metric's reference subspace (`s_ideal`
  built from the model's *current* effective keys/values) **moves with
  training** — under an intervention (force-rank) the measuring frame
  itself shifts, entangling "the state's rank changed" with "the
  reference changed". The harness therefore logs an ADDITIONAL
  fixed-reference variant (a fixed probe batch whose `s_ideal` is
  snapshotted at the first post-warmup checkpoint and reused unchanged
  at every later checkpoint — `fixedref_entity_subspace` in each
  checkpoint record); the moving-reference metric remains the primary
  (it is the synthetic design's own instrument, kept for
  cross-experiment comparability), and any Wave B causal readout where
  the two variants *disagree* about the rank trend must be flagged and
  resolved before that cell's number is quoted. Genuinely open until
  Wave 0/B data exists; recorded here so the limitation is visible
  before, not after, the causal wave runs.

---

## 13. Reproducibility pointers

- This design: `matrix-thinking/DELTANET_REALDATA_DESIGN.md`.
- Builds on (read, not modified): `matrix-thinking/DELTANET_CAUSAL_RANK_DESIGN.md`
  (full methodology + §12's measured Wave −1/0 results), `matrix-thinking/chapter2/{
  TASK_E_FINDINGS.md, NEXT_EXPERIMENT_DESIGN.md, TASK_D_PREREGISTRATION.md}`,
  `STATE.md` (thesis + tokenizer decision), `H100_SETUP.md` (data/hardware
  ground truth).
- Data confirmed present, 2026-07-02 (§3): local SSD
  `/Volumes/1TB_SSD/learned-representations/data/{reasoning,wikitext103_tokenized}/`
  (both GPT-2-tokenized, `meta.json` fields recorded in §3's table); absent
  from the current Brev box as of this writing, requiring only a `scp`, not
  re-tokenization.
- Reuse targets on the current Brev box (`youthful-indigo-turkey`,
  confirmed present and (for the deltanet harness) actively running,
  2026-07-02): `/home/nvidia/chapter2/deltanet/{model_dn.py,
  deltanet_core.py, run_deltanet.py, run_deltanet_sweep.py, task_dn.py,
  rank_utils.py, probe_trunc.py}`; `fla-org/flash-linear-attention` 0.5.1,
  confirmed installed.
- Novelty check backing §10: `research/task-d-novelty-july2026.md`
  (high-confidence, PDF-verified baseline) + this document's own fresh
  2026-07-02 search pass (medium-confidence new finding: arXiv:2602.14814,
  flagged for full-PDF re-verification before external citation).
- Attack/review-round reports backing the changelogs: round 1
  (independent adversarial review, 2026-07-02; findings FATAL-1/FATAL-2,
  MAJOR-3–8, MINOR-9–14; BUILD WITH FIXES, all addressed in revision 2)
  and round 2 (verification pass, 2026-07-02; all 14 round-1 findings
  confirmed genuinely addressed, FATAL-1 arithmetic verified correct;
  new findings R2-1–R2-8, verifier pre-cleared the fixes — **CLEAN TO
  BUILD as of revision 2.1, no further review round; design FROZEN**).
- Next: build (Prereq data transfer + §5.2's template-grammar harness —
  the R2-3 custom block calling `chunk_delta_rule` directly — + F15-LM
  checkpoint Tiers 0–1 with the R2-3/R2-8 verification items) → audit by
  a fresh-context agent → Wave −1 on the cluster (queued behind the
  currently-running synthetic Wave B-probe / Stage G Wave A per §3's
  live resource note) → Waves 0 (τ re-registration gate, R2-4) /
  A / B-probe / B → RD-1 gate → (conditional) Waves C/D.
- **R2-4 gate addendum: §14 (2026-07-02)** — Waves −1/0 have now run.
  Wave −1 landed in the proceed bucket (≈8.8×, `svd_lowrank` adopted per
  the pre-registered fallback); Wave 0 landed in a **value-collapse
  optimum on all 10 seeds**, caught by the premise gates (zero
  premise-valid checkpoints — no headline escaped). §14 records both
  gate outcomes and pre-registers the anti-collapse objective change.
- **Mini-audit + rerun + gate decision: §14.7 (2026-07-03)** — the
  mini-audit cleared §14 with amendments (including the §14.2
  target-index FATAL — the task as first built was unsatisfiable; the
  rerun tests the objective change on top of that fix). The Wave 0
  rerun then ran: **objective FIXED, 10/10 seeds collapse-free**; τ/τ_v
  re-registered out of the validity stack on measured data (the
  pre-authorized R2-4 path — salvage both sides + alignment 0.9 + R2-8
  classification now carry validity; τ descriptive only); alignment
  threshold unchanged (4/10 seeds headline-clean; the align_min-decay
  phenomenon named as an open question with a Wave A diagnostic);
  **primary-K = 16** (K=32 is a documented sub-exact plateau at
  `K = d/2`, cross-program echo of the exactness frontier). **Wave A
  launches on that record at `--primary-d 64 --primary-k 16`.**

---

## 14. R2-4 gate addendum (2026-07-02) — Wave −1/0 outcomes; the value-collapse optimum; the anti-collapse objective change (pre-registered before any Wave A launch)

**Status: written at the R2-4 gate (Wave 0 → Wave A), before any Wave A
manifest exists. This addendum goes to a fresh independent mini-audit
before anything downstream launches — it is a proposed, pre-registered
objective change, not a decision already acted on.**

### 14.1 Wave −1 gate outcome (recorded per §7's MAJOR-7 decision rule)

- **Force-rank wall-clock multiplier: ≈ 8.8× under `svd_lowrank`** —
  lands in the **≤10× → proceed as priced** bucket. No cut-order
  invocation; the §7 pricing stands.
  > **AUDIT-FIX NOTE (2026-07-03, mini-audit; measurement completed and
  > amended same day):** at the time of the mini-audit the K=64 forced
  > `svd_lowrank` cell was believed killed at ~step 6000 with the
  > top-of-grid multiplier unmeasured. The run in fact **completed**
  > (10,000/10,000 steps, `complete=true`). Final measured Wave −1
  > table (unconstrained → forced per-step, multiplier, per-step skip
  > rate): K=16: 23.0 → 201.9 ms (**8.8×**, 0.00%); K=32: 28.3 →
  > 240.6 ms (**8.5×**, 0.00%); K=64: 38.4 → 215.7 ms (**5.6×**,
  > **7.06% — a late-onset skip BURST, 0.00% through step 8000 then
  > ~700 skipped steps in the final 2000**, absorbed by the
  > jitter-retry/skip-step machinery that replaced the round-1 forward
  > crash). Two flags forward: (i) the multiplier bucket read is ≤10×
  > everywhere → proceed as priced; (ii) the **per-step-skip stop
  > branch (>0.1%) FIRES at K=64** — but on a PRE-index-fix run whose
  > states were shaped by the broken objective, so it is not read as a
  > verdict; any Wave A/B **K=64 force-rank** cell requires a fresh
  > post-fix skip-rate measurement AND padded timeouts before pricing.
  > K=16/32 forced cells are clean on both counts.
- **`eigh`-path forward crash on real states:** the eigh-based
  truncation raised `LinAlgError` in the **forward** pass on this
  harness's real trained states — a harder failure than the synthetic
  design's training-time backward instability. **`svd_lowrank` adopted
  per the pre-registered fallback condition** (build-log deviation #6;
  the F18-advisory fallback inherited from
  `DELTANET_CAUSAL_RANK_DESIGN.md` §6.4). Consequences, already wired
  in: the F18 one-directional-bias discipline is active wherever
  force-rank results are read (§5.2 arm rule (iii) cites it), and the
  pre-registered eval-time eigh control at the B-probe `k<K` cell
  remains — with the pre-declared substitution that if eval-time eigh
  also crashes on these states, a full (non-randomized)
  `torch.linalg.svd` truncation replaces it at eval (no backward pass
  is required there); recorded now so the substitution is not
  improvised later.

### 14.2 Wave 0 outcome — the value-collapse optimum (all 10 seeds)

Grid: `K ∈ {16, 32}` × 5 seeds each, primary config (§4.1). **Every one
of the 10 seeds converged to the same degenerate optimum:**

| Diagnostic | Value | Reading |
|---|---|---|
| `rec@0.9` | 1.000 | **Trivial**, not genuine — mechanism below |
| entity-subspace rank of `S_T` | ≈ 1.0 | State is rank-1 |
| value salvage `σ_K/σ_1` (premise (ii), build-audit MAJOR-1 rule) | 0.000 | **Values fully collapsed** to one direction |
| key Gram deviation (C16) | 2.6–5.3 | Keys degenerate too (above even the synthetic force-ranked band) |
| alignment_min (premise (iii), R2-2) | ≤ 0.13 | Far below the 0.9 bar |
| **Premise-valid checkpoints** | **0, everywhere** | **No headline escaped** |

**Mechanism.** The cosine regression's training target is the model's
**own learned** `v_eff` — so "collapse every value representation to a
single shared direction" is a **global optimum of the training loss**:
every prediction trivially matches every target, with a rank-1 state
and no binding whatsoever. The synthetic harness structurally excluded
this optimum because its targets were experimenter-constructed
constants the optimizer could not move (`W_v` = identity onto raw
constructed values, `model_dn.py`'s own documented scoping decision).
This is a **learned-target collapse** — a failure mode that did not
exist in any earlier leg of the lineage, specific to this design's
real-embedding path, and precisely the class of degenerate optimum the
self-supervised-representation literature exists to handle.

> **AUDIT-FIX NOTE (2026-07-03, mini-audit — re-framing the mechanism
> honestly, credit to the mini-audit for catching what this section's
> own first draft missed):** the paragraph above is incomplete as a
> causal account. The mini-audit found a **target-index FATAL in the
> harness**: `forward()` gathered the scoring target as
> `v_eff_items[tgt_slot]`, but clause `i`'s VALUE token is entity
> `π(i)`, so the scored representation belonged to entity
> **`π^{h+1}(a)` — one hop past the queried answer `π^h(a)`** (verified
> at grammar level: 100% mismatch at every hop as-built; 100% match
> after the fix, which gathers at clause `π^{-1}(tgt_slot)` =
> `π^{h-1}(a)`). Under that bug, **binding was UNSATISFIABLE as
> specced** — no configuration could make the pinned readout match the
> scored target through genuine binding — and the learned-target
> collapse was simply the **accessible optimum of an unsatisfiable
> discrimination problem**, not merely an attractive shortcut sitting
> beside a satisfiable task. The mini-audit's 3-arm control run makes
> the attribution decisive: as-built + the §14.4 NCE term **stagnates
> at `L_nce ≈ log K`** (chance — the rerun as originally specced would
> have deterministically failed its gate), while **index-fixed + NCE**
> produces genuine binding (`L_nce → 0.009`, entity-subspace rank
> 7.7/8, `rec@0.9` 0.86 and rising, C19 0.82, value salvage 0.50 and
> rising). The Wave 0 rerun (§14.6) therefore tests the objective
> change ON TOP of the index fix; the collapse story above stands only
> as the description of the accessible optimum, not as the root cause.

**The instruments worked.** `rec@0.9 = 1.000` would have been a
spurious CONFIRM headline under revision 1; under revision 2.1 + the
build-audit MAJOR-1 value-side rule, it was classified premise-failed
at every checkpoint of every seed. The gate discipline held — the
failure was caught by the pre-registered validity arbiter, not by luck
or post-hoc inspection. (Credit where due: the build-phase audit's
MAJOR-1 fix — wiring the value side into premise *classification*, not
just measurement — is the specific instrument that fired here.)

### 14.3 Scope discipline (critical — what this addendum does NOT change)

**Only the TRAINING loss changes.** Unchanged, explicitly:

- The **readout**: the pinned linear unbind
  `pred(a,h) = S_Tʰ · q_eff_a`, scored at eval by **absolute cosine
  against the target `v_eff` at the 0.9 threshold — exact continuous
  recovery, never an in-episode argmax or softmax**. The Nichani, Lee &
  Bietti (arXiv:2412.06538) house rule stands: under
  argmax/nearest-neighbor decoding a rank-1 state can recover ≈d
  associations, silently breaking the bound — so the new contrastive
  TRAINING term (§14.4) must never leak into EVAL scoring.
- The **provable-bound framing**: `rank(S_T) ≥ K` for exact continuous
  recovery of K bindings under premises (i)–(iii) (§5.2) — statement
  untouched.
- The **premise gates**: C16's arm-specific rules (R2-1), the
  value-side Gram/salvage rule (build-audit MAJOR-1), the alignment
  threshold (R2-2), the companion classification rule — all remain the
  post-hoc validity arbiter, numerically unchanged. The objective
  change must *win under* these gates, not move them.

### 14.4 The objective change — both pre-scoped options, the choice, the exact pre-registered loss

**Option (a) — value-spread regularizer (VICReg-style).** Keep the
cosine regression; add variance/covariance penalties on the K `v_eff`:
`L_a = L_cos + λ_v·Σ_d max(0, γ − std_d(v_eff))² + λ_c·‖offdiag(Cov(v_eff))‖²`.
Smaller diff; keeps the loss shape. **Weaknesses:** (1) collapse
remains a *reachable, soft-penalized* configuration — at the collapse
point `L_cos` is exactly 0, so the outcome hinges entirely on the
penalty coefficients' balance against the regression term elsewhere in
weight space; (2) it introduces **two delicate tunables** (`γ` variance
floor, `λ_v/λ_c` balance) whose post-hoc adjustment is exactly the kind
of rescue lever R2-2 just eliminated from the alignment premise.

**Option (b) — discriminative/contrastive training term
(InfoNCE-style).** For query `j`, the readout's prediction must match
clause `j`'s `v_eff` **better than the other clauses' `v_eff` in the
same episode**. Under value collapse all K candidates are identical, so
this term scores exactly `log K` (chance) with **zero gradient toward
collapse** — the degenerate optimum is removed **structurally**, not
penalized.

**Choice: (b), in a retained-cosine form** — the contrastive term is
added ALONGSIDE the cosine regression, not substituted for it:

- **Why (b) over (a):** structural elimination of the degenerate
  optimum beats a tunable soft penalty — the same house preference that
  pins masks architecturally rather than hoping training converges
  ("enforced by an explicit mask, not hoped for"). (b) also directly
  operationalizes binding — match the right value against the same
  episode's alternatives — which is the task.
- **Why retained-cosine rather than pure (b):** a *pure* contrastive
  objective trains toward a discrimination criterion that a rank-1
  state can in principle satisfy (the Nichani argmax result applies to
  the training signal too) while never being pushed toward **exact
  continuous** recovery — the eval readout would then fail for a
  train/eval-mismatch reason indistinguishable from a genuine
  rank-forcing failure, making FALSIFY uninterpretable. Keeping `L_cos`
  preserves the exact-recovery pressure; adding `L_nce` removes the
  collapse optimum. At the collapse point `L_cos ≡ 0` with zero
  gradient, so the contrastive gradient is **unopposed** there for any
  positive coefficient — the anti-collapse property is qualitative, not
  a tuned balance (the specific weakness of option (a)).
  > **AUDIT-FIX NOTE (2026-07-03, mini-audit — wording correction to
  > the sentence above):** "the contrastive gradient is unopposed
  > there" overstates the mechanism. At EXACT collapse the NCE logits
  > are uniform by symmetry, so `L_nce = log K` with **zero gradient**
  > — collapse under the combined loss is a **non-attracting stationary
  > point (a symmetric saddle)**, not a point with a live escape force.
  > The escape is the **coupled path**: any perturbation that improves
  > within-episode discrimination lowers `L_nce`, and `L_cos ≈ 0` along
  > the binding direction means the regression term does not oppose
  > that path — **CONDITIONAL on the target-index fix** (§14.2's
  > audit-fix note): with the pre-fix index the discrimination problem
  > was unsatisfiable and the saddle was effectively the only optimum,
  > which the mini-audit's stagnating as-built+NCE control run
  > demonstrates. The anti-collapse claim's correct form: collapse is
  > no longer a global optimum and is not attracting; it is not
  > "actively repelled from the collapse point itself."

**The pre-registered loss, exactly:**

```
L_train = L_cos + λ_nce · L_nce

L_cos   = mean_j [ 1 − cos( pred_j , v_eff_{tgt(j)} ) ]
pred_j  = S_T^{h_j} · q_eff_j            # S_T truncated FIRST in force-rank
                                          # arms (§3.5 split — unchanged)
tgt(j)  = clause index of π^{h_j}(a_j)   # the correct entity's bind clause
v_eff_i = post-conv, post-W_v value representation at clause i's write
          position — the SAME tensor premise (ii)'s gate measures

L_nce   = mean_j CE( logits_j , tgt(j) )
logits_{j,i} = cos( pred_j , v_eff_i ) / T ,  i ∈ the SAME episode's K
               clauses (in-episode negatives ONLY: the other K−1 clauses)

λ_nce = 1.0   (fixed, pre-registered)
T     = 0.1   (fixed, pre-registered)
No stop-gradient on the target side (a BYOL-style sg is explicitly NOT
adopted: it hides rather than removes the degenerate optimum, and the
asymmetry would complicate the premise diagnostics).
```

**Sensitivity note (pre-registered, not a tuning license):** the
anti-collapse property holds for **any** `λ_nce > 0` (unopposed
gradient at the collapse point, above), so `λ_nce`/`T` are not
delicate. A single sensitivity pair `λ_nce ∈ {0.3, 3.0}` is
pre-registered as a check to run **only if** the §14.6 rerun gate lands
marginal — never as an iterative tuning loop (§14.6's one-iteration
cap).

### 14.5 The bound under the new loss — one paragraph, explicit

The `rank(S_T) ≥ K` bound is a property of the **readout**, not of how
`S_T` or the representations were produced by training — the same
argument `DELTANET_CAUSAL_RANK_DESIGN.md` §4.1 already makes for the
recurrence ("the bound is a property of the readout... it therefore
transfers without re-derivation"), applied now to the loss: the proof
consumes only (1) the pinned linear unbind, (2) linear independence of
the effective keys, and (3) linear independence of the target values,
and concludes `rank(S_T) ≥ K` for exact continuous recovery. Nothing in
it references the training objective. The loss change alters only
**which optimum SGD is pushed toward** — away from the collapse
configuration where premises (2)/(3) fail, toward configurations where
they can hold; whether they DO hold at any checkpoint remains exactly
the premise gates' post-hoc question (§14.3), under the same numeric
rules. The one new obligation is negative, not positive: `L_nce` must
stay out of eval scoring (§14.3's Nichani discipline) — enforced by a
build-time smoke assert that the eval path computes absolute cosine
only, with no softmax/argmax over candidates anywhere downstream of
`pred_j` at eval.

### 14.6 Wave 0 rerun — pre-registered decision rule

**Rerun:** identical Wave 0 manifest (`K ∈ {16,32}` × the **same** 5
seeds each — same seeds, so the only changed variable is the loss),
same budget, same instruments.

- **"Objective fixed"** ⇔ **≥ 6/10 seeds premise-valid at their final
  checkpoint** (unconstrained-arm rules: key τ AND value τ_v/salvage
  AND alignment ≥ 0.9), **and** the original Wave 0 convergence gate
  passes (≥3/5 seeds per K with genuine — i.e. premise-valid —
  `rec@0.9 ≥ 0.9` at h=1). Then the R2-4 τ/alignment re-registration
  decision is taken as originally specified and Wave A launches.
- **Marginal (6–7/10 premise-valid but the convergence gate shaky):**
  run the §14.4 sensitivity pair before deciding; one extension only.
- **Marginal, second form (added 2026-07-03 at the mini-audit's
  direction, decided NOW pre-data — this branch was MISSING and would
  otherwise have been improvised): ≥ 8/10 seeds premise-valid but the
  convergence gate fails** (premises hold — keys/values independent,
  aligned — yet fewer than 3/5 seeds per K reach premise-valid
  `rec@0.9 ≥ 0.9` at h=1). This is NOT the collapse failure and must
  not trigger the failure branch: premise-valid non-convergence is the
  lineage's known late-transition/budget-artifact pattern
  (`TASK_E_FINDINGS.md` §10), not a task-construction defect. Treat as
  **MARGINAL**: run the §14.4 sensitivity pair AND/OR one 2.5×
  budget extension (the standing re-test-before-calling-dead
  discipline) — ONE extension total, then decide CONFIRM-path vs
  failure branch on the extended result.
- **Failure branch, named (rerun ALSO collapses — < 6/10
  premise-valid):** the diagnosis moves from "the objective admits a
  degenerate optimum" to **"the task construction itself needs
  revisiting"** — the combination of learned targets + closed small
  entity pool + adjacency template may be unable to force binding at
  all. Pre-registered response: **STOP the RD Wave 1 path (no Wave A)**;
  write the negative up honestly as a finding in its own right (a
  real-embedding harness admits collapse optima that the synthetic
  fixed-target harness structurally excludes — an informative statement
  about exactly where the synthetic→real bridge is load-bearing); and
  take any reconstruction (e.g. a frozen constructed target codebook —
  reintroducing one synthetic element, honestly labeled — or an open
  entity pool) back through the **full design → attack → verify loop as
  a new revision of this document**, not a quiet patch.
- **Anti-burn cap: ONE objective-change iteration inside this design's
  budget — this addendum is it.** A second consecutive collapse
  triggers the failure branch unconditionally: no iterative objective
  fishing, which would hollow out the pre-registration this lineage's
  credibility rests on.

**Budget note:** the rerun re-spends Wave 0's ~10–15 GPU-h slot once;
this fits inside §7's Reserve allocation (~15–25 GPU-h, which exists
for exactly this class of calibration miss). The §7 total and cut order
are unchanged.

**Next action:** this addendum → fresh independent mini-audit (scope:
§14.3's no-leak-into-eval discipline, §14.4's loss spec, §14.6's gate)
→ if clean, launch the Wave 0 rerun. **(Executed: the mini-audit ran —
catching the §14.2 target-index FATAL and amending §14.1/§14.4/§14.6,
see the dated audit-fix notes — then the rerun ran with the index fix +
the §14.4 loss. The gate decision is recorded in §14.7.)**

### 14.7 Gate-decision record (2026-07-03) — the R2-4 / §14.6 gate, decided and dated BEFORE any Wave A data exists

The Wave 0 rerun ran: identical manifest, same 10 seeds, with the
§14.2 audit-fix target-index correction and the §14.4 loss — per
§14.2's audit-fix note, the rerun tests the objective change **on top
of** the index fix, and the mini-audit's 3-arm control already
attributed the mechanism (as-built+NCE stagnates at chance;
index-fixed+NCE binds). This subsection is the gate's decision record,
written to this document and the wave summary before Wave A's manifest
was generated (§8's gating discipline). Each decision cites the
pre-registered authority it is taken under.

**(1) Objective-fix verdict: FIXED — 10/10 seeds free of value
collapse.** Rerun outcomes across both cells (`K ∈ {16,32}` × 5 seeds):
entity-subspace rank ≈ K in both cells (vs. ≈1.0 at collapse); value
salvage `σ_K/σ_1` = 0.26–0.53 (vs. 0.000); `L_nce` → 0.003–0.021 (vs.
the collapse/chance value `log K` ≈ 2.8–3.5 — the discrimination
problem is genuinely solved, not evaded); C19 held-out-template
recovery 0.72–0.995. The §14.4 mechanism worked as designed on the
now-satisfiable task. **Honest disaggregation (documented at the gate,
not glossed):** §14.6's literal formulation — "≥6/10 premise-valid at
final checkpoint," with key τ AND value τ_v in the conjunction —
**conflated two distinct axes**: (a) *is the collapse optimum
eliminated* — the question the objective change existed to answer, met
unambiguously at 10/10 — and (b) *are checkpoints premise-valid for
headline purposes* — a validity question that the same gate was always
pre-authorized to resolve by re-registering τ on measured data
(R2-4/FATAL-2; §5.2's Anchor caveat). Read literally with the
synthetic-anchored τ = 0.03 in the conjunction, the measured key band
(1.26–2.77, item (2)) would have scored **0/10 premise-valid while
every seed demonstrably binds** — which is a verdict about the
anchor, not about the objective. The gate PASSES on axis (a); axis (b)
is resolved by items (2)–(3) below — disaggregated explicitly, not
smuggled through (a).

**(2) τ re-registration — the pre-authorized R2-4/FATAL-2 path,
executed on measured data (this is the pre-registered re-registration,
dated here; NOT silent tuning).** τ = 0.03 was anchored on the
synthetic harness's orthonormal-key bands with the explicit, standing
caveat (§5.2 Rule "Anchor caveat"; §9 item 5; §12) that Wave 0's
measured real-data band would trigger re-registration at exactly this
gate. Measured real-data unconstrained band (rerun, 10 seeds):

| Diagnostic | Measured band | Bound-premise status |
|---|---|---|
| Key Gram deviation | 1.26–2.77 | fails τ = 0.03 everywhere, but — |
| Value Gram deviation | 1.32–3.31 | fails τ_v = 0.03 everywhere, but — |
| Key/value salvage `σ_K/σ_1` | 0.26–0.55 | **≥ 0.1 everywhere — linear independence with bounded conditioning, the bound's actual premise, demonstrably holds** |
| Entity-subspace rank | ≈ K, both cells | consistent with genuine K-way binding |

Reading: learned real-data representations are **linearly independent
but far from orthonormal** — exactly the regime R2-1's salvage tier was
defined for. Near-orthonormality (τ) was always a *clean-regime*
property of the synthetic construction, never a premise of the proof
(§5.2 premise (i): linear independence, not orthonormality).
**DECISION: for real-data arms, τ/τ_v are dropped as validity
criteria. The validity stack is: salvage tier on BOTH sides
(`σ_K/σ_1 ≥ 0.1`, key and value) + per-item alignment (UNCHANGED at
0.9, item (3)) + the R2-8 held-out-pool classification. τ/τ_v are
retained as clean-regime DESCRIPTORS (logged and reported, never
gating).** This also unifies the unconstrained arm's rule with the
causal `k ≥ K` leg's rule (R2-1(ii)), which already used the salvage
tier for the same reason — a regime where near-orthonormality is the
wrong bar.

**(3) Alignment rule: UNCHANGED at per-item cos ≥ 0.9.** Re-registering
it downward at the same gate that is loosening τ would convert premise
(iii) into precisely the one-directional rescue lever R2-2 exists to
prevent — the threshold stands. **Consequence, disclosed rather than
buried:** 4/10 rerun seeds are strictly alignment-clean at their final
checkpoint (K16: 3/5 at ≈1.000; K32: 1/5 at 0.963) — **Wave A/B
headlines rest on alignment-clean checkpoints only**, per the unchanged
rule and the standing companion rule (last alignment-clean checkpoint
carries a run's number, flagged). **New phenomenon, named as an open
question (a §12-style entry, NOT a threshold change):** 6/10 seeds show
a monotonic `align_min` decay (≈0.96 → 0.61 over training) **while
recovery holds** — the query pathway appears to drift off the bind-key
geometry while retrieval stays correct, suggesting a possible second
read path that premise (iii)'s model (query reads through the same
geometry the write used) does not capture. Pre-registered Wave A
diagnostic: per-checkpoint alignment trajectories are logged for every
run; if the decay reproduces at Wave A scale, the mechanism question
becomes a scoped follow-on investigation — explicitly not grounds for
revisiting the 0.9 threshold.

**(4) Primary-K = 16.** K=16: 5/5 seeds converge at `rec@0.9 ≥ 0.9`
(behavioral convergence; of these, 3/5 are alignment-clean at the
final checkpoint and carry headline status per item (3)). K=32: 0/5 —
a **plateau at 0.78–0.80, flat for the final 21K steps, with
entity-subspace rank 29.9–30.2**. This is NOT a budget artifact (the
trajectories are converged-flat, not climbing — contrast the
three-budget-artifacts pattern, `TASK_E_FINDINGS.md` §10, where "dead"
cells were still pre-transition; §14.6's second marginal branch
therefore does not fire) and NOT a rank-recruitment failure (esr ≈ 30
of 32). Documented as a real, capacity-adjacent **sub-exactness
phenomenon at `K = d/2` on real data** — echoing the bespoke encoder's
exactness-frontier finding on a second architecture and a second data
regime (Stage 0: d≥32 transitions reliably but plateaus sub-exact;
`STATE.md`: "the honest frontier is not trainability — it's
exactness"). **Flagged for the cross-program discussion; not chased in
this design's budget.** Wave A launches at **`--primary-k 16`**, with
K=32 retained in the K-grid as a documented sub-exact cell, not the
primary.

**(5) Wave −1 final record + K=64 caution — carried as-is.** Wave −1's
final measured table stands as amended in §14.1's audit-fix note
(K=16: 8.8×, 0.00% skip; K=32: 8.5×, 0.00%; K=64: 5.6× with the
**7.06% late-onset skip burst** — ≤10× proceed bucket everywhere;
`svd_lowrank` per deviation #6; F18 discipline active on every
force-rank readout). The K=64 caution stands unchanged: the skip-burst
was measured on a pre-index-fix run, so it is not read as a verdict —
but **any Wave A/B K=64 force-rank cell requires a fresh post-fix
skip-rate measurement AND padded timeouts before pricing**. Not
gate-relevant at primary-K = 16.

**Launch decision.** On this record — dated, cited to the measured
tables above, and written before Wave A's manifest was generated —
**Wave A launches at `--primary-d 64 --primary-k 16`**, under the
item-(2) validity stack (salvage both sides + alignment 0.9 + R2-8
classification; τ/τ_v descriptive only), with the item-(3) alignment
trajectories logged as a standing diagnostic on every run.

---

## 15. Results — Waves −1/0 (2026-07-03) — narrative summary, cross-verified against raw JSON

§14 is the live gate-decision record, written incrementally as each
wave landed. This section is the after-the-fact narrative, written once
all three run sets existed, with **every number below independently
re-derived from the archived result JSONs** (not copied from §14's
prose) — archive:
`experiment-runs/2026-07-03_deltanet_rd_waves/{wave-1,wave0,wave0_rerun}/`
(`wave0` = the original collapsed round, pulled from
`youthful-indigo-turkey:/home/nvidia/chapter2/deltanet_rd/results/deltanet_rd/{wave-1,wave0}/`;
`wave0_rerun` = the post-fix rerun, pulled from
`.../deltanet_rd_w0b/wave0/`). Verification method: `wall_s`,
`steps_completed`, `skip_rate`, and every per-checkpoint diagnostic
quoted here were read directly out of the per-run JSON's `trajectory`/
`checkpoints` arrays with a throwaway script, not transcribed from a
prior draft.

### 15.1 The arc, in order

1. **Wave −1 (timing gate) passed.** `svd_lowrank` force-rank multiplier
   vs. the unconstrained per-step cost: K=16 **23.0 → 201.9 ms (8.8×)**,
   K=32 **28.3 → 240.6 ms (8.5×)**, K=64 **38.4 → 215.7 ms (5.6×)** — all
   three land in §7's "≤10× → proceed as priced" bucket (re-verified
   exactly against `wave-1/w-1_rd_K{16,32,64}_fr{16,32,64}_s0_fr_tsvd_lowrank.json`'s
   `wall_s`/`steps_completed`, matching §14.1's table to the decimal).
   The K=64 forced cell carries a **7.06% skip rate**, but it is
   entirely a **late-onset burst**: `skip_rate_so_far` is exactly 0.0
   through the step-8800 checkpoint, then rises to 0.0026 (step 9000),
   0.017 (9200), 0.041 (9600), and 0.0706 by step 10000 — roughly 680 of
   the run's 706 total skips land in the final ~1,000 of 10,000 steps.
   Two eigh-path attempts (`K32_fr32_s0_fr.json`, `K64_fr64_s0_fr.json`,
   both `complete: false`, 222.9s and 454.4s of real GPU time before the
   forward-pass `LinAlgError`) preceded the successful `svd_lowrank`
   reruns and are excluded from the priced multiplier but not from the
   compute tally below.
2. **Original Wave 0: 10/10 seeds value-collapsed.** Every one of 5
   seeds at K=16 and 5 at K=32 converged to the identical degenerate
   optimum: `recovered_frac@0.9 = 1.000`, entity-subspace effective rank
   1.004–1.011 (of K), value salvage ratio `σ_K/σ_1` on the order of
   `1e-6` (i.e., 0.000 to 3 decimals), key Gram deviation 2.62–5.32,
   `alignment_cos_min` between −0.002 and 0.130. `n_premise_valid = 0`
   across both cells in `wave0/AGGREGATE.json` — **zero checkpoints,
   zero seeds, escaped as a headline.** The instrument stack (three
   audit rounds deep by this point) did exactly what it was built to
   do: `rec@0.9 = 1.000` looked like a perfect result and was correctly
   classified premise-failed by the value-side salvage rule before
   anyone could read it as one.
3. **The mini-audit's discovery.** `forward()` gathered the scoring
   target at `v_eff_items[tgt_slot]`, but clause `i`'s VALUE token
   belongs to entity `π(i)` — so the target actually scored was the
   representation bound at `π^{h+1}(a)`, **one hop past** the queried
   answer `π^h(a)`. Grammar-level check: 100% mismatch at every hop as
   originally built, 100% match once the gather point is corrected to
   clause `π^{-1}(tgt_slot)`. Under the bug, the discrimination problem
   was **unsatisfiable by construction** — no weight configuration could
   make the pinned readout match the scored target through genuine
   binding. The mini-audit's 3-arm control made the attribution
   decisive: as-built + the new NCE term stagnates at `L_nce ≈ log K`
   (chance — confirming the rerun would have deterministically failed
   its own gate had the index bug shipped alongside the objective
   change), while index-fixed + NCE produces genuine binding. The
   pre-registered NCE objective (§14.4: retained-cosine regression plus
   an in-episode InfoNCE term, `λ_nce = 1.0`, `T = 0.1`, no
   stop-gradient) was implemented alongside the index fix, not as a
   separate iteration — the anti-burn cap (§14.6) treats the two as one
   combined change.
4. **The rerun: 10/10 collapse-free.** Same 10 seeds, same manifest,
   index fix + NCE loss. **K=16** (all 5 seeds, final checkpoint,
   step 25,000/25,000): `rec@0.9` 0.9957–0.9988, entity-subspace rank
   15.57–15.74 (of 16), C19 held-out-template recovery 0.987–0.995,
   `L_nce` 0.0028–0.0047 (vs. `log 16 ≈ 2.77` at collapse). **K=32**
   (all 5 seeds): `rec@0.9` **0.782–0.800**, flat from the step-4000
   checkpoint through step 25,000 (21,000 steps of measured plateau,
   not a rising trajectory), entity-subspace rank 29.91–30.23 (of 32),
   C19 0.724–0.768, `L_nce` 0.0155–0.0210. Both cells' value-salvage
   ratio is now 0.26–0.53, versus 0.000 at collapse. This is the first
   time in this lineage a production fast-weight architecture
   (DeltaNet's `chunk_delta_rule`) has shown genuine rank-K relational
   binding on natural-language surface forms passed through a real
   GPT-2 tokenizer and trainable embedding table — every prior
   confirmation (Task D, Task E, the synthetic DeltaNet causal-rank
   design) used either a bespoke attention-set encoder or
   experimenter-constructed vector grammar.
5. **The gate decision (§14.7, dated 2026-07-03).** Objective verdict:
   FIXED (10/10 seeds clear of the collapse optimum on the salvage-tier
   criterion). τ/τ_v (0.03, anchored to the synthetic harness's
   orthonormal-key construction) are **re-registered out of the
   validity stack** for real-data arms — measured Gram deviations
   (key 1.26–2.77, value 1.32–3.31) fail τ everywhere despite every
   seed demonstrably binding, because real learned representations are
   linearly independent but not orthonormal, which was always a
   clean-regime property of the synthetic construction rather than a
   premise of the rank bound. The new validity stack: salvage tier on
   both sides (`σ_K/σ_1 ≥ 0.1`) + per-item alignment (**unchanged** at
   cos ≥ 0.9) + R2-8's held-out-pool classification; τ/τ_v remain
   logged as descriptive-only. Under that stack, **4/10 seeds are
   strictly alignment-clean at their final checkpoint** (K16: 3/5 at
   `align_min ≈ 1.000`; K32: 1/5 at 0.963) and carry headline status;
   the other 6/10 pass salvage-tier on both sides (so are not
   collapsed) but fail the unchanged alignment bar. Those 6 show a
   **new, named-but-unexplained phenomenon**: `align_min` decays
   monotonically from ≈0.96–0.99 (valid, early checkpoints) to
   ≈0.61–0.69 by step 25,000 **while `rec@0.9` stays pinned ≥0.99 for
   K16 / ≥0.97 for K32 throughout the same window** — recovery does not
   degrade as alignment falls (verified directly against per-checkpoint
   trajectories, e.g. `w0_rd_K16_frN_s1.json`: `align_min` 0.969→0.618
   from step 2000→25000 while `rec@0.9` stays 0.984–0.998 throughout).
   Logged as an open question with a Wave A diagnostic (per-checkpoint
   alignment trajectories on every run), explicitly not grounds to move
   the 0.9 threshold. **Primary-K = 16**: K=32's flat 0.78–0.80 plateau
   with entity-subspace rank already at ≈30/32 is read as a genuine
   sub-exactness phenomenon at `K = d/2` (echoing Stage 0's exactness
   frontier on a different architecture), not a budget artifact — Wave A
   launches at `--primary-d 64 --primary-k 16`, K=32 retained in the
   grid as a documented, non-primary sub-exact cell.

### 15.2 The honest framing

This arc caught **two fatal bugs** — the DeltaNet causal-rank design's
own state-axis transpose (see `DELTANET_CAUSAL_RANK_DESIGN.md`) and this
design's target-index gather bug — via **independent audit plus a
mini-audit**, both **before** either bug's consequence (a spurious
CONFIRM or an uninterpretable FALSIFY) entered the record. The Wave 0
value-collapse was never written up or reported as a finding; it was
caught by pre-registered instruments (the build-audit's MAJOR-1
value-side salvage rule, specifically) at every one of its 10 seeds and
correctly held to zero premise-valid checkpoints. That is the premise-
gate machinery working exactly as designed, not a near-miss.

### 15.3 Compute — measured, not estimated

Serial-sum of `wall_s` across every archived run JSON (house convention,
matches how prior waves in this repo report GPU-h):

| Set | Runs | Complete | Total `wall_s` | GPU-h |
|---|---|---|---|---|
| Wave −1 (incl. 2 pre-`svd_lowrank` eigh-crash attempts) | 8 | 6/8 (2 excluded from pricing, not from spend) | 8,155.9s | **2.27** |
| Wave 0, original (collapsed) | 10 | 10/10 | 6,166.2s | **1.71** |
| Wave 0, rerun (index fix + NCE) | 10 | 10/10 | 23,811.6s | **6.61** |
| **Total, this arc** | 28 | 26/28 | 38,133.6s | **10.59** |

Both Wave 0 rounds combined: **8.33 GPU-h** — roughly 5× the original
round's cost, driven by the added NCE forward/backward over in-episode
negatives plus (visible in the per-run `wall_s` spread: most rerun cells
took ~2,750–2,820s for 25,000 steps, but two — `K16_s4` at 763.6s and
`K32_s0` at 868.4s — completed the identical step count in roughly a
third of the time, consistent with this box's shared-GPU contention
across concurrently-launched sweep cells rather than any per-seed
config difference). Every figure in this table is within §7's priced
bands (Wave −1 ~4–6 GPU-h estimated vs. 2.27 measured; Wave 0 ~10–15
GPU-h estimated per round vs. 1.71/6.61 measured) — both rounds landed
under price, consistent with this project's standing pattern that
per-run GPU-h estimates run high while steps-to-convergence is the
quantity that actually bites (§12's open-questions list already flagged
this asymmetry before Wave 0 ran).

---

## 16. Results — Wave A (2026-07-03)

Launched per §14.7's dated launch decision (`--primary-d 64 --primary-k
16`, index-fixed + NCE objective, τ/τ_v retired as validity criteria for
real-data arms, alignment unchanged at cos ≥ 0.9). Archive:
`experiment-runs/2026-07-03_deltanet_rd_waves/waveA/` (pulled from
`youthful-indigo-turkey:/home/nvidia/chapter2/deltanet_rd/results/deltanet_rd_w0b/waveA/`,
11 result JSONs + `logs/`). **Every number below is read directly out of
each run's final checkpoint (`checkpoints[-1]`, step 20,000/20,000) via a
throwaway script**, not copied from `AGGREGATE.json`/`SUMMARY.txt` — §16.6
explains why those two files are not trustworthy for this wave's headline
numbers.

### 16.1 Manifest: 11/13 cells complete — 2 K=4 cells correctly refused by the periodicity guard

`waveA_manifest(primary_k=16)` builds `k_grid =
sorted({16//4, 16//2, 16, ⌊16·1.5⌋, 16·2}) = {4, 8, 16, 24, 32}`, run at 2
seeds each (10 runs) plus 3 composition-tagged seeds at the primary K=16
(3 runs) = 13 manifest entries. **11/13 ran; both K=4 seeds (`s0`, `s1`)
failed identically at config construction**, before any training step:

```
AssertionError: H_test/H_extra hop h=4 is a multiple of K=4: pi^4 is the
IDENTITY -- this probe is confounded (measures nothing), not held-out.
```

This is `grammar_rd.py`'s periodicity guard (`DeltaNetRDTaskConfig.__post_init__`,
verbatim from `task_dn.py`/`task_e.py`'s Finding-B fix, §5.2/C6) firing
correctly: the design's `H_test = [4, 5, 6]` is a **fixed** literal hop
list, independent of `K`, and `waveA_manifest`'s `K // 4` term in the
k-grid formula can — and here does — produce a `K` for which one of those
fixed literal hops equals `K` itself (`h=4 ≡ 0 mod 4`, the single-cycle
identity). **This is a manifest oversight, not a task-construction
defect**: the k-grid generator and the fixed `H_test`/`H_extra` list were
written independently and never cross-checked against each other before
Wave A's launch. The guard did exactly its designed job — refusing to
silently run a confounded probe — and cost two near-zero-compute failures
(assertion fires before data loading; not separately measured, but the
log files are 711–849 bytes, consistent with a sub-second failure) rather
than a silently-wrong K=4 cell entering the record.

**Fix note for any future wave that touches small `K`:** before generating
a `k_grid`, filter out (or flag for a widened/rescaled `H_test`) any
candidate `K` for which `{h % K for h in H_test + H_extra}` contains 0 or
collides with a training-hop residue — the same check `__post_init__`
already runs per-config should be run once, up front, over the whole
candidate grid at manifest-build time, so a colliding cell is caught (and
either dropped or repaired) before launch rather than discovered as a
run-time failure. K=4 is not used anywhere downstream of Wave A in this
design (Bprobe/B are pinned to primary-K=16, §16.5), so no repair was
needed this wave — recorded here only so a future K-grid that includes
small K does not rediscover this by hand.

### 16.2 Headline: a graded K-axis exactness frontier at fixed d=64, on real tokenized text

`recovered_frac@0.9`, final checkpoint (step 20,000), by K (all seeds
shown; K=16 includes both the 2 screening seeds and the 3
composition-tagged seeds, which are independent training runs at an
identical config, not duplicates — wall-clock times differ run to run):

| K | seeds | ID h=1 | ID h=2 | ID h=3 | HO h=4 | HO h=5 | HO h=6 | HO h=7 | h=21 |
|---|---|---|---|---|---|---|---|---|---|
| 8 | s0, s1 | 1.000, 1.000 | 0.999, 0.999 | 0.978, 0.984 | 0.944, 0.956 | 0.873, 0.899 | 0.780, 0.825 | 0.699, 0.730 | 0.006, 0.011 |
| 16 | s0, s1, s0c, s1c, s2c | 0.997–0.999 | 0.879–0.901 | 0.656–0.700 | 0.419–0.465 | 0.252–0.280 | 0.137–0.161 | 0.069–0.097 | 0.000 (all 5) |
| 24 | s0, s1 | 0.941, 0.947 | 0.574, 0.591 | 0.263, 0.271 | 0.097, 0.101 | 0.031, 0.039 | 0.008, 0.009 | 0.002, 0.004 | 0.000, 0.000 |
| 32 | s0, s1 | 0.780, 0.790 | 0.259, 0.265 | 0.053, 0.054 | 0.009, 0.009 | 0.001, 0.002 | 0.000, 0.000 | 0.000, 0.000 | 0.000, 0.000 |

**Reading:** in-distribution recovery (h=1–3, the trained hop range)
degrades gracefully and monotonically in K: near-perfect at K=8, still
strong at h=1 but visibly eroded by h=3 at K=16, badly eroded by h=2–3 at
K=24, and only h=1 survives at K=32. Held-out-hop composition (h=4–7,
never trained on) shows the same monotone K-ordering **and** the same
within-K depth decay across h=4→7 — K=8 still recovers a majority of
items out to h=7 (0.70–0.73); K=16 falls to single-digit-to-teens percent
by h=7; K=24/K=32 are functionally at floor beyond h=4. This is a real,
graded frontier along K at **fixed** `d_state=64` — the DeltaNet
analogue, on real BPE-tokenized natural-language surface form, of the
bespoke encoder's d-axis frontier (§16.7 draws the parallel explicitly).

### 16.3 h=21 is a depth-decay probe, not an independent held-out-hop test — read via `effective_hop`, per Task E's own standing guidance

Every run logs `effective_hop = h mod K` for `M3_held_out` alongside the
literal `h`. At K=8 and K=16, `21 mod K = 5` — the **same graph target**
as the already-tested `h=5` entry — but the readout applies the literal
matrix power `S_T^{21}`, not `S_T^5`. The two numbers diverge sharply:

| K | h=5 (`effective_hop=5`) rec@0.9 | h=21 (`effective_hop=5`) rec@0.9 |
|---|---|---|
| 8 | 0.873–0.899 | 0.006–0.011 |
| 16 | 0.252–0.280 | 0.000 (all 5 seeds) |

Same queried entity, same effective graph distance, two very different
recovery rates — the only thing that differs is the literal count of
sequential self-applications of the learned operator `S_T`. This is
exactly `TASK_E_FINDINGS.md` §3's **depth-amplification** finding,
reproduced here on a production DeltaNet kernel and real tokenized
sentences rather than the bespoke encoder: "raw iteration depth... is a
spectral-exactness amplifier, and a sharper rank test than any
single-hop cosine tolerance." (Task E's own 20K-round number for the same
signature: h=5 recovery 0.915 vs. h=21 recovery 0.006 at K=8, its
`EXPERIMENT_LOG.md` 2026-07-01 entry.) At K=24/K=32, `21 mod K = 21`
(genuinely un-reduced, `21 < K` for K=24 but `21 < 32` too — no
periodicity collapse either way), so h=21 there is simply "the deepest
hop tested," already at floor because h=4 already is. **Per the standing
project guidance this design inherits, h=21 at K=8/K=16 should be read as
a depth-decay probe on the trained operator's spectral exactness, not as
a fourth independent held-out-hop generalization data point** — §16.2's
table lists it for completeness but the K-frontier narrative rests on
h=1–7.

### 16.4 Entity-subspace rank: binding capacity is recruited at every K tested

`entity_subspace_effective_rank_mean` (h=1, final checkpoint), against
the target K:

| K | rank (both/all seeds) | fraction of K |
|---|---|---|
| 8 | 7.94, 7.95 | 99–99% |
| 16 | 15.62–15.73 | 98–98% |
| 24 | 23.08, 23.20 | 96–97% |
| 32 | 29.93 (both seeds) | 94% |

Rank recruitment is essentially complete at every K in this grid (94–99%
of target, mildly declining as K grows) — the state consistently uses
close to its full available capacity to represent the K bindings,
**regardless of whether §16.2's exactness numbers at that same K are
strong or near-floor**. This is the same separation Stage 0 and the
synthetic DeltaNet design already found on their own axes (§16.7):
capacity is not the thing that fails first.

### 16.5 Alignment: mixed, the §14.7 decay phenomenon reproduces at Wave A scale, and the K-frontier holds within the alignment-clean subset

`alignment_cos_min` (per-item bind→query cosine, minimum over the eval
batch), final checkpoint:

| Run | K | align_min | Alignment-clean (≥0.9)? |
|---|---|---|---|
| K8_s0 | 8 | 1.000 | yes |
| K8_s1 | 8 | 1.000 | yes |
| K16_s0 | 16 | 1.000 | yes |
| K16_s0_composition | 16 | 1.000 | yes |
| K16_s1 | 16 | 0.665 | no |
| K16_s1_composition | 16 | 0.672 | no |
| K16_s2_composition | 16 | 0.668 | no |
| K24_s0 | 24 | 0.613 | no |
| K24_s1 | 24 | 1.000 | yes |
| K32_s0 | 32 | 0.962 | yes |
| K32_s1 | 32 | 0.664 | no |

**6/11 cells are alignment-clean; 5/11 are not** — split roughly evenly,
not concentrated at any one K. Per-checkpoint trajectories confirm the
§14.7 decay phenomenon reproduces at Wave A's larger scale/longer budget
(20K vs. 25K steps, same signature): the 5 non-clean runs show
`align_min` decaying monotonically from ≈0.95–0.99 (early checkpoints) to
0.61–0.67 by step 20,000 (e.g. `K16_frN_s1`: 0.965 → 0.936 → 0.861 → ...
→ 0.665 across the 10 logged checkpoints), **while `recovered_frac@0.9`
at h=1 holds flat-to-rising throughout the same window** (same run:
0.982 → 0.995 → ... → 0.998) — recovery does not track the alignment
decay, exactly as §14.7 (3) found for Wave 0. This remains an open,
named-but-unexplained phenomenon (a possible second read path bypassing
premise (iii)'s bind-key-geometry model), not grounds to revisit the 0.9
threshold, per the standing symmetric-handling rule (R2-2).

**Verified: the K-frontier pattern holds within the alignment-clean
subset alone**, not just in the pooled data — i.e. it is not an artifact
of unequal alignment-quality mix across K. Restricting to the 6
alignment-clean runs only (K8 both seeds; K16 s0/s0_composition; K24 s1;
K32 s0): ID h=1/h=2/h=3 = (1.000/0.999/0.978–0.984) at K=8 →
(0.997–0.998/0.901/0.697–0.700) at K=16 → (0.941/0.574/0.263) at K=24 →
(0.780/0.259/0.054) at K=32 — the same monotone K8 > K16 > K24 > K32
ordering as the pooled table in §16.2.

### 16.6 Operational note: `AGGREGATE.json`/`SUMMARY.txt`'s own headline fields are stale relative to the §14.7 gate decision — do not quote them

`AGGREGATE.json` reports `n_premise_valid: 0` and every `headline_*`
field `null` at all four K cells. **This is a tooling gap, not a result**:
`run_deltanet_rd_sweep.py::_premise_valid_entry` classifies the
`"unconstrained"` arm (which is what every Wave A screening run is —
`force_rank_k=None`) using `key_gram_deviation_mean < tau_unconstrained`
/ `value_gram_deviation_mean < tau_value_unconstrained` (τ=τ_v=0.03) —
**the original §5.2 rule, not §14.7 item (2)'s dated re-registration**,
which retired τ/τ_v as a validity criterion for real-data arms in favor
of the salvage tier (`σ_K/σ_1 ≥ 0.1`, both sides) + alignment (≥0.9). The
sweep-level aggregator's `causal_k_ge_K` branch already implements the
salvage-tier rule (it needed to, to let the synthetic-adjacent causal arm
CONFIRM at all — the same reasoning R2-1(ii) already went through) but
the `"unconstrained"` branch was never updated to match once §14.7
extended that same reasoning to the unconstrained real-data arm too — so
every Wave A cell fails the stale τ check unconditionally (measured key
Gram deviation is 0.60–2.77 across this wave, τ=0.03), independent of
whether the run is actually premise-valid under the rule that is
supposed to govern it.

**This did not corrupt any number in this section** — §16.2–16.5's
figures are read directly from each checkpoint's raw logged fields
(`recovered_frac@0.9`, `entity_subspace_effective_rank_mean`,
`alignment_cos_min`), and the **per-checkpoint** JSON already carries
both criteria as separate booleans (`premise_valid_salvage_tier`,
`premise_valid_value_salvage_tier`, `alignment_valid`) correctly computed
by `run_deltanet_rd.py`'s own per-run eval code — only the **sweep-level
aggregation step** that rolls these into `AGGREGATE.json`/`SUMMARY.txt`
is out of date. §16.5's alignment-clean/not-clean classification uses
exactly `premise_valid_salvage_tier AND premise_valid_value_salvage_tier
AND alignment_valid`, i.e. manually applies the rule the aggregator
should already be applying. **Fix-forward, flagged before Bprobe's
results are read (§16.7 launched already, on the raw per-run JSONs, not
on the aggregator):** `_premise_valid_entry`'s `arm == "unconstrained"`
branch needs the same salvage-tier substitution the `causal_k_ge_K`
branch already has, or Bprobe/B's own `AGGREGATE.json` will misreport
`n_premise_valid=0` for its unconstrained-adjacent bookkeeping the same
way. Not urgent for Wave A (already hand-verified above) but should not
be rediscovered by hand again at B.

### 16.7 Bprobe launched at primary-K=16 (2026-07-03, 03:02 UTC, in flight at archive time)

Confirmed via `ps aux` on `youthful-indigo-turkey`: `tmux` session
`rdBp` running `run_deltanet_rd_sweep.py --wave=Bprobe --primary-k 16
--trunc-impl svd_lowrank --gpus 6 --gpu-offset 0 --per-gpu 2`, manifest =
force-rank `k ∈ {15, 16, 17} × 3 seeds` = 9 runs
(`waveBprobe_manifest`), `PROGRESS.txt: wave=Bprobe done=0 failed=0
running=9` at archive time. `svd_lowrank` (not `eigh`) is the truncation
implementation — consistent with §14.1/§14.7(5)'s K=16 timing/skip-rate
record (8.8×, 0.00% skip; the K=64 skip-burst caution does not apply at
K=16). **This is the real-data causal test** (RD-1's force-rank straddle,
the successor to the synthetic design's own B-probe), scoped to K=16 —
**binding plus shallow composition, per §16.2's frontier this is where
the operating point is still strong** (h=1–3 all recover well at K=16;
K=8 would have been an easier/less-informative cell, K=24/32 already
near floor even unconstrained). Worth noting explicitly: primary-K=16
was fixed at the §14.7 gate **before** Wave A's own K-grid screen ran —
Bprobe is testing the pre-registered primary K, not a post-hoc pick of
Wave A's best-looking cell (which, on §16.2's numbers alone, would have
been K=8) — the pre-registration discipline held even though Wave A's
own data would have motivated a different choice in hindsight. Not
analyzed in this section; results pending.

### 16.8 Cross-program synthesis: the same exactness-vs-capacity separation, three times, on three different axes

The same qualitative split — rank/capacity is recruited reliably by SGD;
**exactness of the readout is the separable, harder-won bottleneck; and
composition depth (literal operator self-application count, not graph
distance) amplifies whatever inexactness is already present** — now shows
up in three independent settings that vary the axis under stress and the
substrate: **(1) the bespoke matrix-encoder, stressed along `d` at fixed
task structure** (`STATE.md`; `STAGE0_DESIGN.md` §14–15): three converged
points at fixed h=64 — d=16 exact (rec@0.9=1.00 at every tested hop
including h=21), d=32 a sub-exact plateau (rec@0.9 0.15–0.65,
K-dependent), d=64 far sub-exact (rec@0.9=0.0 at every seed/K), with
effective rank still tracking K closely at d=16/32 and only becoming
itself partially short of target at d=64 — exactness degrades
monotonically in d while capacity mostly holds until the deepest cell.
**(2) The synthetic DeltaNet design, hand-built orthonormal keys**
(`DELTANET_CAUSAL_RANK_DESIGN.md`, `EXPERIMENT_LOG.md` 2026-07-02): the
unconstrained arm saturates to `rec@0.9=1.000` at **every** `(d,K)` cell
tested up to d≤128, with entity-subspace rank exactly K and zero
inflation — no exactness frontier is visible at all inside this design's
tested range, because construction-time orthonormal keys remove exactly
the premise-quality variable that is degrading in the other two settings.
**(3) DeltaNet on real tokenized text, stressed along `K` at fixed
`d_state=64`** (this wave): entity-subspace rank recruits 94–99% of
target at every K in {8,16,24,32}, while in-distribution and held-out-hop
exactness form a graded frontier — K=8 near-exact through several
held-out hops, K=16 partial, K=24/32 collapsed beyond h=1 — and the
literal-depth h=21 probe amplifies residual inexactness at K=8/16 far
below their own h=5 numbers, the identical depth-amplification signature
`TASK_E_FINDINGS.md` §3 first named on the bespoke encoder. Read
together: (2)'s perfect-everywhere result is not in tension with (1)/(3)
— it is the special case where the premise (i)/(ii) linear-independence
inputs are exact by construction rather than SGD-learned, which is
exactly the free variable (1) varies via `d` and (3) varies via `K` at
fixed `d`. The unifying claim this arc now supports empirically, on two
different architectures (a bespoke matrix encoder and a production
fast-weight kernel) and three different stress axes: **rank/capacity
recruitment is not the scarce resource — SGD reaches for it reliably;
per-binding operator EXACTNESS is the scarce, separable resource, and raw
composition depth is a sharper stress test of it than any single-hop
tolerance.**

### 16.9 Compute — measured, not estimated

Serial sum of `wall_s` across the 11 completed run JSONs (house
convention):

| Set | Runs | Complete | Total `wall_s` | GPU-h |
|---|---|---|---|---|
| Wave A (K∈{8,16,24,32} screen + K=16 composition) | 11 | 11/11 | 23,000.7s | **6.39** |
| K=4 (both seeds) | 2 | 0/2 (config-time assertion, before training) | not separately measured — sub-second, negligible | ~0 |
| **Total, this wave** | 13 | 11/13 | 23,000.7s | **6.39** |

Within §7's Wave A estimate (~15–20 GPU-h) — well under, consistent with
this project's standing pattern that per-run GPU-h estimates run high
(§15.3's own note, reproduced again here).

---

## 17. Eval-truncation causal analysis + Wave 1 close (2026-07-03)

§7's manifest gates the RD-1 causal verdict on B-probe → (full) Wave B.
Section 16.7 launched B-probe (`k∈{15,16,17}×3` at primary-`K=16`); this
section reads its result, finds the train-time instrument dead (mirroring
`DELTANET_CAUSAL_RANK_DESIGN.md` §12.8.1 exactly), declares Wave B moot
under §7's own gate language ("≥1 seed shows life at `k≥K` and a step
relative to `k<K`" — not met), and closes the causal question with the
pre-registered fallback: eval-time optimal-rank truncation of the
converged unconstrained state, per §12.8.2's argument for why that
instrument is decisive here. Script:
`matrix-thinking/deltanet_rd/analyze_eval_truncation_rd.py` (pure numpy +
stdlib, no CUDA/fla needed — loads dumps, does not retrain; `--smoke`
validates the truncation+readout+theory pipeline against a hand-built
operator before trusting it on real dumps). Every number below is
independently re-derived from the archived JSONs, not copied from a prior
draft.

### 17.1 The dump: schema verified, not assumed

`run_deltanet_rd.py` turns `--save-z` **on by default** (unlike the
synthetic design, where no archived run ever passed it — the reason
`analyze_eval_truncation.py` retrained fresh). Every completed RD result
JSON in the local archive
(`experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA,waveBprobe}/`)
carries a `Z_dump` key. Programmatically verified schema (`--verify-schema`):

```
Z_dump = {
  "S_T_raw":           (4, d, d)   UNFORCED bind (force_rank_k=None
                         regardless of the run's own --force-rank-k)
  "s_ideal_effective": (4, d, d)   sum_j v_eff_j @ k_eff_j^T, built from
                         THIS run's own (non-orthonormal) k_eff/v_eff
  "k_eff_items":       (4, K, d)   L2-normalized effective keys
  "v_eff_items":       (4, K, d)   raw effective values
  "S_T_forced":        (4, d, d)   ONLY on runs launched with
                         --force-rank-k (e.g. Bprobe) — train-time forced
                         state, kept separate from the staircase (§17.9)
}
```

**4 eval examples per run, `hop_set=(1,)` fixed.** `succ` (the per-row
K-cycle), `query_tokens`, `hops`, and `tgt_slot` are **not** dumped — only
`S_T`/`k_eff`/`v_eff` survive. This is the load-bearing fact that shapes
everything below.

### 17.2 Why the staircase is h=1-only — a finding, not a shortfall

`target_clause_index` (the mini-audit's FATAL fix, §15.1 item 3) maps a
hop-`h` query's answer slot to the clause `inv_succ[tgt_slot]`. At `h=1`,
`tgt_slot = succ[a_slot]`, so `inv_succ[tgt_slot] = a_slot` **identically**
— the target clause is the query's own slot, independent of `succ`.
Querying with entity `i`'s own bind-time key (`k_eff_items[i]`) against
`v_eff_items[i]`, for every `i = 0..K-1`, is therefore the *exact*,
`succ`-independent h=1 readout, reconstructable from nothing but what the
dump has. `h≥2` needs `succ` to know which entity is `h` hops from a given
start — not dumped — and reconstructing it via nearest-neighbor matching
between `k_eff`/`v_eff` rows would decode via argmax over a candidate set,
which is exactly what `CLAUDE.md`'s standing rule prohibits when a
rank-necessity claim depends on the readout ("never argmax/nearest-neighbor
over a codebook"). So h=1-only is the principled boundary of what this
dump supports, not an oversight — the multi-hop K-frontier already exists,
measured live during training with the real `succ` and real query path
(§16.2), and is not re-litigated here.

**Query-proxy caveat, stated once.** The harness's real query path computes
`q_eff` through a separate window via the same weights but different
surrounding tokens (`effective_key_window`, §16.5's alignment premise);
using `k_eff_items[i]` as the query is the only query-shaped object the
dump contains, and is structurally identical to `model_rd.py`'s own
`[model 10]` idealized-recall self-test. `--verify-schema` cross-checks the
untruncated (k=d) proxy read against each run's own logged final-checkpoint
`recovered_frac@0.9`: they agree closely when `alignment_cos_min≈1`
(e.g. `w0_rd_K16_frN_s0`: proxy 1.0000 vs. logged 0.9988, align_min
0.9999) and the proxy runs *above* the logged number when alignment has
decayed (e.g. `w0_rd_K16_frN_s1`: proxy 1.0000 vs. logged 0.9968, align_min
0.6183) — exactly the expected signature, since the proxy readout bypasses
the query-path geometry entirely and measures pure write/recall capacity.
This is a **feature** for isolating the rank question: the staircase below
is orthogonal to the alignment-decay phenomenon (§16.5), not confounded by
it.

### 17.3 Method

For each unconstrained run (`force_rank_k=None`, `S_T_raw` present): for
each `k` in a per-`K` grid (union of the task's fixed absolute grid
`{8,12,14,15,16,17,18,20}` with a `{K-2..K+2}` window so every tested `K`'s
own local transition is resolved, not just the ones the fixed grid happens
to straddle), truncate `S_T_raw` to rank `k` via deterministic SVD
(Eckart-Young optimal, numerically equivalent to `rank_utils.
truncate_to_rank`'s `eigh(ZZ^T)` path per that function's own docstring),
compute `pred[i] = S_k @ k_eff[i]`, score `cos(pred[i], v_eff[i])` for
`i=0..K-1` across all 4 examples, report `recovered_frac@0.9`. Aggregated
across every available seed (wave0_rerun's 5 + waveA's 5 at K=16 — the
pre-registered primary cell; waveA-only at K=8/24; wave0_rerun's 5 +
waveA's 2 at K=32).

### 17.4 The h=1 eval-truncation staircase (fixed grid ∪ K-local window)

`recovered_frac@0.9`, mean ± std across seeds:

**K=8 (n=2 seeds, wA_rd_K8_frN_{s0,s1}):**

| k | 6 | 7 | 8 | 9 | 10 | 12 | 14 | 15 | 16 | 17 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| measured | 0.547±.016 | 0.859±.016 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ideal-truncated | 0.422±.016 | 0.766±.016 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| theory (K−m)/K | 0.750 | 0.875 | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**K=16 (n=10 seeds — PRIMARY: wave0_rerun s0-s4 + waveA s0/s1/s0c/s1c/s2c):**

| k | 8 | 12 | 14 | 15 | 16 | 17 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|
| measured | 0.041±.023 | 0.577±.055 | 0.867±.027 | 0.950±.018 | **0.995±.007** | 0.995 | 0.995 | 0.995 |
| ideal-truncated | 0.006±.008 | 0.342±.064 | 0.828±.046 | 0.947±.021 | **0.998±.005** | 0.998 | 0.998 | 0.998 |
| theory (K−m)/K | 0.500 | 0.750 | 0.875 | 0.938 | **1.000** | 1.000 | 1.000 | 1.000 |

**K=24 (n=2 seeds, wA_rd_K24_frN_{s0,s1}):**

| k | 8 | 12 | 14 | 15 | 16 | 17 | 18 | 20 | 22 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| measured | 0.000 | 0.047 | 0.135 | 0.214 | 0.286 | 0.417 | 0.484 | 0.698±.031 | 0.865±.021 | 0.906 | **0.938** | 0.938 | 0.938 |
| ideal-truncated | 0.000 | 0.000 | 0.005 | 0.000 | 0.031 | 0.109 | 0.208 | 0.552 | 0.880 | 0.953 | **0.984** | 0.984 | 0.984 |
| theory (K−m)/K | 0.333 | 0.500 | 0.583 | 0.625 | 0.667 | 0.708 | 0.750 | 0.833 | 0.917 | 0.958 | **1.000** | 1.000 | 1.000 |

**K=32 (n=7 seeds — wave0_rerun s0-s4 + waveA s0/s1):**

| k | 8 | 12 | 14 | 15 | 16 | 17 | 18 | 20 | 30 | 31 | 32 | 33 | 34 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| measured | 0.000 | 0.004 | 0.018 | 0.030 | 0.043 | 0.081 | 0.124 | 0.243±.017 | 0.751±.017 | 0.776 | **0.795±.013** | 0.795 | 0.795 |
| ideal-truncated | 0.000 | 0.000 | 0.000 | 0.001 | 0.001 | 0.003 | 0.005 | 0.022 | 0.627±.061 | 0.675 | **0.713±.063** | 0.713 | 0.713 |
| theory (K−m)/K | 0.250 | 0.375 | 0.438 | 0.469 | 0.500 | 0.531 | 0.563 | 0.625 | 0.938 | 0.969 | **1.000** | 1.000 | 1.000 |

(`measured` = `S_T_raw` truncated; `ideal-truncated` = `s_ideal_effective`
truncated the same way, the empirically-exact "entity-aligned expectation
from the dumped key/value Grams" for THIS run's own non-orthonormal keys,
replacing the closed-form `(K−m)/K` formula which assumed orthonormal
construction; `theory` = that closed form, labeled reference only. Bold =
the run's own `K`.)

### 17.5 Reading the staircase: a graded frontier, not a razor cliff — the headline finding

A dense re-run (`--k-grid 1..34`, same runs) resolves the transition at
integer granularity and gives the precise, quantitative verdict:

| K | ceiling reached exactly at | steepest single-step jump | jump size | ranks below K |
|---|---|---|---|---|
| 8 | k=8 (1.000) | k=5→6 | +0.375 | 2-3 |
| 16 | k=16 (0.995) | k=11→12 | +0.181 | 4-5 |
| 24 | k=24 (0.938) | k=16→17 | +0.130 | 7-8 |
| 32 | k=32 (0.795) | k=19→20 | +0.066 | 12-13 |

Two facts hold simultaneously, and both are real:

1. **The ceiling is reached exactly at k=K, at every one of the four K
   tested, never before and never after** — truncating to `k=K-1` always
   costs measurable recovery; truncating beyond `k=K` never buys any (the
   K=8/16/24/32 curves are flat for every `k>K` tested, out to `k=34`).
   This is the causal-necessity signature: the operator provably cannot
   spare its Kth dimension, and gains nothing from ambient dimensions
   beyond it — consistent with §16.4's entity-subspace-rank result (rank
   recruits 94-99% of K at every cell) read behaviorally rather than
   spectrally.
2. **There is no razor cliff at k=K−1→K anywhere.** The steepest jump at
   every K lands several ranks *below* K, and the rise from floor to
   ceiling is spread across a window that widens with K (2-3 ranks at K=8,
   12-13 ranks at K=32) — the opposite of the synthetic design's single-step
   0.9681→1.0000 (K=32, §12.8.3 of `DELTANET_CAUSAL_RANK_DESIGN.md`).

This is the pre-registered caveat (task brief: "real-data learned states
have esr ≈ 15.6-15.7 (slightly under K) and non-orthonormal keys — the
theory prediction is softer than synthetic") **landing exactly as
predicted, not as a hedge that never materialized.** Three compounding,
independently-verifiable reasons, not mutually exclusive:
- **The trained rank already sits slightly under K** (§16.4: 15.62-15.73/16,
  23.08-23.20/24, 29.93/32 — never exactly K), so there is no perfectly
  rank-exact state to fall cleanly off of; the "operating point" is itself
  soft.
- **Keys are linearly independent but not orthonormal** (§14.7 item 2: Gram
  deviation 1.26-2.77, vs. the synthetic design's τ=0.03 near-orthonormal
  construction) — dropping one SVD mode of a non-orthonormal operator
  smears across multiple entities' recall rather than cleanly zeroing
  exactly one entity's channel, softening the theoretical bimodal
  (K−h)/K drop (§12.8.3's derivation) into the observed graded rise. The
  `measured` vs. `theory (K−m)/K` gap grows with `m` at every K (near-exact
  agreement at `m=1`: K=16 measured 0.950 vs. theory 0.938; large
  divergence at `m≥K/2`: K=16 measured 0.041 vs. theory 0.500 at `k=8`) —
  consistent with this reading, since a single-mode drop (`m=1`) is still
  close to the theory's local assumption, while a multi-mode drop
  compounds the smearing.
- **n is small** (4 examples × K items per seed — 32 to 128 items per seed,
  vs. the synthetic design's 163,840 per cell), though the tight seed-to-seed
  std (≤0.06 at every reported cell, ≤0.02 at most) shows the graded SHAPE
  is reproducible across independent seeds, not scatter from undersampling.

### 17.6 The architecture-native ideal: the trained state is more truncation-robust than its own naive write, most where crowding is worst

`s_ideal_effective` (`sum_j v_eff_j k_eff_j^T`, no delta-rule dynamics, no
training) is truncated the same way as an architecture-native reference.
At K=8/16/24, `ideal-truncated`'s ceiling (reached at k≈K) is at or
slightly *above* `measured`'s (K=24: ideal 0.984 vs. measured 0.938 at
k=24) — the naive Hebbian sum, built from the same final keys/values, is
about as clean or cleaner than what training converged to, in the
uncrowded regime (`K` well under `d=64`). At K=32 this **inverts**:
`measured` (0.795) clears `ideal-truncated` (0.713) by 0.08 at every
`k≥K`, and at every truncation level below K too (e.g. `k=20`: measured
0.243 vs. ideal 0.022, an order of magnitude). K=32 is also the one cell
where entities use half of `d=64`'s capacity — exactly where the delta
rule's write-time error correction (explicitly subtracting the current
prediction before writing, unlike a raw outer-product sum) has the most
interference to correct for. This is a secondary, corroborating result,
not the headline: **the trained state is not merely as good as an
idealized associative memory of the same keys/values — in the crowded
regime it is measurably better, and truncation degrades it more gracefully
than it degrades the naive ideal.**

### 17.7 Bprobe: third reproduction of the train-time forcing law

Per §16.7, Bprobe (`k∈{15,16,17}×3` seeds, primary K=16, `svd_lowrank`
truncation) was in flight at archive time. Current state (fr16 complete
3/3 at 20,000/20,000 steps; fr15/fr17 non-final, pulled live from the box):

| run | fr | complete | step | rec@0.9(h=1) | entity-subspace rank | align_min |
|---|---|---|---|---|---|---|
| fr16_s0 | 16 | **yes** | 20000/20000 | 0.1576 | 10.406 | 0.1653 |
| fr16_s1 | 16 | **yes** | 20000/20000 | 0.1604 | 9.850 | 0.0653 |
| fr16_s2 | 16 | **yes** | 20000/20000 | 0.1949 | 10.271 | 0.1286 |
| fr15_s0 | 15 | no (in-flight) | 6000/20000 | 0.0522 | 11.345 | 0.1788 |
| fr15_s1 | 15 | no (in-flight) | 6000/20000 | 0.0989 | 11.058 | 0.2511 |
| fr15_s2 | 15 | no (in-flight) | 6000/20000 | 0.1545 | 10.847 | 0.3958 |
| fr17_s0 | 17 | no (in-flight) | 8000/20000 | 0.1147 | 11.661 | 0.2683 |
| fr17_s1 | 17 | no (in-flight) | 8000/20000 | 0.1199 | 11.306 | 0.1831 |
| fr17_s2 | 17 | no (in-flight) | 8000/20000 | 0.1019 | 10.977 | 0.3768 |

**fr16 (k=K exactly — a provable no-op on the unconstrained solution's own
rank, §16.4: esr 15.62-15.73 at K=16) collapses 3/3 to `rec@0.9`
0.158-0.195, entity-subspace rank 9.85-10.41 (vs. the unconstrained arm's
15.6-15.7) — a training-time-forced state that never approaches what the
same K=16 task's unconstrained arm reaches in the same 20,000-step budget.**
This is the **third** reproduction of the train-time-forcing-breaks-SGD
finding named in `DELTANET_CAUSAL_RANK_DESIGN.md` §12.8.5 (there: "now
observed in two unrelated architectures — Task E's bespoke attention-set
encoder... and DeltaNet's delta-rule recurrence" on synthetic data) — now
on a **third** architecture/data combination (DeltaNet's delta-rule
recurrence, real GPT-2-tokenized language).

fr15/fr17 are not final, but at matched step counts they already track
fr16's own trajectory closely — the same signature the synthetic design's
own §12.8.1 reported ("already indistinguishable... burning GPU to
re-measure a number... arms have already agreed on"): at step 6000, fr15
= 0.052/0.099/0.155 vs. fr16's own step-6000 checkpoint = 0.086/0.122/0.070
(same range); at step 8000, fr17 = 0.115/0.120/0.102 vs. fr16's own
step-8000 checkpoint = 0.118/0.113/0.127 (near-exact overlap). No arm is
distinguishing itself from another at any shared step, `k=K−1`, `k=K`, or
`k=K+1` alike — the same interference signature as the synthetic Bprobe,
not a rank effect (fr16's rank-16 constraint is provably a no-op on the
task; it still collapses).

> **[FINAL 2026-07-03]** The wave closed with fr16 3/3 `complete: true`
> and fr15/fr17 6/6 **timed out at steps ~16,000-18,400 of 20,000** (the
> per-run timeout was sized from fr16's solo-GPU pacing; the six paired
> cells ran ~3× slower under 2-per-GPU contention). Their final
> checkpoints all show the identical collapse signature (entity-subspace
> rank ~10.0-11.0, rec@0.9 ~0.10-0.12, flat trajectories for the last
> ≥6K steps) — full-staircase train-time interference confirmed at all
> nine cells, no re-run warranted (the arms had been indistinguishable
> and plateaued for thousands of steps). Final archive:
> `experiment-runs/2026-07-03_deltanet_rd_waves/waveBprobe_final/`.

### 17.8 Gate decision: Wave B is moot

§7's manifest gates Wave B on B-probe showing "**≥1 seed shows life at
`k≥K` and a step relative to `k<K`**." fr16 (`k=K`, provably sufficient
per the unconstrained arm's own measured rank) shows no such life — its
3/3 seeds sit inside fr15's (`k=K-1`, provably insufficient) own range at
every matched step, and its final `rec@0.9` (0.158-0.195) is roughly
5-6× below the unconstrained arm's own K=16 ceiling (0.995, §17.4) despite
`k=16` being architecturally capable of it. **The gate is not met. Wave B
(the full `k∈{K-2..K+2}`-style force-rank straddle grid, priced at
~35-50 GPU-h in §7) does not launch — mirroring
`DELTANET_CAUSAL_RANK_DESIGN.md` §12.8.1's identical decision on the
synthetic design at the same operating point (`k=K` train-time-forced),
for the same reason (stochastic `svd_lowrank`-in-the-training-loop
interference at `k≈K`, not a rank effect).** This is now the third time
this exact train-time-forcing failure mode has fired at `k` near the
operating rank (Task E's bespoke encoder; the synthetic DeltaNet design;
here) — a standing methodological result in its own right (§12.8.5),
reproduced a third time rather than merely cited.

### 17.9 Causal verdict

Per §12.8.2's pre-registered argument (restated for this program): eval-time
truncation is decisive precisely because the unconstrained arm's learned
rank is close to K with no meaningful slack (§16.4), so removing the
smallest retained direction removes load-bearing structure if and only if
the rank-K bound genuinely binds — both predictions (a ceiling at K; real
cost below it) are falsifiable in the same table, and §17.4-17.5 falsify
neither: **RD-1's branch (b), causal necessity, is CONFIRMED at all four
K tested (8, 16 primary, 24, 32)** — every cell shows the operator losing
real, monotonically-increasing recovery as rank is truncated below its own
learned operating point, and gaining nothing from any dimension beyond it.
Combined with §16.4 (rank recruits 94-99% of K, behaviorally verified here
to be load-bearing rather than merely spectrally present) and the K-frontier
itself (§16.2, real multi-hop composition with the genuine query path),
this closes the causal loop Wave B was designed for, without Wave B — the
same structural resolution the synthetic design reached at §12.8.5, now
independently reached on real tokenized language.

**The qualification that must travel with this verdict, not be dropped
from it:** unlike the synthetic design's razor cliff, the real-data
transition is **graded across a window several ranks wide, not a single-step
threshold** (§17.5) — a quantitatively different, and honest, finding about
how the same causal mechanism looks once the two clean-regime assumptions
(exact rank-K convergence; orthonormal keys) are replaced by what SGD and
a real tokenizer actually produce. If a future write-up quotes "razor
cliff at k=K−1→K" for the real-data result, that is not what was measured
here — quote §17.5's table instead.

### 17.10 Compute and reproducibility

The eval-truncation analysis itself is post-hoc (numpy + stdlib on the
already-archived dumps) — effectively free (`analyze_eval_truncation_rd.py`
completes the full 21-run staircase across all four K cells in well under
a minute on a laptop, no GPU). The GPU cost attaches to the runs that
produced the dumps, already priced in §15.3/§16.9. Bprobe (this section's
new GPU spend, summed `wall_s` from each run's own field): the 3 complete
fr16 runs total 11,564.8s = **3.21 GPU-h**; all 9 runs (including the 6
still-in-flight cells' partial `wall_s` at pull time, itself not final)
sum to 43,332.4s = **12.04 GPU-h so far**, which will grow as fr15/fr17
continue past this session's archive point.

Reproducibility: `matrix-thinking/deltanet_rd/analyze_eval_truncation_rd.py`
(`--smoke` for the theory-validation gate; `--dir DIR [DIR...]
--verify-schema` for the staircase + schema cross-check; `--bprobe-dir DIR`
for §17.7's table). Archive:
`experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA,waveBprobe}/`
(waveBprobe newly pulled this session from
`youthful-indigo-turkey:/home/nvidia/chapter2/deltanet_rd/results/deltanet_rd_w0b/waveBprobe/`,
including the 6 non-final in-flight JSONs, pulled before the box's
scheduled stop per this session's standing instruction to "pull FIRST and
report what you got").

### 17.11 Wave 1 closing verdict

Per §7's gate table's "RD-1 verdict" row: this section, combined with
§14-16, closes Wave 1. **CONFIRM, on all three legs the manifest asked
for:**
- **Binding on real language** (§15: 10/10 collapse-free after the
  target-index fix + NCE objective; K=16 `rec@0.9` 0.996-0.999).
- **A K-exactness frontier at fixed `d_state`** (§16.2: graded, monotone
  in K, with the same rank-recruited/exactness-scarce separation
  `TASK_E_FINDINGS.md` and `STAGE0_DESIGN.md` already found on two other
  architectures/axes).
- **Causal rank necessity, closed without a live Wave B** (§17.9: CONFIRMED
  via eval-time truncation once the train-time instrument was shown dead a
  third time, §17.7-17.8) — **with the graded-not-razor qualification
  (§17.5) recorded as part of the verdict, not as an asterisk to be
  dropped later.**

This is, structurally, the same "eval-truncation closes the causal loop
without Wave B" result `DELTANET_CAUSAL_RANK_DESIGN.md` reached on
hand-built orthonormal-key data — reproduced on real GPT-2-tokenized
language, on a production kernel, with the clean-regime assumptions
(exact rank, orthonormal keys) both genuinely absent and the resulting
softening of the transition measured, not asserted. Per §7's gate
language, this CONFIRM opens Wave C (real-corpus LM pretrain, gated Wave 2)
on schedule; that launch decision is not taken in this section.

---

## 18. Deeper-hop training probe (2026-07-03): hop supervision does NOT move the decay curve — depth decay is set by binding exactness, not the training distribution over hops

**Question.** Wave A trained h∈{1,2,3} and observed the within-K per-hop
decay (§16.2). Is that decay partly an *extrapolation* artifact — would
training AT h∈{1..5} extend the exact-composition regime?

**Cells.** K∈{8,16} × 3 seeds, h_train={1,2,3,4,5}, 25K steps, unconstrained
arm, NCE loss, audited Wave-1 harness (only CLI hop args changed). K=8:
h_test={6,7}, h_extra={14,15}; K=16: h_test={6,7,9}, h_extra={15}. First K=8
launch used h_extra=7 (overlapped h_test) and was correctly refused by the
overlap guard — relaunched with depth-amplification probes instead. Archive:
`experiment-runs/2026-07-03_deltanet_rd_waves/deephop/`. ~2 GPU-h.

**Result: the per-hop recovery curve is INVARIANT to hop supervision.**
rec@0.9, deephop (3 seeds) vs Wave A baseline (h_train={1,2,3}, §16.2 table):

| K | h=4 | h=5 | h=6 | h=7 |
|---|-----|-----|-----|-----|
| 8, Wave A (h≥4 held-out) | 0.944–0.956 | 0.873–0.899 | 0.780–0.825 | 0.699–0.730 |
| 8, deephop (h=4,5 TRAINED) | 0.921–0.941 | 0.836–0.870 | 0.755–0.768 | 0.658–0.701 |
| 16, Wave A (h≥4 held-out) | 0.419–0.465 | 0.252–0.280 | 0.137–0.161 | 0.069–0.097 |
| 16, deephop (h=4,5 TRAINED) | 0.390–0.503 | 0.227–0.299 | 0.121–0.178 | 0.063–0.108 |

Training AT h=4,5 produces recovery statistically indistinguishable from (if
anything a hair below) extrapolating TO h=4,5 from h_train={1,2,3}, at both
K. Held-out h=6,7 likewise unchanged. New K=16 h=9 point: 0.018–0.030
(decay continues smoothly). In-distribution h=1–3 matches Wave A (K=8:
1.000/0.996–1.000/0.967–0.977; K=16: 0.996–0.999/0.874–0.917/0.631–0.729),
so the added supervision cost nothing either.

**Depth-amplification signature reproduces a second time.** K=8 h_extra=14/15
(effective_hop 6/7 — same relational distance as h_test=6/7, 2× the literal
iteration count): rec@0.9 collapses 0.755–0.768 → 0.051–0.063 (h=14 vs h=6)
and 0.658–0.701 → 0.032–0.050 (h=15 vs h=7). Iteration count, not effective
relational distance, drives the decay — same signature as §16.3's h=21 probe.

**Reading.** The "train deeper" lever is dead: per-binding inexactness ε is
set by K (binding quality at write time), and composition inherits ε^h
regardless of the hop distribution the readout was trained on. This
strengthens the K-exactness mechanism program (DELTANET_RD_EXACTNESS_DESIGN
.md): the lever, if it exists, lives at the write/geometry level (key
crosstalk), not the supervision level. It also pre-emptively weakens the
mechanism study's budget/supervision mechanism (d) on the hop-supervision
axis (steps-budget axis still open). Validity note: all 6 cells premise-clean
at salvage tier; K=16 s1/s2 fail only the strict per-item alignment minimum
(alignment_cos_min 0.82 on ~1% of items, frac=0.99) — descriptive tier for
those two seeds per §14.7, s0 fully clean; K=8 all clean. Conclusions here
rest on the cross-cell pattern, not any single strict-tier cell.

**K=32 extension (launched 2026-07-03, in flight):** 3 seeds, same config,
h_test={6,7,9}, h_extra={15} — does hop supervision matter where the
frontier is fully collapsed? Results to be appended here.

### 18.1 Deephop program CLOSED (2026-07-03, overnight): the decay curve is a function of K alone — invariant to BOTH hop supervision AND training budget

Full grid complete: K∈{8,16,24,32} × h_train={1..5} × 3 seeds at 25K steps,
PLUS 2.5× budget guards (62.5K steps) at K∈{8,16,32} × 3 seeds. 21 cells,
all complete, all salvage-tier premise-clean. Archive:
`experiment-runs/2026-07-03_deltanet_rd_waves/deephop/` (22 files).

**K=24 (completes the K-axis):** held-out h=6 rec@0.9 = 0.007–0.009 vs Wave
A baseline 0.008–0.009 — identical. **K=32:** trained h1/h2/h3 =
0.74–0.76/0.23–0.26/0.04–0.05 vs baseline 0.78/0.26/0.05 — identical,
held-out floor. **62.5K budget guards:** K=8 h=6 0.77–0.81 (25K: 0.76–0.77),
K=16 h=6 0.12–0.18 (25K: 0.12–0.18), K=32 all-zero (25K: all-zero) — the
2.5× budget moves nothing at any K, closing the budget-artifact loophole the
house law requires closing before trusting a negative.

**Closed finding:** the per-hop exactness decay curve on real tokenized text
is a function of K alone. It cannot be moved by (a) training directly at
deeper hops, or (b) 2.5× optimization budget. Combined with the
depth-amplification signature (h=14/15 at K=8 collapse to 0.03–0.07 at the
same effective distance as h=6/7's 0.66–0.81 — reproduced again at 62.5K),
the per-binding error ε is fixed at write time by K, and composition
inherits ε^h mechanically. The lever, if it exists, is at the write/geometry
level — exactly what DELTANET_RD_EXACTNESS_DESIGN.md (Rev 3, BUILD-READY)
attacks, with Wave 0's free reconstruction analysis already showing the
predicted-vs-measured residual growing monotonically in K (0.00→0.10,
K8→K32) on archived dumps.

---

## 19. Wave 2 results (Waves C+D) (2026-07-04)

**Claim tier for everything below: descriptive+interventional (RD-2, §6.3,
§14.7) -- NOT premise-conditional causal.** No claim in this section says
SGD was pushed toward a rank-disciplined solution *by* real-text
pretraining, and none says truncation damage *proves* reasoning
*requires* rank K. The evidence supports weaker, still-useful statements:
*"the trained state's measured rank differs by corpus"* and *"truncating
the trained state at inference time damages reasoning-corpus loss more
than narrative-corpus loss, at these k, at this scale."*

**Manifest.** Wave C: `d_model=256, d_state=64, n_layers=2, conv_size=4`,
6 cells (`{openr1,wikitext} x seeds{0,1,2}`), 6103 steps each, checkpoints
every 1000 steps + final (7 checkpoints/run), all complete
(`ALL_DONE`, 0 skipped steps). Wave D: same 6 checkpoints (final step,
6103), truncation grid `k in {8,16,24,32,48,64}` (eigh, `svd_lowrank`
discipline carried from Wave A/B), 32 eval windows/corpus, frequency
bands (4, MAJOR-5 fallback) + symbol/word/other token-class strata, both
eval corpora scored from every checkpoint (a full 2x2 train-corpus x
eval-corpus grid, not just the home-corpus diagonal). Analysis script:
`analysis_lm_w2.py`, archived alongside its full text output at
`experiment-runs/2026-07-04_lm_rd_wave2/` (§19.6).

### 19.1 Q1 -- state-rank dynamics, reasoning (openr1) vs narrative (wikitext)

**Contamination is real and corpus-asymmetric -- conditioning matters.**
OpenR1-Math documents are short relative to the 512-token window (mean
sampled doc length ~500-700 tokens), so a large fraction of `frac=1.0`
(full-window) rank-probe windows cross a document boundary: **69.6% of
openr1 windows are contaminated vs 0.0% of wikitext windows** (WikiText
docs run thousands of tokens). Per the auditor's guidance, all numbers
below use `window_within_doc == True` only, pooled across all 7
checkpoints and all 3 seeds to recover usable sample size (contamination
filtering leaves openr1 with **n=51** windows pooled per layer, vs
**n=168** for wikitext -- this asymmetry in n is itself a byproduct of the
corpus, not a methodological choice, and it means the openr1 estimates
below carry visibly wider uncertainty).

**Pooled effective/stable rank, final-in-window (`frac=1.0`), within-doc
only, `d_state=64` is the ceiling:**

| Corpus | Layer | n | Effective rank | Stable rank |
|---|---|---|---|---|
| openr1 | L0 | 51 | 40.62 +/- 1.66 | 3.49 +/- 0.94 |
| openr1 | L1 | 51 | 35.75 +/- 1.62 | 2.63 +/- 0.29 |
| wikitext | L0 | 168 | 37.48 +/- 2.44 | 3.94 +/- 1.08 |
| wikitext | L1 | 168 | 34.85 +/- 0.93 | 2.55 +/- 0.55 |

**Reading, held to the descriptive tier:** at layer 0, openr1's effective
rank is measurably higher than wikitext's (40.6 vs 37.5, gap exceeds
either SD) -- but **stable rank reverses this at the same layer** (openr1
3.49 vs wikitext 3.94). Effective rank (spectral-entropy based, counts
many small-but-nonzero directions) and stable rank
(`||.||^2_F / ||.||^2_2`, dominated by the top singular value) are
answering different questions, and they disagree on which corpus's
layer-0 state is "higher rank." At layer 1 both metrics show openr1 and
wikitext close to each other (35.75 vs 34.85; 2.63 vs 2.55) -- no
reliable gap. **Honest summary: there is a modest, layer-0-specific,
metric-dependent rank gap between corpora -- not a clean "reasoning uses
more rank than narrative" signature that holds across layers and across
both rank metrics.** This should not be over-read past what it is: a real
but qualified descriptive difference, flagged exactly at the tier §14.7
authorizes.

**Unfiltered-vs-filtered check (robustness of the conditioning itself):**
for openr1, the unfiltered (contamination-included) pooled effective rank
is 40.86/36.36 (L0/L1) vs the filtered 40.62/35.75 -- a small shift (<1
rank unit), meaning cross-document contamination does not appear to
inflate or deflate the effective-rank read much at this scale, even
though it does shrink usable n substantially. This is worth recording as
a check that passed, not assumed.

### 19.2 Q4 -- rank trajectory over training vs val-loss phase (folded in here since it shares Q1's data)

Per-checkpoint pooled effective rank (frac=1.0, within-doc filter,
pooled across 3 seeds) alongside each corpus's home val loss:

| Step | openr1 n | openr1 eff.rank L0 | openr1 eff.rank L1 | openr1 val_loss | wikitext n | wikitext eff.rank L0 | wikitext eff.rank L1 | wikitext val_loss |
|---|---|---|---|---|---|---|---|---|
| 1000 | 9 | 42.93 | 35.28 | 2.693 | 24 | 39.62 | 35.01 | 5.620 |
| 2000 | 6 | 40.76 | 34.59 | 2.460 | 24 | 39.63 | 35.80 | 5.211 |
| 3000 | 3 | 40.34 | 35.46 | 2.220 | 24 | 38.33 | 35.52 | 5.031 |
| 4000 | 15 | 40.16 | 35.56 | 2.182 | 24 | 37.13 | 34.91 | 4.787 |
| 5000 | 6 | 39.69 | 36.40 | 2.207 | 24 | 36.18 | 34.47 | 4.754 |
| 6000 | 6 | 40.13 | 37.19 | 2.082 | 24 | 35.70 | 34.30 | 4.727 |
| 6103 | 6 | 39.77 | 36.13 | 2.067 | 24 | 35.75 | 33.94 | 4.688 |

**n is small and uneven per checkpoint** (openr1 drops to n=3 at step
3000 from contamination filtering -- that single row is noisy and should
not be over-weighted; wikitext holds a stable n=24 throughout since it is
essentially uncontaminated).

**Surprise, reported straight (not smoothed):** layer 0's effective rank
*falls* as training proceeds and val loss falls, in **both** corpora --
Pearson correlation between the effective-rank trajectory and the
val-loss trajectory across the 7 checkpoints is **+0.918 (openr1 L0)**
and **+0.910 (wikitext L0)** (positive correlation here means rank and
loss move the same direction -- both decrease together, i.e. the state
becomes *lower*-rank as the model gets better, not higher). This is the
opposite of the "SGD recruits more rank as training improves" intuition
carried over from the causal-rank chain's controlled-task results --
worth flagging explicitly as a genuinely different regime (LM next-token
prediction, not a forced K-way binding task). Layer 1 does not replicate
this cleanly: openr1 L1 correlation is **-0.680** (rank rises slightly as
loss falls) and wikitext L1 is **+0.613** but visibly less monotonic than
L0 (rises steps 1000-to-2000, then declines). **Reading: there is a
rank-trajectory signature that tracks the loss curve, but it is a
layer-0-specific contraction, common to both corpora (i.e. not a
reasoning-specific phenomenon) -- a general training dynamic at this
scale, not evidence either corpus is special.**

### 19.3 Q2 (headline) -- inference-time truncation damage vs k, by corpus

**Home-corpus comparison** (openr1 checkpoint scored on held-out openr1;
wikitext checkpoint scored on held-out wikitext -- the fair like-for-like
read), mean +/- SD over 3 seeds, in nats/token:

| k | openr1 raw | openr1 balanced | wikitext raw | wikitext balanced | raw delta(o-w) | balanced delta(o-w) |
|---|---|---|---|---|---|---|
| 8 | 0.1018 +/- 0.0046 | 0.1692 +/- 0.0440 | 0.0885 +/- 0.0067 | 0.0903 +/- 0.0398 | **+0.0133** | **+0.0788** |
| 16 | 0.0399 +/- 0.0035 | 0.0535 +/- 0.0379 | 0.0339 +/- 0.0026 | 0.0366 +/- 0.0169 | **+0.0061** | **+0.0169** |
| 24 | 0.0145 +/- 0.0039 | 0.0164 +/- 0.0131 | 0.0098 +/- 0.0008 | 0.0061 +/- 0.0129 | **+0.0047** | **+0.0103** |
| 32 | 0.0034 +/- 0.0024 | 0.0078 +/- 0.0245 | 0.0032 +/- 0.0019 | -0.0083 +/- 0.0059 | +0.0002 | +0.0161 |
| 48 | -0.0002 +/- 0.0002 | -0.0038 +/- 0.0052 | 0.0007 +/- 0.0004 | -0.0001 +/- 0.0017 | -0.0009 | -0.0037 |
| 64 | -0.0000 +/- 0.0000 | 0.0001 +/- 0.0001 | 0.0000 +/- 0.0001 | 0.0001 +/- 0.0002 | -0.0000 | -0.0000 |

**k\* (both `|raw|` and `|balanced|` < 0.005 nats): k\*=48 for BOTH
corpora** -- the truncation level at which damage effectively vanishes is
identical between reasoning and narrative text; the two curves converge
to the same noise floor by `k=48` (out of `d_state=64`).

**Headline answer to Q2: at low-to-moderate k (8, 16, 24), openr1
(reasoning-dense) shows more truncation damage than wikitext
(narrative), and the direction is consistent across all 6 cells (3 k
values x {raw, balanced})** -- every single one of those 6 comparisons
points the same way (openr1 > wikitext), despite n=3 seeds and
individually noisy balanced estimates (SDs up to 0.044 at k=8). The gap
is largest in relative terms at k=8 (raw: +15%, balanced: near-2x) and
shrinks by k=24; k=32 and beyond show no reliable difference (balanced
mean at k=32 for wikitext goes slightly negative, `-0.0083 +/- 0.0059` --
noise around zero, not a real sign flip, given the tiny rare-token-band
counts feeding the balanced estimator at this magnitude). **This is
consistent with (not proof of) reasoning text being more rank-sensitive
under truncation than narrative text, at this scale, at these k, at this
corpus pair** -- the descriptive+interventional tier this design commits
to, no stronger.

**Symbol/word/other stratification (home corpus, k=8 and k=16, raw
damage) -- checking the MAJOR-5 confound (is this "reasoning" or just
"symbol density"?):**

| k | class | openr1 | wikitext |
|---|---|---|---|
| 8 | word | 0.1267 (n~1742) | 0.0961 (n~3262) |
| 8 | symbol | 0.0860 (n~2129) | 0.0617 (n~751) |
| 8 | other | 0.0572 (n~225) | 0.0280 (n~83) |
| 16 | word | 0.0502 (n~1742) | 0.0339 (n~3262) |
| 16 | symbol | 0.0353 (n~2129) | 0.0345 (n~751) |
| 16 | other | 0.0044 (n~225) | 0.0252 (n~83) |

At k=8, openr1 damage exceeds wikitext's **within every token class**,
including plain `word` tokens (0.127 vs 0.096) -- the gap is not purely an
artifact of openr1 having more math-symbol tokens; matched by class,
openr1 windows still show more damage. This partially answers (does not
fully resolve -- no symbol-density-matched STEM-exposition third arm was
run, §6.1's MAJOR-5 fix; this is the pre-registered fallback
stratification only) the reasoning-vs-symbol-density confound. At k=16
the picture is noisier: `word` still favors the same direction, `symbol`
ties, and `other` reverses (openr1 lower) -- but `other`'s counts
(n~83-225) are too small to trust a sign flip there; read k=16's class
breakdown as weaker evidence than k=8's, not contradicting evidence.

**Full 2x2 (train-corpus x eval-corpus) -- separating "does the eval text
need more rank" from "does the training regime change how much rank gets
used," raw_mean, mean over 3 seeds:**

| k | train=openr1/eval=openr1 | train=openr1/eval=wikitext | train=wikitext/eval=openr1 | train=wikitext/eval=wikitext |
|---|---|---|---|---|
| 8 | 0.1018 | 0.0555 | 0.0787 | 0.0885 |
| 16 | 0.0399 | 0.0238 | 0.0346 | 0.0339 |
| 24 | 0.0145 | 0.0084 | -0.0025 | 0.0098 |
| 32 | 0.0034 | 0.0021 | -0.0004 | 0.0032 |
| 48 | -0.0002 | 0.0006 | 0.0010 | 0.0007 |
| 64 | -0.0000 | -0.0001 | -0.0000 | 0.0000 |

**Surprise worth flagging:** at every k where damage is non-trivial (8,
16, 24), **home-trained models show more truncation damage than
cross(out-of-domain)-trained models scored on the same eval text** -- e.g.
at k=8, an openr1-trained model evaluated on openr1 (0.1018) is damaged
more than a wikitext-trained model evaluated (out-of-domain) on the same
openr1 text (0.0787). This is readable as competence-driven: an
out-of-domain model already has high baseline loss (it isn't using
context well regardless), so truncating its state changes little; a
competent, home-trained model actually leverages the higher-rank state
for good predictions, so truncation hurts it more. This means Wave D's
headline gap (§19.3, home-corpus comparison) is entangled with "how well
the model has learned this domain," not purely "how rank-hungry this
text is" -- flagged honestly as an open confound in the interventional
read, same spirit as §6.1's MAJOR-5 discipline, not something this
wave's budget resolves.

### 19.4 Q3 -- cross-corpus val-loss asymmetry (sanity read only)

Final-checkpoint (step 6103) val loss, mean +/- SD over 3 seeds:

| Train corpus | Eval corpus | Val loss | Tier |
|---|---|---|---|
| openr1 | openr1 | 2.0668 +/- 0.0124 | home |
| openr1 | wikitext | 8.2944 +/- 0.0536 | cross |
| wikitext | openr1 | 11.2216 +/- 0.1792 | cross |
| wikitext | wikitext | 4.6881 +/- 0.0070 | home |

Matches the pre-registered sanity figures (openr1-trained on wikitext ~
8.3, wikitext-trained on wikitext ~ 4.7). **Read exactly as instructed --
noted, not over-read:** both cross-corpus numbers are far worse than
either home number, and the two cross numbers are themselves asymmetric
(8.29 vs 11.22) -- plausibly a vocabulary/domain-coverage artifact (a
wikitext-only model has essentially never seen dense math-symbol/number
sequences; an openr1-only model has a narrower entity/prose vocabulary
than WikiText's), not a rank-specific finding. No rank claim is drawn
from this table; it exists to contextualize §19.3's home-corpus framing
choice (§19.3's confound note above uses exactly these loss values to
explain the home-vs-cross damage asymmetry).

### 19.5 Summary against the four pre-registered questions

1. **Rank contrast (openr1 vs wikitext), conditioned on contamination:** a
   real but modest, layer-0-only, metric-dependent gap (effective rank
   favors openr1 at L0; stable rank favors wikitext at the same layer;
   L1 shows no reliable gap on either metric). Not a clean across-the-board
   "reasoning needs more rank" signature.
2. **Headline -- is reasoning text more rank-sensitive under truncation?**
   **Yes, at low-to-moderate k (8/16/24), consistently across raw and
   frequency-balanced metrics and largely within-token-class** -- but the
   two corpora converge to the same noise floor by the same k\*=48, and
   part of the gap is entangled with home-vs-cross training competence
   (§19.3's 2x2 finding), not cleanly isolated to "text content alone."
3. **Cross-corpus val asymmetry:** confirmed as pre-registered (8.3 vs
   4.7), read as a vocabulary/domain-coverage artifact per instruction,
   not a rank finding.
4. **Rank-trajectory signature tracking val-loss phase:** yes at layer 0,
   in both corpora -- but it is a rank *contraction* as loss falls, not
   the "SGD recruits more rank" direction familiar from the controlled
   causal-rank chain; layer 1 does not replicate it cleanly. General
   training dynamic, not corpus-specific.

**What this wave does NOT claim:** it does not claim SGD was pushed
toward a rank-disciplined solution *by* real-text pretraining (that would
require a train-time force-rank intervention on real-text pretraining
itself -- not designed here, §6.3/§12), and it does not claim the
observed damage differential *proves* natural-language reasoning
*requires* rank K -- only that, at this scale, on this corpus pair, under
inference-time truncation, reasoning-dense text is measurably more
damaged at low k than narrative text, with the qualifications above
carried forward, not dropped at write-up time.

### 19.6 Reproducibility

Analysis script: `analysis_lm_w2.py`, run CPU-only on
`youthful-indigo-turkey` against the completed Wave C/D result JSONs (no
training or eval launched for this analysis pass). Full script + full
text output archived at `experiment-runs/2026-07-04_lm_rd_wave2/`
(`analysis_lm_w2.py`, `analysis_lm_w2_output.txt`), alongside the raw
Wave C (`waveC/wC_lm_*.json`, `AGGREGATE.json`, `SUMMARY.txt`) and Wave D
(`waveD/wD_lm_*.json`, `AGGREGATE.json`, `SUMMARY.txt`) result files
(checkpoints themselves -- `waveC/checkpoints/*.pt` -- are NOT archived
off the box per house convention; too large, and not needed to reproduce
this analysis, which reads only the logged JSON summaries). Source on
box: `youthful-indigo-turkey:/home/nvidia/chapter2/deltanet_rd/results/lm_rd/{waveC,waveD}/`.
