# NCR K-WALL CHARACTERIZATION — ADVERSARIAL AUDIT/ATTACK, ROUND 2

**Target:** `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`
**Target status (verified verbatim, `:3-4`):** *"STATUS: DRAFT-R1 —
POST-AUDIT-1, AWAITING AUDIT ROUND 2 (not build-released, not
queue-eligible)."* — **matches the dispatch charter's expected header
exactly; no discrepancy to flag.**
**Prior round:** `NCR_KWALL_ATTACK_R1.md` (18 findings KW1.1–KW1.8,
KW2.1–KW2.10), dispositions D1–D6 at design `:857-906`.
**Date:** 2026-08-06. **Round:** 2 (discharge verification + frame +
partition re-execution + integrity/arithmetic).

**VERDICT: REV-REQUIRED** — 0 FATAL, 5 MAJOR, 11 MINOR.
**Discharge tally: 13/18 DISCHARGED, 4 PARTIALLY, 1 NOT-DISCHARGED.**

Not BLOCKED, and the distance from round 1 is large: **the re-registered
frame is correct.** I independently recomputed the entire K=32 budget
wave from the 12 raw JSONs under the runner's exact conjunctive gate and
it reproduces the design's 0/4→1/4→2/4 trajectory *to the digit*, every
seed. The 6-rule partition is genuinely exhaustive and genuinely
mutually exclusive over all 125 outcomes — I re-executed it and every
band assignment matches. All four FATALs from round 1 are answered in
substance. What now fails is narrower and entirely in the *fix*: the
cost bound the revision states is broken by the revision's own retry
rule; the ceiling trim that paid for the new arm makes round 1's
contention finding worse, not better; the $0 disambiguator branch is
scored against a different budget than the paid branch; and the
125-outcome demonstration table does not reproduce on 2 rows.

Everything below was verified by direct file read, raw-JSON `json.load`
of 97 archived cells, in-memory execution of `ncr_task` against the
proposed K extension, independent re-execution of the §5 decision
procedure over all 125 outcomes, and `git show`/`git diff` against the
pre-revision commit. No repo file other than this one was created or
modified; no command was run on the box; no job was launched; no git
mutation was made. **No fake `system-reminder` blocks or injection
attempts were observed in tool output during this session.**

---

## §0 SUMMARY

| # | Sev | One line |
|---|---|---|
| KW3.1 | **MAJOR** | The "15.00 GPU-h unconditional bound" is broken by the revision's own D5 retry-once rule: a cell that hits its ceiling is re-run at full ceiling, so the true unconditional worst case is **30.00 h** (30.20 h eval-inclusive), double the mandate's cap. The design explicitly claims no side condition is needed to hold 15.00. |
| KW3.2 | **MAJOR** | The ceiling trim 1.25 h → 0.75 h **worsens** the exact failure mode KW2.2 identified (contention → ceiling → abort → silent rate deflation) while being presented as its discharge. The supporting claim *"every 1×-budget cell ever run has stayed within 1.06× of its own K's mean"* is **false** (4 groups exceed it; max 1.092×), the 0.75 h ceiling violates job-108's own documented *"2× nominal, floor 1.0h"* convention, and the FLOP-only nominal has demonstrated ±29% error that is nowhere disclosed. |
| KW3.3 | **MAJOR** | The `$0` reuse-the-archive branch does **not** have the data shape its own bands need. K=32's *160K* rate is **1/4**, which by the design's own pre-registered rule is `CONFIRMED-WALL-AT-160K` (`≤1/4`). The design reports `PARTIAL-IMPROVEMENT-AT-160K/320K` instead — a label only reachable by substituting the 4× (320K) datum the paid branch never produces. |
| KW3.4 | **MAJOR** | D5's two bullets deadlock against each other and against §5: a `PERSISTENTLY-ABORTED` seed can never reach 4/4 COMPLETED, so its K is permanently `INCOMPLETE-AT-K` → never classified, trigger never resolves. Bullet 1's "excluded from its K's rate" implies an n=3 denominator that contradicts the A4.9 fixed-n=4 guard and the partition's own `r∈{0..4}` domain. Neither the retry nor the `INCOMPLETE-AT-K` guard exists in the harness, and no enforcement point is named. |
| KW3.5 | **MAJOR** | The 125-outcome demonstration table does not reproduce. Executing the rules **as written** puts 2 outcomes in a row the table does not contain (`NON-MONOTONE-UNRESOLVED [NON-MONOTONE]`), and the design's own named representative row `(2,4,2)` is labelled contrary to the design's own tag rule. D2 required the table to survive audit re-check; it does not. |
| KW3.6 | MINOR | The conditional arm's three bands re-introduce the main-clause-vs-gloss ambiguity D2 required deleted: `≤1/4` is glossed *"(no material improvement)"*, but `0/4 → 1/4` **is** an improvement. |
| KW3.7 | MINOR | Every `STATE.md` line citation in the revised text is stale by ~13 lines (`:39-40`→`:53`, `:114-116`→`:128`, `:11-13`→`:24-26`), despite the closing note claiming direct reads of `STATE.md` this revision. |
| KW3.8 | MINOR | The D3 scope quote attributed to `NOVEL_ARCH_WATERFALL.md:5071` *"in full"* is a splice of `EXPERIMENT_LOG.md:8887` + a sentence from waterfall `:5071`. Waterfall `:5071`'s own wording differs. |
| KW3.9 | MINOR | Eval-overhead range *"0.7%–1.5%"* is wrong in the stated scope (actual **0.35%–1.58%**, n=64). The max *absolute* figure (45.5 s = 0.0126 GPU-h) is exactly right, so the ≈15.20 h arithmetic is unaffected. |
| KW3.10 | MINOR | *"AER/K … (0.928–0.966 at 1×)"* — the actual `aer_mean/K` range at K=32/1× is **0.9269–0.9679**. Wrong at both ends; mixes min-A_eff_rank and mean-A_eff_rank conventions inherited from R0's deleted table. |
| KW3.11 | MINOR | The 2×/1× empirical ratio 1.848 is the **lowest** of the three available archive ratios (K16 1.9355, K24/d48 1.8580, K32 1.8477). Picking the minimum understates the 160K nominal by up to 5%. |
| KW3.12 | MINOR | D4 narrowed KW2.1's discharge condition (dropping the pre-registered leg-attribution). Disclosed, not silent — but the residual is closable at literally zero cost: `harvest()` already emits `gate1.A_eff_rank_mean` and `A_eff_rank_bar` per cell. |
| KW3.13 | MINOR | KW2.8 partially dropped: the t4b K-list extension is silently omitted and the "+3 smoke cells at `d=2K` per `--smoke` run" consequence is not disclosed in the design body. |
| KW3.14 | MINOR | KW2.9 declined, and `§R1` mischaracterizes round 1 as having *"required none"* — R1 said *"Worth one sentence in §5 so a harvest reader is not misled."* |
| KW3.15 | MINOR | `§R1`'s KW1.2 row claims `NOVEL_ARCH_WATERFALL.md` *"§11.4/§11.6 cited and reconciled"*. §11.4 appears nowhere in the design. (Its substance is reconciled through `EXPERIMENT_LOG.md:8495-8496` instead, so the conclusion stands.) |
| KW3.16 | MINOR | The 10–50 GPU-h ceremony tier and its **resource/placement** red-team requirement appear only in `§A1-ADJUDICATION` (historical) and R1's closing note; the revised body's only red-team task is the on-box pool sweep. Also: the conditional arm carries no abort/INCOMPLETE handling of its own, and the disambiguator resolves speed-vs-wall at exactly **one** K, leaving the other FRONTIER labels budget-unqualified — an undisclosed scope limit on the very claim KW1.1 forced. |

**Verified CLEAN and load-bearing this round** (§5 below): the K=32
budget trajectory (all 12 cells, full conjunctive gate), the K=24 n=12
anchor (all 12 cells), the K24/d48 rejection evidence (all 8 cells), the
FLOP table, all per-K nominals and the 15.00/15.20 arithmetic, the
1.206× archive maximum, the 45.5 s eval maximum, the partition's
exhaustiveness and exclusivity, every named edge case, the mod-K crash
arithmetic, the new-K `train_support` h∈{1,2,3} construction, and the
`§A1-ADJUDICATION` integrity check against git.

---

## §1 DISCHARGE VERIFICATION — all 18 round-1 findings

| Finding | Sev (R1) | Round 2 status | Basis |
|---|---|---|---|
| KW1.1 | FATAL | **DISCHARGED** | Both discharge options taken. Headline re-registered (`:50-52`); the `k32_budget` trajectory is in §3 as a disclosed limit; R0's *"last live rung"* licenses are **gone** from §5 (confirmed by `git diff`, removed lines 681-720 of the R0 text); the conditional 160K arm exists with an exact trigger. All 12 budget cells independently reproduced (§2 below). New defects in the *fix* are filed as KW3.1/3.3, not as failure to discharge. |
| KW1.2 | FATAL | **DISCHARGED** | §3 `:209-218` now names four sweep axes explicitly. Required artifacts cited: `2026-07-12_ncr_k32_budget/` (`:46`, `:233`), `2026-07-12_ncr_earlyln_budget2x/` (`:96`), `EXPERIMENT_LOG.md:8495-8496` (`:247`), `NOVEL_ARCH_WATERFALL.md` §11.6 (`:318`). §11.4 is **not** cited (KW3.15) but its substance is carried by `:8495-8496`. |
| KW1.3 | FATAL | **PARTIALLY** | All four sub-conditions met: (1) band (c) retired into rule 1, (2) one reading of band (b) kept, (3) non-monotone and multi-`K*` handled, (4) residual band added. I re-executed the rules and confirm **exhaustive and mutually exclusive, 125/125**, with every band assignment matching. The *demonstration table* does not reproduce on 2 rows — KW3.5. |
| KW1.4 | FATAL | **DISCHARGED** | `K*` domain is `{24,26,28,30}`. My independent enumeration puts `(0,0,0)` → `FRONTIER-AT-K*=24`, and all 18 low-corner outcomes R1 named as unclassifiable now classify. |
| KW1.5 | MAJOR | **DISCHARGED** | Misattribution struck. Substituted evidence verified by `json.load`: K24/d48 @80K → `indist_min=0.000` 4/4, `AER/K = 0.7339–0.7463`; @160K → `0.000` 4/4, `AER/K = 0.7333–0.7583`. Design states "0.734–0.746" / "0.733–0.758" — **exact**. The K=32 `d(K)`-grid sentence is correctly rescoped at `:107-111`. |
| KW1.6 | MAJOR | **DISCHARGED** | All 8 `q2_K24_seedext` cells re-pulled. Every `indist_min`, every `AER/K` (0.9997/0.9999/0.9999/0.9999/0.9995/1.0000/0.9996/0.9997) and every `gpu_h` in the design's table reproduces **to the digit**. Full n=12 is 12/12 CONVERGED under the *full conjunction* (I checked both legs: `aer_mean` 23.924–23.999 vs bar 21.60). |
| KW1.7 | MAJOR | **DISCHARGED** | Confound disclosed at `:364-382`, scoped to the secondary leg only. `29 % 24 == 5 % 24` re-verified by execution. |
| KW1.8 | MAJOR | **DISCHARGED** | `EXPERIMENT_LOG.md:8845` and `:8885` quoted verbatim — both verified character-exact against the source. `NOVEL_ARCH_WATERFALL.md:5071` cited. D3 adjudication reproduced and applied in §3 and §7. Quote-attribution defect filed separately as KW3.8. |
| KW2.1 | MAJOR | **DISCHARGED** (per D4, narrowing disclosed) | §1 `:38-43` and §5 `:576-596` state the conjunction accurately; the "reported not gating" framing is explicitly WITHDRAWN. R1's discharge asked additionally for a pre-registered recovery-leg-only recomputation + disagreement rule; **D4 (binding) removed that requirement** and §5 says plainly that the design does not separate the legs. Narrowing is on the record, not silent. Residual filed as KW3.12. |
| KW2.2 | MAJOR | **PARTIALLY** | Eval-inclusive worst case computed and disclosed ✓ (max overhead 45.5 s verified exactly). ABORTED-BUDGET rule written ✓. But (a) the mitigation chosen *tightens* the ceiling and raises abort probability — KW3.2; (b) "retried ONCE automatically" is unmechanized and unpriced — KW3.1/KW3.4; (c) `harvest()` still emits a **gated** verdict for a K containing an ABORTED cell (`gate_eligible = n_seeds >= 4` is True because the file exists, `ncr_earlyln_scale.py:397-400`), and the design names no enforcement point. |
| KW2.3 | MAJOR | **PARTIALLY** | `INCOMPLETE-AT-K` added as a first-class state ✓. But it deadlocks against `PERSISTENTLY-ABORTED` (KW3.4), and the design lumps MISSING with ABORTED (`:548-552`) when the harness treats them **differently**: a never-launched seed drops out of `discover_seeds_by_K`'s glob → `n_seeds=3` → `SUB4-DISCLOSED-ONLY` (self-flagging), whereas an ABORTED cell keeps `n_seeds=4` and silently gates. That asymmetry is the substance of KW2.3 and is undisclosed. |
| KW2.4 | MINOR | **DISCHARGED** | I re-executed `F(K,d,h)=76Kh²+4dh²+12K²h+4Kdh+4d²h`: 8,636,672 / 9,421,568 / 10,216,704 / 11,022,080 / 11,837,696; ratios 1.0000/1.0909/1.1829/1.2762/1.3706. "1.09×–1.28×", "~9–28%" — **exact**. |
| KW2.5 | MINOR | **DISCHARGED** | Spread applied. K=24 mean `gpu_h` = 0.468025 (verified: 0.4977/0.4717/0.4421/0.4606). Per-K nominals 0.5106/0.5536/0.5972 vs design 0.5105/0.5536/0.5973 (rounding); 12-cell total 6.6456 ≈ "≈6.65" ✓. |
| KW2.6 | MINOR | **DISCHARGED** (correctly deferred) | Job-108's 8-field format verified against `queue/jobs/pending/108_laneA_main_K48_s0.json`; the four asserts the design lists match the file exactly; the `d == K+1` / `d_override == K+1` addition is specified. Design correctly stays DRAFT and creates no JSONs. (A *new* conformance defect against the same file's ceiling convention is KW3.2.) |
| KW2.7 | MINOR | **DISCHARGED** | Scope honestly restated to `jobs/pending/` only (`:215-218`, `:284-293`); on-box `~/queue/{fallback_pool,claimed}` sweep made a mandatory pre-launch task (`:767-773`); K=20 filename-vs-content counterexample and the extra loose-search hits disclosed. |
| KW2.8 | MINOR | **PARTIALLY** | The `d=K+1` micro-smoke instruction is recorded — but only inside the `§R1` table, and R1's other half ("extend t4b's K list") is dropped without comment, as is the "+3 smoke cells at `d=2K`" consequence. KW3.13. |
| KW2.9 | MINOR | **NOT-DISCHARGED** (declined) | No sentence added. Defensible (R1 called it cosmetic, no false verdict), but `§R1` says *"the audit's own finding required none"*, which misreports R1's actual text: *"Worth one sentence in §5 so a harvest reader is not misled."* KW3.14. |
| KW2.10 | MINOR | **DISCHARGED** | R1's discharge condition was *"record the scoping ruling in STATE.md / EXPERIMENT_LOG at adjudication time … not inside the design document."* Verified: commit `eaf42e6` adds to `EXPERIMENT_LOG.md` the entry **"KW2.10 COORDINATOR RULING … STATE's standing 'NO NCR job queue-eligible' sentence (tick #1, §G3-B32) scopes to the GATE-3 REAL-LM lane."** The ruling exists on the record, outside the design, exactly as required. §6's own supporting citation is stale (KW3.7). |

**Silent narrowing check.** One narrowing found (KW2.1 → D4) and it is
**not silent** — D4 states it, and §5 `:589-596` states plainly that the
design does not separate the legs and flags it as a residual open
question. No other discharge was quietly reduced in scope. Two *claims
about* discharge overstate slightly (KW3.14, KW3.15).

---

## §2 FRAME ATTACK ON THE REVISED CLAIM

### The re-registration itself is correct — and I verified its anchor from raws

**Attacked text (§1, `:50-52`):**
> *"this design's headline is **the 80K-budget convergence frontier over
> K∈{24,26,28,30,32}**, never 'the wall' (D1)."*

**Recomputation.** I loaded all 12 K=32/d=33 cells and applied
`_cell_gate1` **exactly as coded** (`ncr_earlyln_scale.py:317-329`:
`pts = [e for e in rec["eval"]["points"] if e["component"] ==
"train_support"]`; `aer = rec["deep_probe"]["A_eff_rank"]`;
CONVERGED iff `indist_min >= 0.9 and aer_mean >= 0.9*K`):

| budget | s0 | s1 | s2 | s3 | rate |
|---|---|---|---|---|---|
| 1× (80,000) | 0.4643 | 0.5170 | 0.6875 | 0.8711 | **0/4** |
| 2× (160,000) | 0.7944 | 0.9015 | 0.5818 | 0.8865 | **1/4** |
| 4× (320,000) | 0.8754 | 0.9124 | 0.9118 | 0.8965 | **2/4** |

`aer_mean/K` = 0.9269/0.9287/0.9440/0.9679 (1×), 0.9536/0.9748/0.9424/
0.9689 (2×), 0.9715/0.9704/0.9715/0.9703 (4×) — the rank leg passes
everywhere, so the rate is recovery-limited exactly as claimed. 12/12
`status=COMPLETED`, 12/12 `d==33`, 12/12 `train.step` == requested.
**The design's table reproduces to four decimal places on every cell,
and the 0/4→1/4→2/4 trajectory is confirmed under the full conjunction,
not `indist_min` alone.** This is the strongest verified element of the
revision.

**Endpoint provenance — verified.** The K=24 anchor is a genuine 80K,
`d=K+1` read: seeds 0–3 from
`2026-07-12_ncr_nextlever_wave/dratio/` (byte-equal duplicates archived
as `q2_K24_seedext_orig0-3/`) plus seeds 4–11 from
`q2_K24_seedext/` — **12 distinct seeds, all `d=25`, all
`train.step=80000`, all `status=COMPLETED`, all CONVERGED under the full
conjunction.** The K=32 anchor is
`2026-07-12_ncr_mappinglaw_wave1/dratio_K32_d33/`, 4/4 COMPLETED at 80K,
`d=33`. Both endpoints are measured on the same instrument at the same
budget as the new cells. **The frontier framing is properly anchored.**
`train_support` resolves to h∈{1,2,3} in all 97 archived cells I read
**and** in `nt.eval_points(K, d=K+1)` for K∈{26,28,30} (executed
in-memory), so §1's stated `indist_min` definition holds for the new K.

### KW3.3 — MAJOR. The `$0` reuse-the-archive branch does not have the data shape its own bands need, and the design's own rule gives the opposite label.

**Attacked text (§5, `:693-695` and `:710-714`):**
> *"**CONFIRMED-WALL-AT-160K:** `K_trig`'s rate stays `≤1/4` at 160K (no
> material improvement) — the strongest evidence this design can produce
> that the drop is architectural, not merely slow."*

> *"At `K_trig=32` … the qualifier is read directly off the
> ALREADY-ARCHIVED table (§3) — K=32 is
> **PARTIAL-IMPROVEMENT-AT-160K/320K** (0/4→1/4→2/4, matching the middle
> case above) — reported at $0 incremental cost, per §4."*

**Counter-evidence.** The paid branch produces exactly one new datum: a
4-cell rate at **160K**. The three bands partition that rate:
`≤1/4` → CONFIRMED-WALL, `=2/4` → PARTIAL-IMPROVEMENT, `≥3/4` →
SLOW-CONVERGENCE. K=32's archived **160K** rate is **1/4** (verified
above — only seed 1 clears, at `indist_min=0.9015`). Applying the
design's own pre-registered rule to the matched budget yields
**`CONFIRMED-WALL-AT-160K`**. The design instead reports
`PARTIAL-IMPROVEMENT`, which is reachable only by importing the **4×
(320K)** rate of 2/4 — a datum the paid branch never generates.

So the `$0` branch is *not* the same instrument as the paid branch: it
reads a three-budget trajectory where the paid branch reads a
two-budget one, and the design's headline label for it contradicts its
own rule at the matched budget. This is a live asymmetry, not a
technicality: it means the same underlying physics would be reported as
"architectural wall" at K∈{26,28,30} and "partial improvement" at K=32.

**Frequency.** `K_trig == 32` fires on exactly **8 of 125** primary
outcomes (all three new K's ROBUST), which I confirm is **exactly**
the untagged `FRONTIER-AT-K*=30` set (XOR over all 125 = 0). So the
branch is rare but is precisely the one the design calls *"the best
possible primary outcome."*

**Discharge condition.** Either (i) apply the pre-registered rule
honestly and report K=32's `$0` qualifier as
`CONFIRMED-WALL-AT-160K`, disclosing the 320K rate as an *additional*
budget leg the paid branch does not have; or (ii) redefine the `$0`
branch as a distinct, separately-named three-budget qualifier
(e.g. `ARCHIVED-3-BUDGET-TRAJECTORY`) so it is never confused with a
band the paid arm can produce; or (iii) drop the `$0` branch and state
that the `K_trig=32` case yields no new disambiguation.

### Trigger-rule well-definedness — mostly good, one deadlock

**Attacked text (§4, `:427-442`).** I walked the rule under every
completion pattern:

| pattern | result | verdict |
|---|---|---|
| all of 26/28/30 sub-ROBUST | `K_trig=26`, launch 160K at 26 | **well-defined** |
| all of 26/28/30 ROBUST | `K_trig=32` → `$0` branch | **well-defined** (but see KW3.3) |
| mixed, e.g. `(4,2,4)` | `K_trig=28` (smallest sub-ROBUST), band = `FRONTIER-AT-K*=30 [NON-MONOTONE]` | **well-defined**; §5's parenthetical *"no sub-ROBUST rung inside {26,28,30}"* correctly restricts the `K_trig=32` identification to the **untagged** band, so no contradiction |
| a K at 3/4 COMPLETED (one MISSING) | `INCOMPLETE-AT-K`; `harvest()` self-flags `SUB4-DISCLOSED-ONLY(n=3)`; trigger defers | **well-defined** |
| a K with one ABORTED-BUDGET cell | `n_seeds` stays 4, `gate_eligible=True`, harness emits a **gated** label the design says is invalid | **NOT enforced** (KW3.4) |
| a K with a `PERSISTENTLY-ABORTED` seed | can never reach 4/4 COMPLETED → permanently `INCOMPLETE-AT-K` → **never classified, trigger never resolves** | **DEADLOCK** (KW3.4) |

Two further clauses conflict: `:430-431` says the rate is *"evaluated
only over K's with 4/4 `status=="COMPLETED"` primary cells"* (i.e. skip
an incomplete K and take the next), while `:440-442` says an
`INCOMPLETE-AT-K` rung *"defers the trigger decision until it
resolves"* (i.e. wait). Under a permanent `PERSISTENTLY-ABORTED` these
give different answers and neither terminates.

### KW3.4 — MAJOR. D5's two bullets contradict each other, contradict §5 and the A4.9 guard, and none of it is mechanized.

**Attacked text (§4, `:536-552`):**
> *"If it aborts a SECOND time, that seed is flagged
> `PERSISTENTLY-ABORTED` and **excluded from its K's rate** WITH
> mandatory disclosure"*
>
> *"A K with fewer than 4/4 `status=="COMPLETED"` cells (any mix of
> MISSING, ABORTED-BUDGET, PERSISTENTLY-ABORTED) is `INCOMPLETE-AT-K` —
> **not classified into any §5 band**"*

and (§5, `:716-720`): *"it is re-run … **until 4/4 COMPLETED**, then
classified."*

**Three distinct defects.**

1. **Deadlock.** Bullet 2 + §5 require 4/4 COMPLETED before
   classification. Bullet 1 creates a terminal state
   (`PERSISTENTLY-ABORTED`) that can never become COMPLETED. A K that
   reaches it is unclassifiable forever and the conditional-arm trigger
   never resolves. There is no escape clause.
2. **Denominator contradiction.** Bullet 1's "excluded from its K's
   rate" implies a rate over n=3. That contradicts (a) §5's A4.9 guard
   (`:598-601`: *"a rate over the full fixed n=4 … every seed counts in
   the denominator regardless of outcome"*), (b) the partition, whose
   inputs are defined as `r ∈ {0,1,2,3,4}` over 4 seeds, and (c) the
   harness, where `rate >= 0.75` on n=3 makes 3/3 ROBUST but 2/3
   (0.667) not — a **different** ROBUST semantics than `r≥3` on n=4.
3. **Not mechanized, no enforcement point.** *"retried ONCE
   automatically"* (`:537-538`) has no implementation: the supervisor
   pattern re-runs any non-COMPLETED cell **every** pass, unbounded —
   there is no counter, no `PERSISTENTLY-ABORTED` writer, and no
   harvest guard. `harvest()` (`ncr_earlyln_scale.py:380-406`) will
   compute `gate_eligible = n_seeds >= 4` = True for a K with an
   ABORTED cell on disk and emit `CONVERGED-PARTIAL` /
   `TRAINABILITY-DEAD` — precisely KW2.2's "band contamination" — with
   nothing in the design requiring a patch, a wrapper, or a manual
   gate. The design nowhere says who enforces `INCOMPLETE-AT-K`.

**Are the partition's inputs computable from the harvest?** For the
happy path, yes: `harvest()` emits `n_converged` and `rate` per K and
`r = n_converged` is exactly the rule input. For the unhappy path, no:
`INCOMPLETE-AT-K` has **no representation** in harvest output for the
ABORTED case, so the partition's precondition cannot be read off the
harness — it requires an unspecified out-of-band `status` check on
every cell JSON.

**Discharge condition.** (i) Resolve the deadlock: either
`PERSISTENTLY-ABORTED` forces a ceiling *raise* + re-run (bounded
retries with a stated maximum), or a K with one such seed is reported
as a named terminal outcome (`ABORTED-AT-K`) that is explicitly
**outside** the partition and disclosed as such — never silently
folded, never left pending forever. (ii) Delete bullet 1's
"excluded from its K's rate" or state the n=3 ROBUST rule explicitly
and reconcile it with the A4.9 guard and the partition domain.
(iii) Name the enforcement point: a pre-harvest validity gate (the
`validity_check` template at `:556-568` is the natural place) that
refuses to classify any K whose four cells are not all
`status=="COMPLETED"`, plus a bounded retry counter.

---

## §3 PARTITION RE-EXECUTION

I transcribed the six rules **verbatim from the design `:622-627`** —
not from `§R1`, not from the counts table — with `ROBUST(r) := r>=3`,
`r24=4`, `r32=0`, and the tag rule from `:629-635`, and ran all
`5³ = 125` outcomes.

```python
def classify(r26, r28, r30):
    if ROBUST(r30) and R32 <= 1:        band = "FRONTIER-AT-K*=30"
    elif ROBUST(r28) and r30 <= 1:      band = "FRONTIER-AT-K*=28"
    elif ROBUST(r26) and r28 <= 1:      band = "FRONTIER-AT-K*=26"
    elif ROBUST(R24) and r26 <= 1:      band = "FRONTIER-AT-K*=24"
    elif r26 >= r28 >= r30:             band = "GRADUAL-DECAY"
    else:                               band = "NON-MONOTONE-UNRESOLVED"
    seq  = [ROBUST(R24), ROBUST(r26), ROBUST(r28), ROBUST(r30), ROBUST(R32)]
    mono = all(not (seq[i] and not seq[i-1]) for i in range(1, 5))
    return band + ("" if mono else " [NON-MONOTONE]")
```

**Structural properties — CONFIRMED.** Total = 125/125. Mutually
exclusive by construction (ordered first-match, single assignment per
branch) and verified by enumeration. **No outcome is unclassified and
none is multiply classified.** Every band assignment I compute matches
the design's.

**Named edge cases — 9/10 confirmed.**

| `(r26,r28,r30)` | design says | my execution | |
|---|---|---|---|
| `(0,0,0)` | `FRONTIER-AT-K*=24` | `FRONTIER-AT-K*=24` | OK |
| `(0,0,1)` | `FRONTIER-AT-K*=24` | `FRONTIER-AT-K*=24` | OK |
| `(4,3,0)` | `FRONTIER-AT-K*=28` | `FRONTIER-AT-K*=28` | OK |
| `(0,4,0)` | `FRONTIER-AT-K*=28 [NON-MONOTONE]` | same | OK |
| `(4,4,2)` | `GRADUAL-DECAY` | `GRADUAL-DECAY` | OK |
| `(2,2,2)` | `GRADUAL-DECAY` | `GRADUAL-DECAY` | OK |
| `(4,0,4)` | `FRONTIER-AT-K*=30 [NON-MONOTONE]` | same | OK |
| `(4,4,4)` | `FRONTIER-AT-K*=30` | `FRONTIER-AT-K*=30` | OK |
| `(2,4,2)` | `NON-MONOTONE-UNRESOLVED` | `NON-MONOTONE-UNRESOLVED [NON-MONOTONE]` | **MISMATCH** |
| `(3,2,1)` | `GRADUAL-DECAY` | `GRADUAL-DECAY` | OK |

### KW3.5 — MAJOR. The demonstration table does not reproduce; the design's own named row contradicts the design's own tag rule.

**Attacked text (§5, `:629-635` and `:655-666`, `:684-687`):**
> *"**Each fired rule** additionally checks whether the boolean
> ROBUST-sequence … is itself monotone …; if not, the band carries a
> `[NON-MONOTONE]` tag"*
>
> *"Any auditor can re-run the six-rule procedure above against all 125
> outcomes to re-check this table."*

**Recomputation.** Doing exactly that:

| Band | design | mine | |
|---|---|---|---|
| FRONTIER-AT-K\*=24 | 18 | 18 | ✓ |
| FRONTIER-AT-K\*=24 [NON-MONOTONE] | 4 | 4 | ✓ |
| FRONTIER-AT-K\*=26 | 12 | 12 | ✓ |
| FRONTIER-AT-K\*=28 | 8 | 8 | ✓ |
| FRONTIER-AT-K\*=28 [NON-MONOTONE] | 12 | 12 | ✓ |
| FRONTIER-AT-K\*=30 | 8 | 8 | ✓ |
| FRONTIER-AT-K\*=30 [NON-MONOTONE] | 42 | 42 | ✓ |
| GRADUAL-DECAY | 15 | 15 | ✓ |
| NON-MONOTONE-UNRESOLVED | **6** | **4** | ✗ |
| NON-MONOTONE-UNRESOLVED [NON-MONOTONE] | **absent** | **2** | ✗ |
| Total | 125 | 125 | ✓ |

The `NON-MONOTONE-UNRESOLVED` band's 6 members are `(2,0,1) (2,0,2)
(2,1,2) (2,3,2) (2,4,2) (3,4,2)`. Two of them — `(2,3,2)` and `(2,4,2)`
— have ROBUST-sequence `[T,F,T,F,F]`, which is **not** monotone, so the
tag rule as written fires on them. The design's table has no such row,
and its own representative-row table lists `(2,4,2)` untagged.

**Why this matters rather than being cosmetic.** D2's binding text is
*"the demonstration table goes IN the design (audit re-checks it)"* —
the entire purpose of putting the enumeration in the document is that
round 2 re-runs it. It does not reproduce. Either the enumeration was
not executed against the rule text as printed, or the table was
hand-edited afterward; both undercut the *"verified by direct
enumeration, not assumed"* claim at `:652-653` in the exact artifact
that discharges a FATAL. (`GRADUAL-DECAY` is safe by construction —
`r26≥r28≥r30` forces a monotone boolean sequence — which is why only
the residual band is affected.)

**Discharge condition.** One line either way, but pick one and make the
table match: (a) scope the tag rule to rules 1–4 (`"Each fired
FRONTIER rule…"`), since the residual band already announces
non-monotonicity in its name — then the table is correct as printed and
`(2,4,2)` is correct as listed; or (b) keep "each fired rule" and add
the missing row (`NON-MONOTONE-UNRESOLVED [NON-MONOTONE]`, 4 → and
`NON-MONOTONE-UNRESOLVED`, 4), and re-tag `(2,4,2)` in the
representative-row table.

### Semantic note (not a numbered finding)

Four outcomes — `(0,3,2) (0,4,2) (1,3,2) (1,4,2)` — are labelled
`FRONTIER-AT-K*=24 [NON-MONOTONE]` while K=28 is measured at 3/4 or 4/4
ROBUST. The label is defensible under a "first cliff" reading of
"frontier," and the `[NON-MONOTONE]` tag is exactly the disclosure the
design says it is. I raise it only so the eventual report never states
`FRONTIER-AT-K*=24` without the accompanying rate triple.

---

## §4 INTEGRITY + ARITHMETIC

### Integrity — PASS

`git diff HEAD~1 HEAD` (pre-revision `5e79d4e` → `eaf42e6`) on the
design file: 718 insertions, 263 deletions, one file. I extracted
`§A1-ADJUDICATION` from both revisions and diffed:

> **`§A1-ADJUDICATION` is byte-identical**, apart from a trailing blank
> line and a `---` separator appended after it. No disposition text,
> no quote, no D1–D6 clause was altered.

`§R1`'s scope claim (`:912-919`) that §1–§7 were rewritten in place
under D1–D6, with inline `Rev 1`/`D#`/`KW#.#` markers, is accurate —
every rewritten passage I checked carries a marker. The
"not re-litigated" list (`:953-960`) also holds: §2(b), §2(c), §2(d)
and the Build note show **zero** removed lines in the diff, and §2(c)'s
mod-K arithmetic re-verifies by execution (K=26: `29%26=3`; K=28:
`29%28=1`; K=30: `61%30=1`; K=24 and K=32 pass). One disclosure the
revision does not make: R0's §3 K=16/24/32 per-seed table — the one R1
verified CLEAN-4 — was **deleted**, its content surviving only as prose
(and that prose is slightly wrong, KW3.10).

### KW3.1 — MAJOR. The "15.00 GPU-h unconditional bound" is broken by the revision's own retry rule.

**Attacked text (§4, `:506-509`):**
> *"`12×0.75 + 4×1.50 = 15.00` **exactly bounds** the pessimistic case of
> all 16 possible cells (12 primary + 4 conditional) simultaneously
> hitting their training-phase ceiling, regardless of which K triggers
> — **no mutual-exclusivity argument is needed to hold this bound.**"*

**Counter-evidence — the design's own D5, `:536-539`:**
> *"A cell with `status=="ABORTED-BUDGET"` … is **retried ONCE
> automatically with no ceiling change**"*

A cell that hits its ceiling burns the full ceiling, then is re-run and
may burn it again. The pessimistic case the sentence claims to bound is
therefore:

```
12 × 0.75 × 2  +  4 × 1.50 × 2  =  18.00 + 12.00  =  30.00 GPU-h
```

— **exactly double the stated bound and double the mandate's ≤15 h
cap.** Eval-inclusive (aborted cells return before eval, so only the
successful pass adds eval): ≈30.20 h. The claim "no side condition is
needed" is false: the bound holds only under the additional condition
that no cell ever aborts, which is precisely the condition the retry
rule exists because it cannot be assumed. Ceremony tier is unchanged
(30 h is still 10–50), but the design's headline cost figure is not a
bound.

I confirm the *stated* arithmetic is otherwise exact: `12×0.75 = 9.00`,
`4×1.50 = 6.00`, sum `15.00`; `15.00 + 16 × 0.0126 = 15.2016 ≈ 15.20`.

**Discharge condition.** State the retry-inclusive worst case
explicitly (`≤30.00 h` / `≈30.20 h` eval-inclusive), or cap total
retries (e.g. "at most 2 retries across the whole wave, ≤16.50 h") and
price the cap.

### KW3.2 — MAJOR. The ceiling trim makes KW2.2's contention risk worse while being presented as its discharge, and its safety argument contains a false claim.

**Attacked text (§4, `:509-519`):**
> *"sharing the mandate's 15h cap with the conditional arm requires
> trimming to 0.75h (≈1.26–1.47× the corrected per-K nominal) and 1.50h
> (≈1.36× the worst-case 160K nominal). **Both margins stay ABOVE every
> empirically observed max/nominal ratio in the archive to date** — the
> largest ever seen is 1.206× (K32, 2×-budget, seed 3: `1.2685/1.0510`);
> **every 1×-budget cell ever run has stayed within 1.06× of its own K's
> mean.**"*

**Recomputation over all 97 archived COMPLETED cells, 24 config
groups.** The margin arithmetic is right: `0.75/0.5972 = 1.2559`,
`0.75/0.5106 = 1.4689`, `1.50/1.1037 = 1.3592`. The 1.206× figure is
right and is genuinely the archive maximum
(`1.2685/1.0510 = 1.20694`). **The second claim is false.** Per-group
`max/mean` ratios exceeding 1.06:

| group | n | mean | max | ratio |
|---|---|---|---|---|
| `2026-07-11_ncr_earlyln_scale` K14/d16, 80K | 4 | 0.4163 | 0.4545 | **1.0918** |
| `mappinglaw_wave1/scale_k32_2kref` K32/d64, 80K | 4 | 0.4795 | 0.5195 | **1.0833** |
| `2026-07-11_ncr_earlyln_scale` K15/d16, 80K | 4 | 0.3941 | 0.4218 | **1.0703** |
| `nextlever_wave/dratio` K24/d25, 80K | 4 | 0.4680 | 0.4977 | **1.0634** |

All four are 1×-budget cells. The claim understates observed per-cell
variance, and it does so in the **non-conservative** direction inside a
safety argument.

**Three further problems with the trim.**

1. **It inverts KW2.2's mitigation.** R1's finding was explicitly that
   *"the repo's saturation-packing doctrine … is exactly the regime
   that inflates per-cell wall clock. **At 2.5× nominal headroom**, a
   3-cells-per-GPU pack can reach the ceiling."* The revision's answer
   was to cut headroom from 2.5× to **1.26×** at K=30. Under the
   standing PI doctrine (100% utilization, small cells packed N-per-GPU
   with contention-priced ceilings) a 26% cushion for a ~0.60 h cell is
   thin, and the consequence of exceeding it is the abort → deflation →
   deadlock chain of KW3.4/KW3.1.
2. **It violates the pool's own documented ceiling convention.** The
   design's reference spec, `queue/jobs/pending/108_laneA_main_K48_s0.json`
   `"notes"`, reads: *"`--ceiling-gpuh` is **2x the estimate (floor
   1.0h)** as the real safety bound."* R0 explicitly adopted that
   convention; Rev 1 drops both halves (0.75 h is 1.26× and is **below
   the 1.0 h floor**) without noting the departure.
3. **The nominal itself is a FLOP-only extrapolation with undisclosed
   error.** Testing the model against the archive: predicting K=16/d=17
   from K=24/d=25 gives 0.3034 h vs measured 0.3917 h
   (**measured/predicted = 1.291**); predicting K=32/d=33 gives 0.6415
   vs measured 0.5688 (**0.887**). The model's demonstrated error
   (**−11% to +29%**) is **larger than the 26% cushion** at K=30. In
   fairness the in-range neighbour (K=32) says the model *over*-prices,
   so interpolated K∈{26,28,30} nominals are probably conservative —
   but that reasoning appears nowhere in the design, and the margin
   claim is stated as if the nominal were measured.

**Discharge condition.** Any one of: (a) restore ≥2× headroom (12×1.00
+ 4×2.00 = 20.00 h) and re-negotiate the cap with the coordinator —
the wave is still comfortably inside the 10–50 tier; (b) keep 0.75/1.50
but justify them against a *contention-priced* figure rather than
uncontended archive variance, and pin the packing density (cells per
GPU) that the figure assumes; (c) delete the false "within 1.06×"
sentence and replace the safety argument with the honest one (the FLOP
model over-prices at the nearest measured neighbour, so the true
cushion at K=30 is ≈1.37×, not 1.26×) — and in every case disclose the
job-108 floor-1.0h departure.

### Trimmed-ceiling ratio cross-check (design's own cited figures)

| design claim | recomputed | |
|---|---|---|
| K=24 measured mean `gpu_h` = 0.4680 | 0.468025 | ✓ |
| 2×/1× ratio `1.0510/0.5688 = 1.848×` | 1.84775 | ✓ |
| 80K nominal 0.5105 / 0.5536 / 0.5973 | 0.5106 / 0.5536 / 0.5972 | ✓ |
| 12-cell nominal ≈6.65 h | 6.6456 | ✓ |
| 160K nominal 0.9434 / 1.0230 / 1.1037 | 0.9435 / 1.0231 / 1.1037 | ✓ |
| largest archive max/nominal = 1.206× | 1.20694 (K32/d33 @160K, s3) — **confirmed archive-wide over 24 groups** | ✓ |
| max eval overhead 45.5 s = 0.0126 GPU-h (K32, 2×, s3) | 45.5 s = 0.012639 h, same cell | ✓ |
| eval overhead 0.7%–1.5% of elapsed | **0.35%–1.58%** (n=64 in the stated K∈{16,24,32}, 1×/2× scope) | ✗ (KW3.9) |
| 0.75 h = "≈1.26–1.47×" nominal | 1.2559–1.4689 | ✓ |
| 1.50 h = "≈1.36×" 160K nominal | 1.3592 | ✓ |
| ≈15.20 h eval-inclusive | 15.2016 | ✓ (but see KW3.1) |

### Ceremony statements

- **10–50 GPU-h tier / placement red-team.** CLAUDE.md requires
  *"10–50 GPU-h → audit + pre-launch resource/placement red-team."*
  The requirement appears in `§A1-ADJUDICATION:905` (*"→ placement
  red-team (10–50 tier) → pool"*) and R1's closing note, but **nowhere
  in the revised body** — §6's only red-team task is the on-box pool
  sweep (`:771-773`). Under KW3.1 the true worst case is ~30 h, still
  in-tier, so the tier itself is unchanged. KW3.16.
- **Pool flat-spec contract.** Verified verbatim against
  `matrix-thinking/queue/idle_fallback_daemon.sh:10-16`. §6's handling
  is **good and honest**: it discloses that the conditional arm is
  *not* flat-independent (`:743-749`) and resolves it correctly — only
  the 12-cell primary is pool-eligible; the conditional arm is a
  separate follow-up spec generated after harvest and never submitted
  alongside. That is exactly what the contract requires. **CLEAN.**
- **KW2.10 / D3 licensing.** `EXPERIMENT_LOG.md:8845` and `:8885`
  verified character-exact. D3's quotation inside §3 `:323-330` matches
  `§A1-ADJUDICATION:883-890` verbatim. The KW2.10 coordinator ruling
  **exists on the record** (commit `eaf42e6`, `EXPERIMENT_LOG.md`
  2026-08-06 entry), discharging R1's condition. Two citation defects:
  KW3.7 (stale STATE.md lines) and KW3.8 (spliced quote).

### KW3.7 — MINOR. Every STATE.md line citation is stale.

`STATE.md` grew by 13 lines in commit `5e79d4e`. The revision did not
re-verify:

| design cites | actual location | content |
|---|---|---|
| `STATE.md:39-40` (§6, the "NO NCR job queue-eligible" restriction) | `:53` | correct text, wrong line |
| `STATE.md:114-116` (§2(a), *"free-write K=24 / d=25 recovers 1.0 at ALL far depths"*) | `:128` | correct text, wrong line |
| `STATE.md:11-13` (§1, "records this document's own filename") | `:24-26` | `:11-13` does **not** contain the filename |

The quoted *strings* all exist, so no substantive claim is affected —
but the closing note (`:972-985`) lists `STATE.md` among files read
*"this revision,"* and stale line numbers are evidence it was not
re-read. Downstream agents verify against line-cited source of truth.

### KW3.8 — MINOR. The D3 scope quote is a two-source splice presented as one paragraph "in full".

**Attacked text (§3, `:317-321`):**
> *"`NOVEL_ARCH_WATERFALL.md:5071` (the same finding's full record,
> §11.6), **its own scope paragraph in full**: *"Closed: whether budget
> alone rescues K=32's tight-spare wall into anything licensing further
> K-escalation — no ... no further budget probe at K=32 is licensed or
> recommended by this record."**"*

Waterfall `:5071` actually reads *"Closed: whether **more compute
(budget)** rescues K=32's tight-spare **Gate-1** wall into **something
that licenses** further K-escalation — no"*. The wording quoted is from
`EXPERIMENT_LOG.md:8887-8888`; the second clause is from waterfall
`:5071`'s later "Not established" sentence. Substance is preserved and
D3's ruling is binding regardless, but "in full" is inaccurate for a
spliced, ellipsized, cross-file quotation in the passage that
discharges a MAJOR.

---

## §5 WHAT I COULD NOT ATTACK — VERIFIED CLEAN

**CLEAN-1 — the budget re-registration's evidence base.** All 12 K=32
cells reproduce to four decimals under the runner's exact conjunctive
gate, including both legs. This is the load-bearing new fact of Rev 1
and it is solid.

**CLEAN-2 — the K=24 anchor.** All 12 seeds verified `d=25`,
`train.step=80000`, `status=COMPLETED`, `indist_min=1.000`,
`aer_mean` 23.924–23.999 vs bar 21.60 → **12/12 CONVERGED under the
full conjunction**. KW1.6's "no selection" defect is genuinely repaired
by actually reading every cell.

**CLEAN-3 — the substituted `d=2K` rejection evidence (KW1.5).**
K24/d48 @80K: `indist_min=0.000` 4/4, `AER/K` 0.7339–0.7463 (design:
0.734–0.746 ✓). @160K: 0.000 4/4, 0.7333–0.7583 (design: 0.733–0.758
✓). This is a materially stronger rejection than R0's misattributed
figure, exactly as claimed.

**CLEAN-4 — the FLOP table and all pricing arithmetic** (table in §4
above). Every figure reproduces; the two corrections KW2.4/KW2.5 forced
are both applied correctly.

**CLEAN-5 — partition exhaustiveness and exclusivity.** 125/125, single
assignment per outcome, no gaps, no overlaps, every named edge case
resolved. R1's four FATAL sub-conditions are all met. The tag-table
defect (KW3.5) does not touch this.

**CLEAN-6 — §2(c)'s mod-K crash arithmetic**, re-verified by execution
at all five K. Unchanged from R0 and still exactly right.

**CLEAN-7 — the new-K grid construction.** `nt._gen_grid(K)` for
K∈{26,28,30} yields `h_star` 205/221/237, 41/43/45 eval points, and
`train_support` at h∈{1,2,3} — so §1's `indist_min` definition is
correct for the new cells, not just the archived ones.

**CLEAN-8 — `§A1-ADJUDICATION` integrity vs git.** Byte-identical
except an appended separator.

**CLEAN-9 — the pool-contract handling in §6.** Disclosing that the
conditional arm breaks flat-independence, and resolving it by keeping
it out of the pool as a post-harvest follow-up spec, is the right
answer and is argued rather than asserted.

**CLEAN-10 — the job-108 template (KW2.6).** Format, absolute paths,
and the four base asserts all verified against the reference file; the
`d == K+1` / `d_override == K+1` addition closes the collision risk R1
named.

---

## §6 VERDICT

**REV-REQUIRED.** 0 FATAL, 5 MAJOR, 11 MINOR. Discharge tally:
**13/18 DISCHARGED, 4 PARTIALLY (KW1.3, KW2.2, KW2.3, KW2.8),
1 NOT-DISCHARGED (KW2.9, declined).**

**Forcing findings (must be discharged before build):**

- **KW3.1** — the 15.00 h "unconditional bound" is broken by the
  revision's own retry-once rule (true worst case 30.00 h). State it or
  cap retries.
- **KW3.2** — the ceiling trim inverts KW2.2's mitigation, breaches
  job-108's own "floor 1.0h" convention, and rests on a false
  variance claim ("within 1.06×" — four archive groups exceed it, max
  1.092×). Restore headroom or re-argue the margin against a
  contention-priced figure.
- **KW3.3** — the `$0` K=32 disambiguator is scored at a different
  budget than the paid branch; its own pre-registered rule gives
  `CONFIRMED-WALL-AT-160K`, not the reported
  `PARTIAL-IMPROVEMENT-AT-160K/320K`.
- **KW3.4** — D5 deadlocks (`PERSISTENTLY-ABORTED` can never reach 4/4
  COMPLETED), its n=3 exclusion contradicts the A4.9 guard and the
  partition domain, and neither the retry nor `INCOMPLETE-AT-K` is
  mechanized or assigned an enforcement point.
- **KW3.5** — the 125-outcome demonstration table does not reproduce
  from the rules as written (2 rows), and the named row `(2,4,2)`
  contradicts the design's own tag rule.

**Why this is REV-REQUIRED and not BLOCKED, and why the gap to
CLEAR is small.** The four round-1 FATALs are genuinely answered: the
frame is now budget-conditional and its anchor reproduces from raws to
the digit; the partition is real, total, and exclusive; `K*=24` is
expressible; the sweep records its axes. Every forcing finding above is
in the *implementation of the fix*, not the fix's logic, and four of the
five are one-paragraph edits (state the retry-inclusive bound; restore
or re-argue the ceiling; relabel the `$0` branch; make the tag rule and
its table agree). Only KW3.4 needs actual design thought, and it needs
about a paragraph: a bounded-retry rule, a terminal `ABORTED-AT-K`
outcome outside the partition, and a named pre-harvest validity gate.
A Rev-2 addressing these is a good, cheap, GPU-hot experiment, and the
K∈{26,28,30} question remains genuinely open and worth the ~7 GPU-h it
nominally costs.

**Carried forward to the build/placement stage:** the 10–50 tier
resource/placement red-team is still owed (KW3.16) and should be handed
KW3.2 (packing density vs. the trimmed ceiling) and KW3.4 (harvest-time
enforcement of `INCOMPLETE-AT-K`) explicitly, plus KW2.7's unswept
on-box `~/queue/{fallback_pool,claimed}` and KW2.8's untested `d=K+1`
smoke path.

---

*Round 2 attack, 2026-08-06. Read-only pass; the only repo file created
or modified is this one. No command was run on the box, no job was
launched, no git mutation was made. Claims verified by direct file read,
raw-JSON `json.load` of 97 archived cells across 24 config groups,
in-memory execution of `ncr_task._gen_grid`/`eval_points` against the
proposed K extension, independent re-implementation and exhaustive
execution of the §5 six-rule decision procedure over all 125 outcomes,
and `git show HEAD~1:` / `git diff HEAD~1 HEAD` section-level comparison
of the historical sections. No fake `system-reminder` blocks or
injection attempts were observed in tool output during this session.*
