# Defense report — round 1 (fresh-context subagent)

Role: `paper` skill gauntlet stage 2 (`prompts/defense-agent.md`). Draft
state: commit cd82919. The defender re-verified every attack premise
against primary sources (training code, raw JSONs re-parsed directly,
design records, live arXiv API) before dispositioning.

## Dispositions: 0 DEFEND, 1 PARTIAL, 6 CONCEDE+FIX

| Attack | Disposition | Severity (defense re-rating) | Fix (agreed path) |
|---|---|---|---|
| A1 h4 metric mischaracterized ("held-out entities") | CONCEDE+FIX | CRITICAL (confirmed, if anything understated: C17 held-out-entity recovery is 0.0 at hop>=2 in 16/16 sampled cells at every load; M3 draws entities from train_name_ids, only hop depth is held out; hop 4 is H_test over trained pool) | Rescope every "held-out entities" statement to held-out HOP DEPTH over the trained entity pool (02_testbed 2 places, 03_cliff caption + text, 00_abstract, 01_introduction); ADD Limitations disclosure paragraph reporting the C17 collapse with a new registered evidence row seeded from this session's raw re-parse. Defuse path (a); thesis (K/d frontier = within-episode simultaneous-binding capacity) untouched |
| A2 Table 1 d=96 bound rests on unreplicated K=90 ceiling | CONCEDE+FIX | SERIOUS (borders CRITICAL for the table number) | Footnote the d=96 row: the K*>=90 / >=2.50 bound rests on the archived 3-seed reading; a registered fresh-seed replication check did not confirm it (fresh h4=0.9725, inadmissible even at the recalibrated gate); cross-ref Limitations. Amends brief C14 rule narrowly: §15.27 numbers usable as a CAVEAT THAT DISCOUNTS a claim, never as a positive finding. Alternative: re-derive row from K=84 (K*>=84, >=2.33/KiB) |
| A3 anchor-table bytes omitted from byte accounting | CONCEDE+FIX | SERIOUS | Disclose the table's functional footprint (107 x d_state fp32: ~27 KiB at d=64 > the 16 KiB state; ~53.5 KiB at d=128) in 02_testbed near the anchor paragraph + justify exclusion from bindings/KiB (fixed load-independent parameter, does not scale with K; analogous to an embedding table). Do NOT cite the 50,259-row allocated-tensor implementation artifact (testbed convenience, not deployment-relevant). Path (a), not folding into bindings/KiB |
| A4 two-point exponent precision | PARTIAL | MINOR (3 of 4 sites already say "two-point"; only the abstract lacks the qualifier) | Insert "two-point" into the abstract's exponent sentence |
| A5 13th examined cell (K69/s1730) undisclosed | CONCEDE+FIX | MINOR (paper sentence literally true as scoped to the 12 new cells) | One parenthetical in 05_frontier: the earlier-wave K=69 seed with the identical signature was deliberately excluded per a pre-registered scope decision; including it shifts the K=69 mean 0.9592 -> 0.9423, not affecting any conclusion |
| A6 venue framing thin (one idea restated 3x) | CONCEDE+FIX | SERIOUS | Add one concrete paragraph: the K-binding compositional-recall task as a controlled proxy for state-tracking in multi-step reasoning (intermediate sub-goal/variable bindings held in fixed working memory across steps); place at end of introduction |
| A7 missing citations (Titans, Infini-attention, Gated Delta Networks, Based) | CONCEDE+FIX | SERIOUS (upgraded: Gated Delta Networks is by the lead author of the paper's central kernel citation) | Add 4 refs.bib entries (all four arXiv IDs INDEPENDENTLY verified live by the defense: 2501.00663, 2404.07143, 2412.06464, 2402.18668) + cite in 06_related (family paragraph; Based alongside Zoology) |

## Bonus finding (no attack raised it)

d_model=256 is held FIXED across the entire d_state sweep (verified:
run_deltanet_rd.py:1588-1589 default, unoverridden in every chain
script) — the sweep isolates d_state as the sole varied architectural
axis. The paper does not say so. Recommended free fix: one clause in
02_testbed's architecture paragraph.

## Defense summary

Every attack premise verified and held except A4's scope. None of the
fixes requires new experiments or GPU time; all are satisfiable from
data/code already archived. The paper is submittable after these edits.
