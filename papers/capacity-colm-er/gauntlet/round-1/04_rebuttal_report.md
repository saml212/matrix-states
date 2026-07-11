# Rebuttal report — round 1 (fresh-context subagent; runs 3rd, artifact 04)

Role: `paper` skill gauntlet stage 3 (`prompts/rebuttal-agent.md`). Draft
state: commit 32d1719. The rebuttal agent independently re-verified every
premise against primary sources (code, 24 raw cell JSONs re-parsed, the
design-record passages for §15.27/§15.26.9 read verbatim, live arXiv
fetches for all four new citations) before adjudicating.

## CRITICAL status: A1 OPEN until FIX-1 lands and the touched sections
re-enter a scoped attack pass.

## Verdict table

| Attack | Severity (final) | Defense disposition | Final verdict | Fix |
|---|---|---|---|---|
| A1 h4 "held-out entities" mischaracterization | CRITICAL | CONCEDE+FIX | OPEN until FIX-1 lands + scoped re-attack | FIX-1 |
| A2 Table 1 d=96 unreplicated bound | SERIOUS | CONCEDE+FIX | fix prescribed | FIX-2 |
| A3 anchor-table bytes omitted | SERIOUS | CONCEDE+FIX | fix prescribed (with a unit correction BOTH prior stages missed: 27,392 B = 26.75 KiB, not 27.4) | FIX-3 |
| A4 exponent precision | MINOR (reduced) | PARTIAL | abstract-only fix | FIX-7 |
| A5 K69/s1730 exclusion | MINOR | CONCEDE+FIX | one-sentence disclosure | FIX-6 |
| A6 venue framing | SERIOUS | CONCEDE+FIX | one new intro paragraph | FIX-5 |
| A7 missing citations | SERIOUS | CONCEDE+FIX | 4 verified refs | FIX-4 |
| Bonus d_model fixed | MINOR | recommend | one clause | FIX-8 |

## Ordered fix list (near-verbatim language; writer applies in order)

1. **FIX-1 (CRITICAL, A1).** `02_testbed.tex` Readout paragraph: replace
   the "held-out entities and bindings" definition with: h4 is the
   rec@0.9 fraction at hop depth 4 drawn from the SAME 107-entity
   training pool; the held-out axis is HOP DEPTH (train 1-3, eval 4),
   not entity identity; reported recovery is held-out-depth
   compositional generalization over familiar entities, with a
   cross-reference to the Limitations disclosure. `07_limitations.tex`:
   new paragraph after Scope reporting the C17 disjoint-entity
   diagnostic collapse (1.0 at hop 1; ~0.0 at hops 2-3 in all 24
   sampled cells at both scales, while trained-pool readouts stay at
   ceiling), citing the pre-registered scope limit (design record: the
   anchor mechanism claims NO held-out-entity gains; held-out entities
   bypass the anchor blend by construction). New brief row C17 (verdict
   record: KEY_ANCHORING_DESIGN.md §3.3 reporting requirement at ~774 +
   §9.3 scope-limit passage at ~1820 + §M1 attack row at ~1550; raw
   artifacts: the SAME C1/C3 cell JSON sets, concat-md5s
   19ca8ed61c065710c627debb1ef3eb82 / 53bc17fb892489a6b820f654c21607e2;
   display: prose, limitations only).
   Adjudication note: only 02_testbed makes the FALSE claim; abstract/
   intro/03_cliff say "held-out recall/4-hop composition," which reads
   correctly once the definition is fixed — a scoped re-attack must
   specifically try to break this narrowing call.
2. **FIX-2 (SERIOUS, A2).** Table 1 d=96 row: dagger both bounds;
   caption footnote: the bound rests on the archived 3-seed K=90
   reading; a fresh-seed replication attempt did not confirm the
   ceiling and was itself inadmissible at the recalibrated gate.
   `07_limitations.tex` C14 paragraph: add the K=90 fresh h4=0.9725
   disclosure as a caveat-that-discounts (never a positive finding).
   Brief C14 row amended accordingly (numbers usable ONLY as caveats;
   no repo raw exists, disclosed). Fallback if a later round objects:
   re-derive the row from K=84 (>=84, >=2.33/KiB).
3. **FIX-3 (SERIOUS, A3).** `02_testbed.tex`: new "Anchor-table byte
   footprint" paragraph: 428*d bytes fp32 (26.75 KiB at d=64, 67% above
   the 16 KiB state; 53.5 KiB at d=128, 84% of the 64 KiB state);
   O(n*d) vs O(d^2); excluded from bindings/KiB because fixed by n and
   load-independent; a deployment budget must add it. New brief row C18
   (derived arithmetic, same convention as C15). Do NOT cite the
   50,259-row allocated-tensor implementation artifact.
4. **FIX-4 (SERIOUS, A7).** refs.bib + citation-verification.json + two
   06_related edits: add behrouz2025titans (2501.00663),
   munkhdalai2024infiniattention (2404.07143), yang2024gateddelta
   (2412.06464), arora2024based (2402.18668) — all live-verified;
   gated delta rules join the family sentence; Titans/Infini-attention
   as bounded test-time memory; Based joins Zoology.
5. **FIX-5 (SERIOUS, A6).** `01_introduction.tex`: new closing-area
   paragraph "A binding count as a reasoning-memory proxy" connecting
   K simultaneous bindings to intermediate sub-goal/variable state in
   multi-step reasoning chains; the frontier converts a byte budget
   into a bound on sub-goals held before recall degrades; the grammar
   models no specific reasoning task (disclosed).
6. **FIX-6 (MINOR, A5).** `05_frontier.tex` after the C12 sentence: the
   thirteenth examined cell (K=69/seed=1730, earlier wave, identical
   signature) was deliberately excluded per the declared 11-cell scope;
   including it would shift the K=69 mean 0.9592 -> 0.9423, changing no
   conclusion. New brief row C20 (verdict record:
   KEY_ANCHORING_SCALING_DRAFT.md ~5133; raws: K69 s1730-1732 cell
   JSONs in the scaling archive + the wide-grid s1733 file).
7. **FIX-7 (MINOR, A4).** Abstract: "grows as a two-point $d^{1.97}$
   exponent" (216 -> 217 words, in band).
8. **FIX-8 (MINOR, bonus).** `02_testbed.tex` architecture paragraph:
   d_model=256 held fixed across the whole sweep; d_state is the sole
   varied architectural axis (verified: argparse default, no chain
   script overrides).

## Residual risks recorded

- A1 residual: scope/ambition critique only (workshop-survivable).
  CROSS-PAPER FLAG for the coordinator: the same "held-out entities"
  framing risk likely recurs in sibling drafts reusing this testbed
  (neurips-ws-2026, workshop-2026, iclr-2027) — check before their own
  deadlines (out of this gauntlet's scope).
- A2 residual: reviewer may prefer the K=84-derived bound; named
  fallback.
- A6 residual: the new paragraph is a motivational analogy, not an
  empirical reasoning-trace result; existing Scope disclaimer covers.

## Re-run instruction

Scoped re-attack on 02_testbed + 07_limitations (A1), lighter pass on
05_frontier/06_related/01_introduction/00_abstract; specifically attempt
to break the "abstract/intro/03_cliff need no A1 edit" narrowing call.
Then gauntlet style stage (03) and format audit (05).
