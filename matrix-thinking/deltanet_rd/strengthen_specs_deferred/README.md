# strengthen_specs_deferred/ -- TRIM (round-2 audit, coordinator election, 2026-09-01)

`0621_h2h_strengthen_C2_lr3e-04_st60000_s0.json`,
`0622_h2h_strengthen_C2_lr3e-04_st60000_s1.json`,
`0623_h2h_strengthen_C2_lr3e-04_st60000_s2.json`
(C2 x lr=3e-4 x 60,000 steps, 3 seeds -- the three single most expensive
cells in the whole sweep, ~17.25 GPU-h at the fast-cluster anchor / ~18.05
GPU-h at the realized anchor) are DEFERRED, NOT staged into
`strengthen_specs/`, pending a pre-registered conditional re-add.

**Why:** with C2 x lr=1e-3 already covering C2's own best-reading LR at
both step counts (20k and 60k), the marginal information from ALSO
running C2's frozen-default LR (3e-4) at the longest, most expensive step
count is lower than for the other 24 staged cells, and dropping it moves
the design-time ledger from ~60.67/~63.4 GPU-h (30 cells, fast-cluster/
realized anchors) to ~43.43/~45.4 GPU-h (27 cells) -- narrowing the
pre-launch ceremony gap.

**Pre-registered conditional re-add (decided NOW, before any cell's
result exists -- see HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.46):** once the
27 staged cells' harvest reports `mean_acc_A` for both `C2 x lr=3e-4 x
20,000` and `C2 x lr=1e-3 x 20,000`, IF the former's mean exceeds the
latter's (the frozen-default LR outperforming C2's own currently-best LR
at the SAME, cheaper step count -- inverting the fix5-established
LR ranking specifically at this capacity), these 3 deferred cells run as
a SEPARATE follow-on, budgeted at the SAME ceiling this directory's own
files were priced at (<=17.25 GPU-h, fast-cluster anchor; re-price before
that follow-on launches, exactly as sec 1.46's RE-PRICE RULE requires for
the staged wave). If the condition does NOT fire, these files stay
deferred indefinitely -- never launched, never silently re-added.

**Outcomes A/B/C (sec 1.46's own decision rule) are UNCHANGED by this
TRIM** -- these 3 cells were never load-bearing for any of the three
outcomes (C2 x lr=1e-3 already covers both step counts; C2 x lr=3e-4 at
20,000 steps is unaffected, only its 60,000-step sibling is deferred).
