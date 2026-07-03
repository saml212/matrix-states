# Control A v2 — Fix Receipt

Per-item status of every audit P0/P1/P2 and every attack F/M/minor. See
`CONTROL_A_AUDIT_REPORT.md`, `CONTROL_A_ATTACK_REPORT.md`, and
`CONTROL_A_IMPLEMENTATION_NOTES.md` for context.

## Audit fixes

| Source | ID   | Description (short)                              | Status     | File:Line                         |
|--------|------|--------------------------------------------------|------------|-----------------------------------|
| audit  | P0-1 | loader `any()` → majority check + format flag    | FIXED      | run_control_a.py:282-308          |
| audit  | P0-2 | critical-keys + param-count asserts, hard fail   | FIXED      | run_control_a.py:310-355          |
| audit  | P0-3 | 10-example unablated smoke, assert >=5 correct   | FIXED      | run_control_a.py:985-1008         |
| audit  | P1-1 | NaN Spearman → ambiguous, not flat               | FIXED      | run_control_a.py:932-944          |
| audit  | P1-2 | KV-cache leakage: redesigned, no KV cache        | FIXED      | run_control_a.py:520-555          |
| audit  | P1-3 | n<3 data points → ambiguous                      | FIXED      | run_control_a.py:919-922          |
| audit  | P1-4 | log first-token text (+id, correctness)          | FIXED      | run_control_a.py:605-635          |
| audit  | P1-5 | drop `p.requires_grad` filter on n_params        | FIXED      | run_control_a.py:758              |
| audit  | P1-6 | add schema_version + parent_schema to JSON       | FIXED      | run_control_a.py:1263-1265        |
| audit  | P1-7 | cap per_example rows in JSON                     | FIXED      | run_control_a.py:651-653, 1117    |
| audit  | P2-1 | smoke test writes _script_used.py                | WONTFIX    | acceptable per auditor; unchanged |
| audit  | P2-2 | add ETA to progress log                          | FIXED      | run_control_a.py:640              |
| audit  | P2-3 | weights_only=False trust note                    | WONTFIX    | accepted by auditor; unchanged    |
| audit  | P2-4 | scipy fallback NaN p-value                       | WONTFIX    | low risk; scipy present on H100   |
| audit  | P2-5 | smoke test covers all three variants             | FIXED      | run_control_a.py:1157-1166 (flag) |
| audit  | P2-6 | emoji-free, no over-commenting                   | PRESERVED  | v2 maintains this                 |
| audit  | P2-7 | deterministic dataset ordering                   | PRESERVED  | ProsQADataset unchanged           |

## Attack fixes

| Source | ID | Description (short)                              | Status | File:Line                          |
|--------|----|--------------------------------------------------|--------|------------------------------------|
| attack | F1 | propagating intervention at 6 analog positions   | FIXED  | run_control_a.py:176-232, 408-568  |
| attack | F2 | intervene before `:` via mid-layer hook          | FIXED  | run_control_a.py:377-405, 1101-1105|
| attack | M1 | per-example first-token accuracy column          | FIXED  | run_control_a.py:610-635, 678-698  |
| attack | M2 | --max-new-tokens default 24                      | FIXED  | run_control_a.py:1112              |
| attack | M3 | binomial-aware AMBIGUOUS decision rule           | FIXED  | run_control_a.py:895-944           |
| attack | M4 | k=None un-ablated baseline in the sweep          | FIXED  | run_control_a.py:783-810           |
| attack | M5 | pooled stats over complete-k-across-all-seeds    | FIXED  | run_control_a.py:1200-1226         |
| attack | m1 | startup env / TF32 / GPU log for reproducibility | FIXED  | run_control_a.py:1144-1152         |
| attack | m2 | tokenization drift visible in smoke              | FIXED  | run_control_a.py:377-405 (colon)   |
| attack | m3 | log full predicted text, not truncated           | FIXED  | run_control_a.py:625-627           |
| attack | m4 | lowercase `decision:` everywhere                 | FIXED  | run_control_a.py:1305, 1313-1315   |

## Additional controls recommended by the attack report

| Source | Description                                 | Status | File:Line                              |
|--------|---------------------------------------------|--------|----------------------------------------|
| attack | randomize-h[:256] sensitivity floor         | ADDED  | run_control_a.py:146-167, 838-866      |
| attack | scope-of-claim documented                   | ADDED  | CONTROL_A_IMPLEMENTATION_NOTES.md §5   |
| attack | separate FLAT/BENDING classifier on first-tok | ADDED | run_control_a.py:1246-1252             |

All audit and attack items are accounted for: 10 FIXED + 2 PRESERVED of
the P0+P1+P2 set, 4 WONTFIX of the P2 set (documented, acceptable), 2
FIXED fatal, 5 FIXED major, 4 FIXED minor. No items silently skipped.
