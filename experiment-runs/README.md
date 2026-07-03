# experiment-runs — archive policy (2026-07-04)

Every experiment's exact scripts + result JSONs live here, size-capped:
files ≤25MB are tracked in git (crash-proof via GitHub); larger payloads
(Z-dumps, checkpoints, >25MB JSONs) live ONLY in the full archive at
/Volumes/1TB_SSD/learned-representations/experiment-runs/ (superset of
this directory). Write new archives to BOTH: small files here (commit
them), big files to the SSD path. If the SSD is unmounted, stop and say so.
