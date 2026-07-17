# Stage-0 build record (2026-07-17)

Archived script: `ncr_ortho_fallback_stage0.py` (this directory)
md5: `70dd7923027f4dcf9f0f1e964fc0930c`

Base script (unmodified, for diffing): `experiment-runs/2026-07-16_ncr_ortho_write/ncr_ortho_write.py`
md5: `83b5d7bd273e9e83698fed27a9f2ef45`

Full record: see `matrix-thinking/NCR_ORTHO_FALLBACK_DESIGN.md` §B1 STAGE-0
BUILD RECORD (2026-07-17), appended same commit as this archive.

## v2 (2026-07-17, post-§B4 audit D1 fix)

`ncr_ortho_fallback_stage0_v2.py` (this directory)
md5: `ce1448ab3d47536ebf3e82b146e33722`

One-line semantic change vs v1 (plus its comment block): the SVD-floor
denominator guard `S.clamp_min(NS_EPS)` -> `S.clamp_min(torch.finfo(S.dtype).tiny)`
(§B4 FATAL D1 — NS_EPS=1e-7 silently defeated the exact floor below 1e-7,
the exact §10 trap regime). v1 is the AS-RUN script for the VOID-CONTENTION
first attempt (§B3) and is preserved unmodified per the reproducibility rule.
Full record: design doc §B5.
