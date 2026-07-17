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

## v3 (2026-07-17, §B6 composed guard — the RELAUNCHED script)

`ncr_ortho_fallback_stage0_v3.py` (this directory)
md5: `c7c2f7c36e0ab33c1efaeed04fbcf1bb`

Two semantic changes vs v2, exactly §B6 Ruling 3's spec: composed denominator
guard `S.clamp_min((self._eps_rel * sigma_max * 1e-6).clamp_min(torch.finfo(S.dtype).tiny))`
(caps scale at 1e6 — kills F-B — with a finfo.tiny backstop for Z==0) +
RUNNER_TAG bump to `_v3`. Relaunched 2026-07-17T08:02:48Z under §B6's
pre-authorized auto-clear (all 5 conditions verified — design doc §B7).
Driver: `orchestration/run_stage0_v3.sh` (md5 `96cc2f8103b9c6d4165d3a5d1af4eea3`).
Auto-clear verification: `verify_v3_autoclear.py` (md5 `2dc6b69cee03b10d800f77d6addf3d2c`).
