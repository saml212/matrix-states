"""K-wall wall-characterization orchestrator library.

Implements the §8 build charter of `NCR_KWALL_CHARACTERIZATION_DESIGN.md`
(STATUS: RELEASED, commit 1c99cc5, §A11-ADJUDICATION) — the ORCHESTRATOR
CONTRACT, the reconstruction/recovery procedure, the trigger + band
classification rules, and `validity_check`. Every module below cites the
exact design-doc section it implements in its own docstring so a reader
(or the build audit) can trace each line back to its binding source.
"""
