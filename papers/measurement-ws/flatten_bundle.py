#!/usr/bin/env python3
"""Flatten main.tex for the submission bundle.

Inlines every \\input{sections/...} and strips LaTeX comments
(unescaped % to end of line) so source-only annotations — the evidence
markers and the repo-token comment in the main.tex header — never reach
a source upload (format-audit round-1 finding M1). Escaped \\% survives.
"""
import re

src = open("main.tex").read()
flat = re.sub(r"\\input\{(sections/[^}]+)\}",
              lambda m: open(m.group(1) + ".tex").read(), src)
stripped = "\n".join(
    re.sub(r"(?<!\\)%.*", "", line).rstrip() for line in flat.splitlines()
)
# Collapse runs of blank lines left by removed comment blocks (3+ -> 2)
# so paragraph breaks (blank lines) are preserved exactly.
stripped = re.sub(r"\n{3,}", "\n\n", stripped)
open("bundle/measurement-ws-submission.tex", "w").write(stripped + "\n")
print("flattened: bundle/measurement-ws-submission.tex")
