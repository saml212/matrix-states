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
out_lines = []
for line in flat.splitlines():
    stripped = re.sub(r"(?<!\\)%.*", "", line).rstrip()
    if stripped == "" and line.strip() != "":
        # Comment-only line: drop it entirely. Leaving a blank line here
        # would insert a paragraph break mid-paragraph and reflow the
        # body (LaTeX treats a blank line as \par; a comment line is
        # layout-neutral, so its removal must be too).
        continue
    out_lines.append(stripped)
open("bundle/measurement-ws-submission.tex", "w").write(
    "\n".join(out_lines) + "\n")
print("flattened: bundle/measurement-ws-submission.tex")
