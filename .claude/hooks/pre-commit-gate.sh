#!/usr/bin/env bash
# PreToolUse(Bash) hook: intercept `git commit` and require a valid /clean sentinel.
# Bypass: prefix command with CLEAN_BYPASS=1 (or set env var).
set +e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "$ROOT/scripts/rotate_hook_log.sh" 2>/dev/null

INPUT=$(cat)
CMD=$(echo "$INPUT" | python3 -c '
import sys,json
d=json.load(sys.stdin)
print(d.get("tool_input",{}).get("command",""))
' 2>/dev/null)

# Fast path: not a git commit
echo "$CMD" | grep -qE '(^|[;&|[:space:]])git[[:space:]]+commit([[:space:]]|$)' || exit 0

echo "[$(date -Iseconds)] pre-commit-gate: fired" >> "$ROOT/memory/hook.log"

# Bypass via command-inline env var OR shell env
if echo "$CMD" | grep -qE '(^|[;&|[:space:]])CLEAN_BYPASS=1\b' || [ "$CLEAN_BYPASS" = "1" ]; then
  echo "[pre-commit-gate] CLEAN_BYPASS=1 — allowing" >&2
  echo "[$(date -Iseconds)] pre-commit-gate: bypass" >> "$ROOT/memory/hook.log"
  exit 0
fi

HASH=$(bash "$ROOT/scripts/compute_clean_hash.sh" 2>/dev/null)
if [ -z "$HASH" ] || [ "$HASH" = "no-changes" ]; then
  exit 0
fi

SENTINEL="$ROOT/.state/clean-ok-$HASH"
if [ -f "$SENTINEL" ]; then
  exit 0
fi

{
  echo "[pre-commit-gate] BLOCKED: no valid /clean sentinel for the current staged set."
  echo "[pre-commit-gate] Expected: .claude/.state/clean-ok-$HASH"
  echo "[pre-commit-gate] Run: python3 .claude/skills/clean/audit.py --scope staged"
  echo "[pre-commit-gate] Or bypass: prefix commit with  CLEAN_BYPASS=1"
} >&2
exit 2
