#!/usr/bin/env bash
# Stop hook: scan the assistant text for the most recent turn (from the last
# user entry to the end of the transcript) for [LEARN] blocks and insert them
# into workflow.db's learnings table.
#
# [LEARN] format (Claude is instructed to emit these in CLAUDE.md):
#   [LEARN] <category>: <rule>
#   Mistake: <what went wrong>
#   Correction: <what the right approach is>
#
# Fail-open: any error exits 0 so a broken hook never blocks the session.
set +e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB="$ROOT/memory/workflow.db"
LOG="$ROOT/memory/hook.log"
SQLITE=/usr/bin/sqlite3

bash "$ROOT/scripts/rotate_hook_log.sh" 2>/dev/null
echo "[$(date -Iseconds)] learn-capture: fired" >> "$LOG"

INPUT=$(cat)
TRANSCRIPT=$(echo "$INPUT" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("transcript_path",""))' 2>/dev/null)
[ -z "$TRANSCRIPT" ] && exit 0
[ ! -f "$TRANSCRIPT" ] && exit 0

PROJECT=$(basename "$(dirname "$ROOT")")

python3 - "$TRANSCRIPT" "$DB" "$PROJECT" "$SQLITE" "$LOG" <<'PYEOF'
import json, re, sys, subprocess, pathlib, datetime

transcript_path, db, project, sqlite_bin, log_path = sys.argv[1:6]

def log(msg):
    with open(log_path, 'a') as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] learn-capture: {msg}\n")

try:
    lines = pathlib.Path(transcript_path).read_text().splitlines()

    # Walk back from end. Collect ALL assistant text blocks until we hit a
    # user entry — that delimits "the current turn." Handles multi-block
    # assistant turns (thinking + text + tool_use as separate entries).
    last_turn_text = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        etype = entry.get('type')
        if etype == 'user':
            # Intra-turn tool_result messages are not the turn boundary —
            # only a user entry carrying actual text (the prompt) delimits turns.
            content = entry.get('message', {}).get('content', [])
            if isinstance(content, str):
                # String content = real user prompt
                break
            has_text = any(
                isinstance(b, dict) and b.get('type') == 'text'
                for b in (content or [])
            )
            if has_text:
                break
            # tool_result-only user entry: keep walking back through this turn
            continue
        if etype == 'assistant':
            content = entry.get('message', {}).get('content', [])
            # Collect text blocks only (ignore thinking, tool_use, tool_result)
            for block in (content or []):
                if isinstance(block, dict) and block.get('type') == 'text':
                    last_turn_text.insert(0, block.get('text', ''))
    text = '\n\n'.join(last_turn_text)

    if not text:
        sys.exit(0)

    # Match [LEARN] <category>: <rule>, optional \nMistake: ..., optional \nCorrection: ...
    # Anchor at end of body: next [LEARN] block, blank line, or end of input.
    pattern = re.compile(
        r'\[LEARN\]\s*([\w][\w\s\-/]*?)\s*:\s*(.+?)'
        r'(?:\n\s*Mistake:\s*(.+?))?'
        r'(?:\n\s*Correction:\s*(.+?))?'
        r'(?=\n\s*\[LEARN\]|\n\s*\n|\Z)',
        re.DOTALL | re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if not matches:
        sys.exit(0)

    def esc(s):
        return s.replace("'", "''") if s is not None else None

    inserted, duplicates = 0, 0
    for category, rule, mistake, correction in matches:
        category = category.strip()
        rule = rule.strip().split('\n')[0].strip()  # first line only
        mistake = (mistake or '').strip() or None
        correction = (correction or '').strip() or None

        if not category or not rule:
            continue

        e_proj = esc(project); e_cat = esc(category); e_rule = esc(rule)
        e_mist = f"'{esc(mistake)}'" if mistake else 'NULL'
        e_corr = f"'{esc(correction)}'" if correction else 'NULL'
        sql = (
            "PRAGMA trusted_schema=1;\n"
            "INSERT OR IGNORE INTO learnings "
            "(project, category, rule, mistake, correction, source) "
            f"VALUES ('{e_proj}', '{e_cat}', '{e_rule}', {e_mist}, {e_corr}, 'claude');\n"
            "SELECT changes();"
        )
        result = subprocess.run([sqlite_bin, db], input=sql, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"insert failed: {result.stderr.strip()}")
            continue
        changes = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else '0'
        if changes == '1':
            inserted += 1
            log(f"captured: [{category}] {rule[:80]}")
        else:
            duplicates += 1
            log(f"dup skip: [{category}] {rule[:60]}")

    if inserted:
        log(f"inserted {inserted} learning(s), {duplicates} duplicate(s) skipped")
        print(f"[learn-capture] {inserted} new learning(s) captured"
              + (f" ({duplicates} duplicate)" if duplicates else ""),
              file=sys.stderr)

except Exception as e:
    log(f"error: {type(e).__name__}: {e}")

sys.exit(0)
PYEOF
exit 0
