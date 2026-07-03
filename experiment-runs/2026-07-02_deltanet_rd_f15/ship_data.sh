#!/usr/bin/env bash
# ship_data.sh -- DELTANET_REALDATA_DESIGN.md section 3/section 7's Prereq
# row (blocking, ~0 GPU-h): scp the OpenR1-Math (Wave 1 does not need this,
# but the Prereq row ships it alongside WikiText-103 since both are needed
# together for the eventual Wave 2 gate) + WikiText-103 GPT-2-tokenized
# sets from the local SSD to the current Brev box's /data volume, then
# verify md5s match exactly -- "the manifest does not assume the data
# landed correctly" (section 8).
#
# This is the EXACT script that was run 2026-07-02 (house rule: "save the
# exact script that was run alongside experiment results for
# reproducibility"). Local SSD paths confirmed present in
# DELTANET_REALDATA_DESIGN.md section 3's table; re-run is idempotent (scp
# overwrites, md5 check is unconditional either way).
#
# Usage: ./ship_data.sh [remote_host]   (default: youthful-indigo-turkey)
set -euo pipefail

REMOTE="${1:-youthful-indigo-turkey}"
SSD=/Volumes/1TB_SSD/learned-representations/data
REMOTE_DATA=/data/deltanet_rd_data

echo "=== shipping reasoning (OpenR1-Math) ==="
ssh "$REMOTE" "mkdir -p $REMOTE_DATA/reasoning"
scp "$SSD/reasoning/train.pt" "$SSD/reasoning/val.pt" "$SSD/reasoning/meta.json" \
    "$REMOTE:$REMOTE_DATA/reasoning/"

echo "=== shipping wikitext103_tokenized ==="
ssh "$REMOTE" "mkdir -p $REMOTE_DATA/wikitext103_tokenized"
scp "$SSD/wikitext103_tokenized/train.pt" "$SSD/wikitext103_tokenized/val.pt" \
    "$SSD/wikitext103_tokenized/test.pt" "$SSD/wikitext103_tokenized/meta.json" \
    "$REMOTE:$REMOTE_DATA/wikitext103_tokenized/"

echo "=== verifying md5s (local vs remote) ==="
for f in reasoning/train.pt reasoning/val.pt reasoning/meta.json \
         wikitext103_tokenized/train.pt wikitext103_tokenized/val.pt \
         wikitext103_tokenized/test.pt wikitext103_tokenized/meta.json; do
  local_md5=$(md5 -q "$SSD/$f")
  remote_md5=$(ssh "$REMOTE" "md5sum $REMOTE_DATA/$f | cut -d' ' -f1")
  if [ "$local_md5" != "$remote_md5" ]; then
    echo "MISMATCH: $f  local=$local_md5  remote=$remote_md5" >&2
    exit 1
  fi
  echo "OK  $f  $local_md5"
done

echo "=== verifying meta.json fields against DELTANET_REALDATA_DESIGN.md section 3's table ==="
ssh "$REMOTE" "python3 -c \"
import json
r = json.load(open('$REMOTE_DATA/reasoning/meta.json'))
w = json.load(open('$REMOTE_DATA/wikitext103_tokenized/meta.json'))
assert r['vocab_size'] == 50257 and r['tokenizer'] == 'gpt2' and r['train_tokens'] == 43725587, r
assert w['vocab_size'] == 50257 and w['tokenizer'] == 'gpt2' and w['train_tokens'] == 117920140, w
print('reasoning meta.json:', r)
print('wikitext103 meta.json:', w)
print('ALL FIELD CHECKS PASSED')
\""

echo "=== DONE: data shipped and verified at $REMOTE:$REMOTE_DATA ==="
