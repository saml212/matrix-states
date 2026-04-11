"""Convert text data to raw bytes for byte-level experiments."""
import torch, json, os, time

t0 = time.time()
out = "/toy_story_slam/data_bytes"
os.makedirs(out, exist_ok=True)

# Load the existing tokenized data and decode back to text, then to bytes
# OR simpler: download raw text and read as bytes directly
from datasets import load_dataset
os.environ["HF_HOME"] = "/toy_story_slam/.cache/huggingface"

print("Loading WikiText-103 raw text...", flush=True)
wiki = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir="/toy_story_slam/.cache/hf")

# Convert to raw bytes
train_bytes = []
for example in wiki["train"]:
    text = example["text"]
    if text.strip():
        train_bytes.extend(list(text.encode("utf-8")))

val_bytes = []
for example in wiki["validation"]:
    text = example["text"]
    if text.strip():
        val_bytes.extend(list(text.encode("utf-8")))

print(f"WikiText bytes: {len(train_bytes):,} train, {len(val_bytes):,} val ({time.time()-t0:.0f}s)", flush=True)

# Also get some code data for domain diversity
print("Loading code data...", flush=True)
try:
    code = load_dataset("bigcode/starcoderdata", "python", split="train", streaming=True, cache_dir="/toy_story_slam/.cache/hf")
    code_bytes = []
    n_docs = 0
    for example in code:
        code_bytes.extend(list(example["content"].encode("utf-8")))
        n_docs += 1
        if len(code_bytes) >= 100_000_000:  # 100MB of code
            break
    print(f"Code bytes: {len(code_bytes):,} from {n_docs} files ({time.time()-t0:.0f}s)", flush=True)
except Exception as e:
    print(f"Code download failed: {e}, continuing without", flush=True)
    code_bytes = []

# Combine
all_train = train_bytes + code_bytes
# Use wiki val only (code has no standard val split)
all_val = val_bytes

print(f"Combined: {len(all_train):,} train bytes, {len(all_val):,} val bytes", flush=True)

train_t = torch.tensor(all_train, dtype=torch.long)
val_t = torch.tensor(all_val, dtype=torch.long)

torch.save(train_t, os.path.join(out, "train.pt"))
torch.save(val_t, os.path.join(out, "val.pt"))
json.dump({"vocab_size": 256, "tokenizer": "bytes",
           "train_tokens": len(all_train), "val_tokens": len(all_val),
           "sources": [f"WikiText-103 bytes ({len(train_bytes):,})", f"Python code bytes ({len(code_bytes):,})"]},
          open(os.path.join(out, "meta.json"), "w"), indent=2)
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
