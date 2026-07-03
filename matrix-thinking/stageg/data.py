"""Stage G Regime-2 byte data pipeline — see STAGE_G_DESIGN.md section 5.4
("byte-vocab (256, not 50257)") and section 12 ("data: byte data at
/Volumes/1TB_SSD/learned-representations/data/text.bin locally").

Reuses `matrix-thinking/src/data.py::ByteDataset`'s memory-mapped-file
pattern (raw bytes, no tokenizer, no pre-tokenized .pt cache needed --
simpler than the Regime-1 scripts' TokenDataset, which expects a
pre-tokenized train.pt/val.pt pair) rather than importing it directly:
that module lives under `matrix-thinking/src/`, a different sys.path root,
and Stage G's build requirement is a self-contained `stageg/` directory
(mirrors chapter2/run_stage0.py's "CLONE, not importer" choice for the
same reason, documented there).

Train/val split: a contiguous 95/5 split by BYTE OFFSET (not by
sequence), matching the standard held-out-tail convention and avoiding
any near-duplicate leakage across the split boundary that a random
per-sequence split could introduce for adjacent, overlapping windows.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_LOCAL_PATH = "/Volumes/1TB_SSD/learned-representations/data/text.bin"
VOCAB_SIZE = 256  # raw bytes


class ByteDataset(Dataset):
    """Serves fixed-length byte sequences from a memory-mapped slice of a
    raw byte file. `byte_range=(start,end)` restricts this dataset to a
    contiguous sub-range of the underlying file (used to carve train/val
    splits from the same file without duplicating it on disk)."""

    def __init__(self, data_path: str, seq_len: int = 512, byte_range: tuple[int, int] | None = None):
        self.seq_len = seq_len
        self.data_path = data_path
        full = np.memmap(data_path, dtype=np.uint8, mode='r')
        if byte_range is not None:
            start, end = byte_range
            self.data = full[start:end]
        else:
            self.data = full
        # Hard-error on an undersized window (audit MINOR-4): the previous
        # max(1, ...) clamp would silently claim one sequence from a window
        # too small to serve a full (seq_len+1)-byte sample, and __getitem__
        # would then return a short/empty tensor instead of failing loudly
        # at construction.
        self.n_sequences = len(self.data) // seq_len - 1
        if self.n_sequences < 1:
            raise ValueError(
                f"{data_path}: window of {len(self.data)} bytes (byte_range={byte_range}) "
                f"yields no full (seq_len+1)={seq_len + 1}-byte sample; "
                f"need >= {2 * seq_len} bytes")

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end].copy()
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y


def train_val_datasets(data_path: str, seq_len: int = 512, val_frac: float = 0.05):
    """Contiguous split by byte offset: first (1-val_frac) of the file is
    train, the tail val_frac is held out."""
    n_bytes = os.path.getsize(data_path)
    split = int(n_bytes * (1 - val_frac))
    train_ds = ByteDataset(data_path, seq_len, byte_range=(0, split))
    val_ds = ByteDataset(data_path, seq_len, byte_range=(split, n_bytes))
    return train_ds, val_ds


def resolve_data_path(cli_path: str | None = None) -> str:
    """Resolution order (design section 12(vi)'s spirit -- explicit,
    logged, never silent): --data-path CLI override > $STAGEG_DATA_PATH env
    > the local SSD default > a same-directory `text.bin` (the box-side
    staging convention, see wave_neg1.py's data-availability check)."""
    candidates = [
        cli_path,
        os.environ.get("STAGEG_DATA_PATH"),
        DEFAULT_LOCAL_PATH,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "text.bin"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No byte data file found. Tried: {[c for c in candidates if c]}. "
        f"Run wave_neg1.py's data-availability check for shipping instructions.")
