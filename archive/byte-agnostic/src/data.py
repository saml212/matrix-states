"""
Multi-domain byte data pipeline.

Loads text (FineWeb), images (CIFAR-10), and audio (LibriSpeech)
as raw byte streams. No tokenizers, no domain indicators.

For Phase 1: text only.
For Phase 2+: interleaved multi-domain.
"""

import os
import struct
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class ByteDataset(Dataset):
    """Dataset that serves fixed-length byte sequences from a memory-mapped file."""

    def __init__(self, data_path: str, seq_len: int = 1024):
        self.seq_len = seq_len
        self.data_path = data_path

        # Memory-map the file for efficient random access
        self.data = np.memmap(data_path, dtype=np.uint8, mode='r')
        self.n_sequences = max(1, len(self.data) // seq_len - 1)  # -1 for target shift

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for next-byte target
        chunk = self.data[start:end].copy()

        # Input: bytes [0..seq_len-1], Target: bytes [1..seq_len]
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y


def prepare_text_data(output_path: str, size_mb: int = 10, cache_dir: str = None):
    """Download and prepare text data from FineWeb as raw UTF-8 bytes."""
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > size_mb * 500_000:
        print(f"Text data already exists at {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return str(output_path)

    print(f"Downloading text data (~{size_mb}MB)...")
    from datasets import load_dataset

    target_bytes = size_mb * 1_000_000
    collected = bytearray()

    # Use wikitext — small, fast to download, good quality text
    ds = load_dataset(
        "Salesforce/wikitext",
        name="wikitext-103-raw-v1",
        split="train",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    for example in ds:
        text = example["text"]
        if text.strip():  # skip empty lines
            collected.extend(text.encode("utf-8"))
        if len(collected) >= target_bytes:
            break

    collected = collected[:target_bytes]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(bytes(collected))

    print(f"Saved {len(collected) / 1e6:.1f} MB of text to {output_path}")
    return str(output_path)


def prepare_image_data(output_path: str, cache_dir: str = None):
    """Prepare CIFAR-10 as raw pixel bytes (no headers, no labels)."""
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 100_000:
        print(f"Image data already exists at {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return str(output_path)

    print("Preparing CIFAR-10 as raw pixel bytes...")
    try:
        import torchvision
        import torchvision.transforms as T
    except ImportError:
        raise ImportError("Install torchvision: pip install torchvision")

    ds = torchvision.datasets.CIFAR10(
        root=cache_dir or "/tmp/cifar10",
        train=True,
        download=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    collected = bytearray()

    for img, _label in ds:
        # Convert PIL image to raw RGB bytes (32x32x3 = 3072 bytes per image)
        pixels = np.array(img, dtype=np.uint8)  # (32, 32, 3)
        collected.extend(pixels.tobytes())

    with open(output_path, "wb") as f:
        f.write(bytes(collected))

    print(f"Saved {len(collected) / 1e6:.1f} MB of image data to {output_path}")
    return str(output_path)


def prepare_audio_data(output_path: str, size_mb: int = 10, cache_dir: str = None):
    """Prepare LibriSpeech as raw 16-bit PCM bytes."""
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > size_mb * 500_000:
        print(f"Audio data already exists at {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
        return str(output_path)

    print(f"Preparing LibriSpeech as raw PCM bytes (~{size_mb}MB)...")
    try:
        import torchaudio
    except ImportError:
        raise ImportError("Install torchaudio: pip install torchaudio")

    target_bytes = size_mb * 1_000_000
    collected = bytearray()

    ds = torchaudio.datasets.LIBRISPEECH(
        root=cache_dir or "/tmp/librispeech",
        url="dev-clean",
        download=True,
    )

    for waveform, sample_rate, *_ in ds:
        # Convert to 16-bit PCM bytes
        # waveform is (channels, samples) float tensor
        audio = waveform[0]  # mono
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        # Convert to 16-bit int
        audio_int16 = (audio * 32767).clamp(-32768, 32767).to(torch.int16)
        collected.extend(audio_int16.numpy().tobytes())
        if len(collected) >= target_bytes:
            break

    collected = collected[:target_bytes]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(bytes(collected))

    print(f"Saved {len(collected) / 1e6:.1f} MB of audio data to {output_path}")
    return str(output_path)


class MultiDomainDataset(Dataset):
    """Interleaves byte sequences from multiple domains without domain labels."""

    def __init__(self, data_paths: list, seq_len: int = 1024):
        self.datasets = [ByteDataset(p, seq_len) for p in data_paths]
        self.total_len = sum(len(d) for d in self.datasets)
        # Build index mapping: flat idx -> (dataset_idx, local_idx)
        self.index_map = []
        for di, ds in enumerate(self.datasets):
            for li in range(len(ds)):
                self.index_map.append((di, li))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        di, li = self.index_map[idx]
        return self.datasets[di][li]


def get_dataloader(data_path: str, seq_len: int = 1024, batch_size: int = 16,
                   num_workers: int = 0, shuffle: bool = True) -> DataLoader:
    """Create a DataLoader for byte sequences."""
    dataset = ByteDataset(data_path, seq_len)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
