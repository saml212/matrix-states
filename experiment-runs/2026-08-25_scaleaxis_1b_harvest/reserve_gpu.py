#!/usr/bin/env python3
"""Hold a small resident allocation on the CURRENT CUDA device so the queue
worker's own B6 free-GPU gate excludes it.

The gate (queue_worker.sh:107-112) is:
    napps = # nvidia-smi compute-apps on the GPU
    mem   = memory.used MiB
    skip the GPU iff  napps > 0  OR  mem >= 2048
This process trips BOTH legs: it is itself a compute app, and it holds ~3 GiB.
Belt and braces, because a memory-only reservation would be defeated by a
driver that reports lazily and a compute-app-only one by a context teardown.

Releases when the stop-file appears. Coordinator election (b), 2026-08-25.
"""
import os
import sys
import time

import torch

GIB = 1024 ** 3


def main():
    stop = sys.argv[1]
    size_gib = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    if not torch.cuda.is_available():
        print("RESERVE_FAIL: no CUDA device visible", flush=True)
        return 2
    n = int(size_gib * GIB // 2)          # fp16 -> 2 bytes/elem
    buf = torch.zeros(n, dtype=torch.float16, device="cuda")
    buf.fill_(1.0)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() / GIB
    print(f"RESERVED pid={os.getpid()} held={held:.2f} GiB stop={stop}", flush=True)
    while not os.path.exists(stop):
        time.sleep(2)
    del buf
    torch.cuda.empty_cache()
    print("RELEASED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
