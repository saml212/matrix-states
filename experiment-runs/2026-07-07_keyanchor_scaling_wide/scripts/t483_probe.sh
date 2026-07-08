#!/bin/bash
# Thin shell wrapper (house convention: every GPU-touching launch in this
# directory goes through a .sh entrypoint, never a bare `python foo.py` in
# the invoking shell command) around smoke_dstate_kernel_t483_probe.py --
# the K=69/d=96 contingency-seed RUN task's own T_bind=483 kernel-safety
# pre-flight gate (sub-minute, no measurable GPU-h, GPU 3 ONLY).
set -euo pipefail
cd /home/nvidia/chapter2/deltanet_rd
CUDA_VISIBLE_DEVICES=3 /home/nvidia/tdenv/bin/python smoke_dstate_kernel_t483_probe.py
