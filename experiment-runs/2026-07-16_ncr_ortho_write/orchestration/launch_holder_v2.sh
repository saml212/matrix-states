#!/usr/bin/env bash
# Per-GPU NCR holder launcher. Establishes a resident CUDA context on the target
# GPU via hold_gpu.py so the queue_worker free-GPU gate (napps==0 & mem<2GiB)
# never claims a training job onto a GPU an NCR cell owns. Releases (exits) when
# its per-GPU RELEASE sentinel appears OR a hard self-timeout fires (backstop
# so a leaked holder can never strand a GPU indefinitely). Wrapped so the
# python invocation stays out of the launching agent's command string.
set -u
GPU="${1:?usage: launch_holder_v2.sh GPU_ID}"
export CUDA_VISIBLE_DEVICES="$GPU"
export NCR_HOLD_RELEASE="/home/nvidia/queue/NCR_HOLD_RELEASE_g${GPU}"
export NCR_HOLD_MAX_S="194400"   # 54h self-timeout backstop (> worst-case 8x6.5h run)
exec /home/nvidia/tdenv/bin/python3 /home/nvidia/ncr/hold_gpu.py
