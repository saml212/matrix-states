#!/bin/bash
# Round 1: Matrix vs Vector head-to-head on 8x H100
# GPU 0-3: Matrix thinker
# GPU 4-7: Vector thinker
# Both run simultaneously, ~1 hour

set -e

echo "=== Round 1: Matrix vs Vector ==="
echo "Starting at $(date -u)"

# Matrix thinker on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone --nproc_per_node=4 \
    --master_port=29500 \
    /root/run_round1.py --model matrix \
    > /root/round1_matrix.log 2>&1 &
MATRIX_PID=$!

# Vector thinker on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone --nproc_per_node=4 \
    --master_port=29501 \
    /root/run_round1.py --model vector \
    > /root/round1_vector.log 2>&1 &
VECTOR_PID=$!

echo "Matrix PID: $MATRIX_PID"
echo "Vector PID: $VECTOR_PID"
echo "Logs: /root/round1_matrix.log, /root/round1_vector.log"

# Wait for both
wait $MATRIX_PID $VECTOR_PID

echo ""
echo "=== Round 1 Complete ==="
echo "Finished at $(date -u)"
echo ""
echo "=== MATRIX RESULTS ==="
cat /root/results/round1_matrix/SUMMARY.txt 2>/dev/null || echo "No summary"
echo ""
echo "=== VECTOR RESULTS ==="
cat /root/results/round1_vector/SUMMARY.txt 2>/dev/null || echo "No summary"
