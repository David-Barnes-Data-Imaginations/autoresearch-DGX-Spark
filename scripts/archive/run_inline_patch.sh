#!/bin/bash
set -e

IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
WORKSPACE="/home/david-barnes/autoresearch-DGX-Spark"
DATA_CACHE="/home/david-barnes/.cache/autoresearch"
export PATH="$HOME/.local/bin:$PATH"

# Patch train.py inside the container: SCALAR_LR=0.525 -> 0.50
# Use sed -f with a file to handle the # character in comments
docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size 64gb --oom-score-adj 1000 \
    -v "$DATA_CACHE":"$DATA_CACHE" \
    -v "$WORKSPACE":"$WORKSPACE" \
    -w "$WORKSPACE" \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e HOME=/home/david-barnes \
    "$IMAGE" \
    bash -c "pip install -q rustbpe tiktoken pyarrow requests 2>&1 | tail -1 && \
        sed -i 's/SCALAR_LR = 0.525  # learning rate for per-layer scalars/SCALAR_LR = 0.50  # testing lower/' train.py && \
        echo 'Patched: SCALAR_LR=0.525 -> SCALAR_LR=0.50' && \
        grep -n 'SCALAR_LR' train.py && \
        python train.py 2>&1; \
        EXIT_CODE=\$?; \
        sed -i 's/SCALAR_LR = 0.50  # testing lower/SCALAR_LR = 0.525  # learning rate for per-layer scalars/' train.py && \
        echo 'Restored: SCALAR_LR=0.50 -> SCALAR_LR=0.525' && \
        exit \$EXIT_CODE"
