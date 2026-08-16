#!/bin/bash
set -e

# Nano-Mythos architecture validation training runner
# Runs the ~50M param RDT model on DGX Spark 1 for ~24 hours

IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
WORKSPACE="/home/david-barnes/autoresearch-DGX-Spark"
DATA_CACHE="/home/david-barnes/.cache/autoresearch"
export PATH="$HOME/.local/bin:$PATH"

echo "=========================================="
echo "Nano-Mythos: 24h Architecture Validation"
echo "=========================================="
echo ""

# Run via SSH on Spark 1 (the training node)
ssh spark "
    cd $WORKSPACE
    docker run --rm --gpus all --ipc=host \\
        --ulimit memlock=-1 --ulimit stack=67108864 \\
        --shm-size 64gb --oom-score-adj 1000 \\
        -v \"$DATA_CACHE\":\"$DATA_CACHE\" \\
        -v \"$WORKSPACE\":\"$WORKSPACE\" \\
        -w \"$WORKSPACE\" \\
        -e HF_HUB_DISABLE_PROGRESS_BARS=1 \\
        -e PYTORCH_ALLOC_CONF=expandable_segments:True \\
        -e NCCL_P2P_DISABLE=1 \\
        -e TORCH_CUDA_ARCH_LIST=12.0 \\
        -e HOME=/home/david-barnes \\
        \"$IMAGE\" \\
        bash -c \"
            pip install -q rustbpe tiktoken pyarrow requests 2>&1 | tail -1
            python training/nano_mythos_train.py
        \"
" 2>&1