#!/bin/bash
set -e

IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
WORKSPACE="/home/david-barnes/autoresearch-DGX-Spark"
DATA_CACHE="/home/david-barnes/.cache/autoresearch"
export PATH="$HOME/.local/bin:$PATH"

# Batch experiment runner for DGX Spark
# Patches train.py inside Docker (on the mounted volume), runs training, restores

# Args: $1 = experiment name, $2 = sed patch command
run_exp() {
    local name="$1"
    local patch="$2"

    echo "========================================"
    echo "EXPERIMENT: $name"
    echo "========================================"

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
        bash -c "
            pip install -q rustbpe tiktoken pyarrow requests 2>&1 | tail -1
            cp train.py train.py.bak
            $patch
            echo '--- Config ---'
            grep -E '^EMBEDDING_LR|^SCALAR_LR|^MATRIX_LR|^UNEMBEDDING_LR|^WARMUP_RATIO|^softcap|^WEIGHT_DECAY' train.py
            python train.py 2>&1
            mv train.py.bak train.py
        " 2>&1

    echo ""
    echo "EXPERIMENT $name COMPLETE"
    echo ""
}

# Experiment 1: softcap=10 (already in code, but verify)
run_exp "softcap10" "sed -i 's/softcap = 15/softcap = 10/' train.py"

# Experiment 2: MATRIX_LR=0.038 (slightly lower from 0.04)
run_exp "matrix_lr_038" "sed -i 's/MATRIX_LR = 0.04/MATRIX_LR = 0.038/' train.py"

# Experiment 3: WARMUP_RATIO=0.01 (tiny warmup)
run_exp "warmup_001" "sed -i 's/WARMUP_RATIO = 0.0/WARMUP_RATIO = 0.01/' train.py"
