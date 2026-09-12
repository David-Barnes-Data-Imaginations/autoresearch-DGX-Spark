#!/bin/bash
set -e

IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
WORKSPACE="/home/david-barnes/autoresearch-DGX-Spark"
DATA_CACHE="/home/david-barnes/.cache/autoresearch"
export PATH="$HOME/.local/bin:$PATH"

# Single experiment runner
# Args: $1 = experiment name, $2 = sed expression (applied to train_exp.py)
# Patches a COPY of train.py, runs it, leaves original untouched

EXPERIMENT_NAME="$1"
SED_EXPR="$2"

echo "========================================"
echo "EXPERIMENT: $EXPERIMENT_NAME"
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
        cp train.py train_exp.py
        sed -i '$SED_EXPR' train_exp.py
        echo '--- Config ---'
        grep -E '^EMBEDDING_LR|^SCALAR_LR|^MATRIX_LR|^UNEMBEDDING_LR|^WARMUP_RATIO|^softcap|^WEIGHT_DECAY' train_exp.py
        python train_exp.py 2>&1
        rm -f train_exp.py
    " 2>&1

echo ""
echo "EXPERIMENT $EXPERIMENT_NAME COMPLETE"
