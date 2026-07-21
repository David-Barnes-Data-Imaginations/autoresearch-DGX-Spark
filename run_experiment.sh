#!/bin/bash
set -e
# Run autoresearch training inside Docker container on DGX Spark
# Usage: ./run_experiment.sh [num_shards]
# Default: 5 shards for quick testing

NUM_SHARDS=${1:-5}
IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
WORKSPACE="/home/david-barnes/autoresearch-DGX-Spark"
DATA_CACHE="/home/david-barnes/.cache/autoresearch"

echo "=== Autoresearch Training Run ==="
echo "Shards: $NUM_SHARDS"
echo "Time: $(date)"
echo ""

export PATH="$HOME/.local/bin:$PATH"

# Ensure cache dir exists
mkdir -p "$DATA_CACHE"
chmod -R 777 "$DATA_CACHE" 2>/dev/null || true

DOCKER_RUN="docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size 64gb --oom-score-adj 1000 \
    -v \"$DATA_CACHE\":\"$DATA_CACHE\" \
    -v \"$WORKSPACE\":\"$WORKSPACE\" \
    -w \"$WORKSPACE\" \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0"

# Prepare data if needed
if [ ! -f "$DATA_CACHE/tokenizer/tokenizer.pkl" ]; then
    echo "Preparing data (tokenizer + $NUM_SHARDS shards)..."
    $DOCKER_RUN "$IMAGE" \
        bash -c "pip install -q --root-user-action=ignore kernels matplotlib numpy pandas pyarrow requests rustbpe tiktoken && CACHE_DIR=$DATA_CACHE python prepare.py --num-shards $NUM_SHARDS"
    echo "Data preparation complete."
    echo ""
fi

# Run training
echo "Starting training run..."
$DOCKER_RUN "$IMAGE" \
    bash -c "pip install -q --root-user-action=ignore kernels matplotlib numpy pandas pyarrow requests rustbpe tiktoken 2>/dev/null; python train.py"
