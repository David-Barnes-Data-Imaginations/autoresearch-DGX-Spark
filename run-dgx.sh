#!/bin/bash
# DGX Spark Docker launch script for autoresearch
# Requires: NVIDIA Container Toolkit installed
#
# Usage:
#   ./run-dgx.sh              # Interactive mode (default)
#   ./run-dgx.sh -d           # Detached mode (runs in background)
#   ./run-dgx.sh --test       # Quick test to verify setup

set -e

CONTAINER_NAME="autoresearch_training"
IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"

# Parse arguments
NON_INTERACTIVE=false
TEST_MODE=false
if [ "$1" = "-d" ] || [ "$1" = "--detach" ]; then
    NON_INTERACTIVE=true
elif [ "$1" = "--test" ]; then
    TEST_MODE=true
fi

echo "=== DGX Spark Autoresearch Launcher ==="
echo "Image: $IMAGE"
echo "Container: $CONTAINER_NAME"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA runtime is available
if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    echo "ERROR: NVIDIA Container Toolkit not found. Install it first:"
    echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

# Check if GPU is available
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: NVIDIA GPU not detected. Check your GPU drivers."
    exit 1
fi

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

if [ "$TEST_MODE" = true ]; then
    echo "=== Testing DGX Setup ==="
    echo ""
    
    # Test 1: Docker with NVIDIA runtime
    echo "✓ Docker is running"
    echo "✓ NVIDIA Container Toolkit is installed"
    
    # Test 2: GPU access
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo "✓ GPU detected: $GPU_INFO"
    
    # Test 3: Can pull and run container
    echo "Testing container startup..."
    docker run --rm --gpus all "$IMAGE" nvidia-smi > /dev/null 2>&1
    echo "✓ Container can access GPU"
    
    echo ""
    echo "=== All tests passed! ==="
    echo ""
    echo "You're ready to train! Run one of the following:"
    echo ""
    echo "  # Interactive mode (recommended for training)"
    echo "  ./run-dgx.sh"
    echo ""
    echo "  # Detached mode"
    echo "  ./run-dgx.sh -d"
    echo "  docker exec -it $CONTAINER_NAME bash"
    echo ""
    exit 0
fi

echo "Starting DGX Spark container with optimized settings..."
echo ""

# Build docker run command
DOCKER_CMD="docker run"

if [ "$NON_INTERACTIVE" = true ]; then
    DOCKER_CMD="$DOCKER_CMD -d"
else
    DOCKER_CMD="$DOCKER_CMD -it"
fi

$DOCKER_CMD \
    --name "$CONTAINER_NAME" \
    --ipc=host \
    --gpus all \
    --oom-score-adj 1000 \
    --shm-size 64gb \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd)":/workspace \
    -w /workspace \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    "$IMAGE" \
    /bin/bash

if [ "$NON_INTERACTIVE" = true ]; then
    echo ""
    echo "Container started in detached mode."
    echo ""
    echo "Next steps:"
    echo "  1. Enter the container:"
    echo "     docker exec -it $CONTAINER_NAME bash"
    echo ""
    echo "  2. Install dependencies:"
    echo "     uv sync"
    echo ""
    echo "  3. Download data (first time):"
    echo "     uv run prepare.py --num-shards 5"
    echo ""
    echo "  4. Run training:"
    echo "     uv run train.py"
    echo ""
    echo "  5. Monitor from another terminal:"
    echo "     docker stats $CONTAINER_NAME"
else
    echo ""
    echo "You are now inside the Docker container!"
    echo ""
    echo "Next steps:"
    echo "  1. Install dependencies:"
    echo "     uv sync"
    echo ""
    echo "  2. Download data (first time):"
    echo "     uv run prepare.py --num-shards 5"
    echo ""
    echo "  3. Run training:"
    echo "     uv run train.py"
    echo ""
    echo "To exit the container, type: exit"
    echo ""
fi
