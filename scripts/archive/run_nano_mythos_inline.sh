#!/bin/bash
set -e

# Nano-Mythos: 24h Architecture Validation
# This script runs INSIDE the Docker container on Spark 1

IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
WORKSPACE="/home/david-barnes/autoresearch-DGX-Spark"
DATA_CACHE="/home/david-barnes/.cache/autoresearch"
export PATH="$HOME/.local/bin:$PATH"

echo "=========================================="
echo "Nano-Mythos: 24h Architecture Validation"
echo "=========================================="
echo ""

pip install -q rustbpe tiktoken pyarrow requests 2>&1 | tail -1
python training/nano_mythos_train.py
