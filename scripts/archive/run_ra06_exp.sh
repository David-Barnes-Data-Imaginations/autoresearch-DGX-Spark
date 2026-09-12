#!/bin/bash
set -e
# RA-06 (Parcae LTI) experiment runner for Nano-Mythos on DGX Spark 1.
#
# Usage (from the repo root, with NANO_* env vars exported by the caller):
#   NANO_TIME_BUDGET=1200 NANO_MICRO_BATCH=4 NANO_GRAD_ACCUM=16 \
#   NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1 \
#   NANO_CKPT_TAG=ra06_full \
#   bash run_ra06_exp.sh
#
# The NANO_* env vars are forwarded into the Docker container. Output is teed
# to logs/<NANO_CKPT_TAG>.log.
#
# Safety: NCCL_P2P_DISABLE=1, micro-batch kept small, expandable segments.

WORKSPACE="/home/david-barnes/autoresearch-DGX-Spark"
DATA_CACHE="/home/david-barnes/.cache/autoresearch"
IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
TAG="${NANO_CKPT_TAG:-ra06_default}"

# Defaults for any NANO_ var the caller did not set (match the original 24h run
# for the batch dimensions; a short-run caller overrides these).
: "${NANO_SEQ_LEN:=512}"
: "${NANO_MICRO_BATCH:=16}"
: "${NANO_GRAD_ACCUM:=64}"
: "${NANO_DEVICE_BATCH:=16}"
: "${NANO_TIME_BUDGET:=86400}"
: "${NANO_EVAL_EVERY:=200}"
: "${NANO_LOG_EVERY:=10}"
: "${NANO_LEARNING_RATE:=0.0003}"
: "${NANO_WARMUP_STEPS:=100}"
: "${NANO_GRAD_CLIP:=1.0}"
: "${NANO_MAX_STEPS:=0}"
: "${NANO_MAX_LOOP_ITERS:=8}"
: "${NANO_PARCAE_E_NORM:=1}"
: "${NANO_PARCAE_DEPTH_SAMPLE:=1}"
: "${NANO_PARCAE_INIT:=1}"

cd "$WORKSPACE"
mkdir -p logs

echo "=========================================="
echo "RA-06 Parcae LTI experiment: $TAG"
echo "  seq=$NANO_SEQ_LEN mb=$NANO_MICRO_BATCH ga=$NANO_GRAD_ACCUM time=${NANO_TIME_BUDGET}s"
echo "  lr=$NANO_LEARNING_RATE eval_every=$NANO_EVAL_EVERY loops=$NANO_MAX_LOOP_ITERS"
echo "  parcae_e_norm=$NANO_PARCAE_E_NORM depth_sample=$NANO_PARCAE_DEPTH_SAMPLE init=$NANO_PARCAE_INIT"
echo "=========================================="

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
    -e NANO_SEQ_LEN="$NANO_SEQ_LEN" \
    -e NANO_MICRO_BATCH="$NANO_MICRO_BATCH" \
    -e NANO_GRAD_ACCUM="$NANO_GRAD_ACCUM" \
    -e NANO_DEVICE_BATCH="$NANO_DEVICE_BATCH" \
    -e NANO_TIME_BUDGET="$NANO_TIME_BUDGET" \
    -e NANO_EVAL_EVERY="$NANO_EVAL_EVERY" \
    -e NANO_LOG_EVERY="$NANO_LOG_EVERY" \
    -e NANO_LEARNING_RATE="$NANO_LEARNING_RATE" \
    -e NANO_WARMUP_STEPS="$NANO_WARMUP_STEPS" \
    -e NANO_GRAD_CLIP="$NANO_GRAD_CLIP" \
    -e NANO_MAX_STEPS="$NANO_MAX_STEPS" \
    -e NANO_MAX_LOOP_ITERS="$NANO_MAX_LOOP_ITERS" \
    -e NANO_PARCAE_E_NORM="$NANO_PARCAE_E_NORM" \
    -e NANO_PARCAE_DEPTH_SAMPLE="$NANO_PARCAE_DEPTH_SAMPLE" \
    -e NANO_PARCAE_INIT="$NANO_PARCAE_INIT" \
    -e NANO_CKPT_TAG="$TAG" \
    "$IMAGE" \
    bash -c "pip install -q rustbpe tiktoken pyarrow requests 2>/dev/null; python training/nano_mythos_train.py" \
    2>&1 | tee "logs/${TAG}.log"

echo "RUN $TAG COMPLETE"
