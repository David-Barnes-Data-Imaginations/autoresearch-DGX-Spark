#!/bin/bash
# RA-01 MoR routing-dynamics monitor: 120 steps, log the per-step mor[] distribution.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra01
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=4 NANO_GRAD_ACCUM=4 NANO_DEVICE_BATCH=4
export NANO_MAX_STEPS=120 NANO_EVAL_EVERY=1000 NANO_LOG_EVERY=10
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=20 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1 NANO_ROPE_LOOP=1
export NANO_MOR=1 NANO_MOR_CHOICES=1,2,4,8 NANO_MOR_AUX_WEIGHT=0.01
NANO_CKPT_TAG=ra01_MON2_mor python training/nano_mythos_train.py 2>&1 | tee logs/ra01_MON2_mor.log
