#!/bin/bash
# RA-04 R2: plan's headline prescription — T=16 loops with MoD cap 0.5
# (16 loops of depth at ~8 loops of block compute) vs R0 baseline.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra04
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1 NANO_MOR=0 NANO_MOD_NOISE=0.1

echo "########## RA04 R2: MoD cap 0.5, T=16 (deep loops, same block compute) 1200 ##########"
NANO_MOD=1 NANO_MOD_CAPACITY=0.5 NANO_MAX_LOOP_ITERS=16 NANO_CKPT_TAG=ra04_R2_mod05_T16_1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra04_R2_mod05_T16_1200.log
echo "R2 DONE"
