#!/bin/bash
# RA-05 R2: robustness check — STRONGER rank-adaptive depth LoRA.
# R1 (gamma=0.1, base_rank=8) was a precise null (2.304971 vs R0 2.304970).
# R2 raises the mechanism's influence: gamma=0.3 (3x), base_rank=16 (2x).
# Isolation: R0 (no LoRA, 2.304970) is the control; only RAD_LO/RAD_GAMMA/
# RAD_BASE_RANK differ. Tests whether any signal emerges at higher influence.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra05
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1
export NANO_MOR=0 NANO_MOD=1 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1

echo "########## RA05 R2: stronger RAD_LO (gamma=0.3, base_rank=16) 1200 ##########"
NANO_RAD_LO=1 NANO_RAD_GAMMA=0.3 NANO_RAD_BASE_RANK=16 NANO_CKPT_TAG=ra05_R2_radlo_strong_1200 \
  python training/nano_mythos_train.py 2>&1 | tee logs/ra05_R2_radlo_strong_1200.log
echo "R2 DONE"
echo "RA05 R2 COMPLETE"
