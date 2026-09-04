#!/bin/bash
# RA-08 venv 1200-step matched pair (OpenMythos GPU venv, same torch for both).
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1
export TORCH_CUDA_ARCH_LIST=12.0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export NANO_SEQ_LEN=512
export NANO_MICRO_BATCH=8
export NANO_GRAD_ACCUM=8
export NANO_DEVICE_BATCH=8
export NANO_MAX_STEPS=1200
export NANO_EVAL_EVERY=150
export NANO_LOG_EVERY=10
export NANO_LEARNING_RATE=0.0003
export NANO_WARMUP_STEPS=100
export NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8
export NANO_PARCAE_E_NORM=1
export NANO_PARCAE_DEPTH_SAMPLE=1
export NANO_PARCAE_INIT=1

echo "########## V0: full Parcae (no RoPE) 1200 [venv] ##########"
NANO_ROPE_LOOP=0 NANO_CKPT_TAG=ra08_V0_baseline1200_venv python training/nano_mythos_train.py 2>&1 | tee logs/ra08_V0_baseline1200_venv.log

echo "########## V1: full Parcae + RoPE loop-index 1200 [venv] ##########"
NANO_ROPE_LOOP=1 NANO_CKPT_TAG=ra08_V1_rope1200_venv python training/nano_mythos_train.py 2>&1 | tee logs/ra08_V1_rope1200_venv.log

echo "RA08 VENV 1200-STEP MATCHED PAIR COMPLETE"
