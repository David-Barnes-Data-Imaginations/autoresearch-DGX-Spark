#!/bin/bash
# RA-11 1200-step matched pair: recycle-off baseline (Parcae+RoPE+MoD-0.5)
# vs recycle-on (alpha 0.7 dynamic). Same seed/data/config except NANO_KV_RECYCLE.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra11
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1
export NANO_MOR=0 NANO_MOD=1 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1
export NANO_KV_ALPHA=0.7 NANO_KV_DYNAMIC=1

echo "########## RA11 R0: baseline (Parcae+RoPE+MoD-0.5, recycle OFF) 1200 ##########"
NANO_KV_RECYCLE=0 NANO_CKPT_TAG=ra11_R0_baseline1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra11_R0_baseline1200.log
echo "R0 DONE"
echo "########## RA11 R1: baseline + KV recycle alpha 0.7 dynamic (ON) 1200 ##########"
NANO_KV_RECYCLE=1 NANO_CKPT_TAG=ra11_R1_recycle1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra11_R1_recycle1200.log
echo "R1 DONE"
echo "RA11 1200-STEP MATCHED PAIR COMPLETE"
