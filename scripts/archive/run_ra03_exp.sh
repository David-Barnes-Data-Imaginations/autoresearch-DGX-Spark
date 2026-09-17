#!/bin/bash
# RA-03 LT2 Hybrid Attention 1200-step matched pair.
# R0 = LT2-off baseline (Parcae+RoPE, MoD OFF per conflict note) 1200
# R1 = +LT2 (GDN on even loops, full GQA on odd) 1200
# MoD is HELD OFF in BOTH arms: MoD's per-loop token-bypass and GDN's
# full-sequence linear scan share the block code path; scanning GDN over MoD's
# top-k subsequence would break the "linear attention over the full sequence"
# semantics. Isolate the LT2 variable (only NANO_LT2 differs R0->R1).
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra03
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1
export NANO_MOR=0 NANO_MOD=0 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1

echo "########## RA03 R0: baseline (Parcae+RoPE, MoD OFF, LT2 OFF) 1200 ##########"
NANO_LT2=0 NANO_CKPT_TAG=ra03_R0_baseline1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra03_R0_baseline1200.log
echo "R0 DONE"
echo "########## RA03 R1: baseline + LT2 hybrid (GDN even/full odd) 1200 ##########"
NANO_LT2=1 NANO_CKPT_TAG=ra03_R1_lt2_1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra03_R1_lt2_1200.log
echo "R1 DONE"
echo "RA03 1200-STEP MATCHED PAIR COMPLETE"
