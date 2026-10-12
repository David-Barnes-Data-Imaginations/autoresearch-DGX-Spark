#!/bin/bash
# RA-07 Latent CoT Supervision: matched pair on adopted baseline.
#   R0 = Parcae+RoPE+MoD-0.5+HC K=4, LATENT_COT=0  (expect ~2.301180)
#   R1 = same + LATENT_COT=1 (aux CE at t={2,4,6}, lambda=0.1*(t/T), warmup 2000)
# Only NANO_LATENT_COT differs.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra07
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1
export NANO_MOR=0 NANO_MOD=1 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1
export NANO_HYPERLOOP=1 NANO_HYPERLOOP_STREAMS=4
export NANO_LATENT_AUX_WEIGHT=0.1 NANO_LATENT_WARMUP_STEPS=2000

mkdir -p logs checkpoints /tmp/triton_cache_ra07

echo "########## RA07 R0: baseline (LATENT_COT OFF) 1200 ##########"
NANO_LATENT_COT=0 NANO_CKPT_TAG=ra07_R0_baseline1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra07_R0_baseline1200.log
echo "R0 DONE"
echo "########## RA07 R1: Latent CoT aux ON 1200 ##########"
NANO_LATENT_COT=1 NANO_CKPT_TAG=ra07_R1_latent1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra07_R1_latent1200.log
echo "R1 DONE"
echo "RA07 MATCHED PAIR COMPLETE"
