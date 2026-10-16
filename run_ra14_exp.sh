#!/bin/bash
# RA-14 hashed bigram embeddings: matched pair on Muon baseline.
#   R0 = Parcae+RoPE+MoD-0.5+HC K=4+Muon, NANO_BIGRAM=0
#   R1 = same + NANO_BIGRAM=1 (gate-init-0 hash table before prelude)
# Both arms: NANO_MUON=1. Only NANO_BIGRAM differs.
# Smoke: RA14_SMOKE=1 (30 steps). Optional R2 trigram later via NANO_TRIGRAM=1.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra14_exp
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1
export NANO_MOR=0 NANO_MOD=1 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1
export NANO_HYPERLOOP=1 NANO_HYPERLOOP_STREAMS=4
export NANO_MUON=1 NANO_MUON_LR=0.02 NANO_MUON_WD=0.2
export NANO_TRIGRAM=0

if [ "${RA14_SMOKE:-0}" = "1" ]; then
  export NANO_MAX_STEPS=30 NANO_EVAL_EVERY=30 NANO_LOG_EVERY=5
  TAG=smoke
else
  export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25
  TAG=1200
fi

mkdir -p logs checkpoints /tmp/triton_cache_ra14_exp
sudo -n chown -R david-barnes:david-barnes checkpoints 2>/dev/null || true

echo "########## RA14 R0: Muon baseline (BIGRAM OFF) ${TAG} ##########"
NANO_BIGRAM=0 NANO_CKPT_TAG=ra14_R0_baseline${TAG} python training/nano_mythos_train.py 2>&1 | tee logs/ra14_R0_baseline${TAG}.log
echo "R0 DONE"
echo "########## RA14 R1: Muon + hashed bigram ON ${TAG} ##########"
NANO_BIGRAM=1 NANO_CKPT_TAG=ra14_R1_bigram${TAG} python training/nano_mythos_train.py 2>&1 | tee logs/ra14_R1_bigram${TAG}.log
echo "R1 DONE"
echo "RA14 MATCHED PAIR COMPLETE"
