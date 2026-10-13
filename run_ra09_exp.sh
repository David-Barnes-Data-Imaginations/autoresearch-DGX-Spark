#!/bin/bash
# RA-09 Dynamic ACT Ponder Loss: matched pair on adopted baseline.
#   R0 = Parcae+RoPE+MoD-0.5+HC K=4, ACT_PONDER=0  (expect ~2.301180)
#   R1 = same + ACT_PONDER=1 (loss += tau * mean(N + R), tau=0.001, no anneal)
# Only NANO_ACT_PONDER differs.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra09_exp
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1
export NANO_MOR=0 NANO_MOD=1 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1
export NANO_HYPERLOOP=1 NANO_HYPERLOOP_STREAMS=4
export NANO_PONDER_TAU=0.001 NANO_PONDER_WARMUP_STEPS=0

mkdir -p logs checkpoints /tmp/triton_cache_ra09_exp
# Container runs can leave root-owned checkpoints/ (skill pitfall 5).
sudo -n chown -R david-barnes:david-barnes checkpoints 2>/dev/null || true

echo "########## RA09 R0: baseline (ACT_PONDER OFF) 1200 ##########"
NANO_ACT_PONDER=0 NANO_CKPT_TAG=ra09_R0_baseline1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra09_R0_baseline1200.log
echo "R0 DONE"
echo "########## RA09 R1: ACT ponder loss ON (tau=0.001) 1200 ##########"
NANO_ACT_PONDER=1 NANO_CKPT_TAG=ra09_R1_ponder1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra09_R1_ponder1200.log
echo "R1 DONE"
echo "RA09 MATCHED PAIR COMPLETE"
