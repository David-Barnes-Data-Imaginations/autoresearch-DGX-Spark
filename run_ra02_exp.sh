#!/bin/bash
# RA-02 Hyperloop Multi-Stream 1200-step matched pair (adoption-relevant, MoD-off).
# Per carry-over rule: HC and MoD both rewrite the cross-loop state update path
# (same code path), so they CONFLICT -> isolate HC by running MoD-OFF (RA-03
# precedent: R0/R1 MoD-off to isolate the mechanism).
#   R0 = NANO_HYPERLOOP=0  (Parcae + RoPE, MoD-off; isolates the HC mechanism)
#   R1 = NANO_HYPERLOOP=1  (+ loop-level hyper-connection over K=4 streams)
# Both: Parcae (e-norm, depth-sample, init rho~0.95) + RoPE loop-index (adopted
# wins). Only NANO_HYPERLOOP differs R0->R1.
# Venv route (proven RA-01/03/04/06/08/11). nccl p2p off, mb=8, seq=512.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra02
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1
export NANO_MOR=0 NANO_MOD=0 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1
export NANO_HYPERLOOP_STREAMS=4

echo "########## RA02 R0: baseline (Parcae+RoPE, MoD-off, HC OFF) 1200 ##########"
NANO_HYPERLOOP=0 NANO_CKPT_TAG=ra02_R0_baseline1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra02_R0_baseline1200.log
echo "R0 DONE"
echo "########## RA02 R1: baseline + HC hyperloop K=4 (MoD-off) 1200 ##########"
NANO_HYPERLOOP=1 NANO_CKPT_TAG=ra02_R1_hc1200 python training/nano_mythos_train.py 2>&1 | tee logs/ra02_R1_hc1200.log
echo "R1 DONE"
echo "RA02 1200-STEP MATCHED PAIR COMPLETE"
