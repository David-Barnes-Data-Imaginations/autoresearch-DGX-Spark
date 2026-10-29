#!/bin/bash
# Generic matched-arm runner (RA-16+). Each arm = "tag:FIXED_LOOPS:RECUR_LAYERS:PRELUDE:CODA:LOOP_CHOICES:ILSD"
#   LOOP_CHOICES uses '/' as separator (e.g. 4/6/8) or '-' for none; ILSD 0/1.
# Usage: ARMS="ra16_A:8:1:2:2:-:0 ra16_B:4:2:2:2:-:0" nohup bash run_nano_arms.sh > logs/<sweep>.log 2>&1 &
# Shared settings = RA-15 baseline (Muon, MoD-0.5, HC K=4, RoPE loop, Parcae), 1200 steps, mb8 ga8 seq512 lr3e-4 cosine.
# Smoke: NANO_ARMS_SMOKE=1. Ends with "ARMS COMPLETE".
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_nano_arms
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_LEARNING_RATE=0.0003 NANO_WARMUP_STEPS=100 NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1 NANO_MOR=0 NANO_MOD=1 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1
export NANO_HYPERLOOP=1 NANO_HYPERLOOP_STREAMS=4
export NANO_MUON=1 NANO_MUON_LR=0.02 NANO_MUON_WD=0.2
export NANO_BIGRAM=0 NANO_TRIGRAM=0 NANO_BEAM=0
if [ "${NANO_ARMS_SMOKE:-0}" = "1" ]; then export NANO_MAX_STEPS=30 NANO_EVAL_EVERY=30 NANO_LOG_EVERY=5; SUF=smoke
else export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25; SUF=1200; fi
mkdir -p logs checkpoints /tmp/triton_cache_nano_arms
sudo -n chown -R david-barnes:david-barnes checkpoints 2>/dev/null || true
for arm in ${ARMS:?set ARMS}; do
  IFS=: read -r TAG FL RL PL CL CH IL <<< "$arm"
  CH_ENV=""; [ "$CH" != "-" ] && CH_ENV="${CH//\//,}"
  echo "########## ARM $TAG fixed=$FL recur=$RL prelude=$PL coda=$CL choices=${CH_ENV:-none} ilsd=$IL ##########"
  NANO_FIXED_LOOPS=$FL NANO_RECUR_LAYERS=$RL NANO_PRELUDE_LAYERS=$PL NANO_CODA_LAYERS=$CL \
  NANO_LOOP_CHOICES="$CH_ENV" NANO_ILSD=$IL NANO_CKPT_TAG=${TAG}_${SUF} \
    python training/nano_mythos_train.py 2>&1 | tee logs/${TAG}_${SUF}.log
  echo "$TAG DONE"
done
echo "ARMS COMPLETE"
