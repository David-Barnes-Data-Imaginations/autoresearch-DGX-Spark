#!/bin/bash
# RA-15 Dory-style fixed loop count. Matched 1200-step runs on the Muon baseline.
#   R0 = baseline (adaptive: ACT halting + Poisson depth, max 8 loops), NANO_FIXED_LOOPS=0
#   R1 = NANO_FIXED_LOOPS=2   R2 = NANO_FIXED_LOOPS=4   R3 = NANO_FIXED_LOOPS=8
# Everything else identical (NANO_MUON=1, MoD-0.5, HC K=4, RoPE loop, Parcae).
# Smoke: RA15_SMOKE=1 (30 steps). Subset: RA15_ONLY="R1 R2".
set -e
cd /home/david-barnes/autoresearch-DGX-Spark
source /home/david-barnes/OpenMythos/.venv/bin/activate
export NCCL_P2P_DISABLE=1 TORCH_CUDA_ARCH_LIST=12.0 HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_cache_ra15_exp
export NANO_SEQ_LEN=512 NANO_MICRO_BATCH=8 NANO_GRAD_ACCUM=8 NANO_DEVICE_BATCH=8
export NANO_LEARNING_RATE=${RA15_LR:-0.0003} NANO_WARMUP_STEPS=${RA15_WARMUP:-100} NANO_GRAD_CLIP=1.0
export NANO_MAX_LOOP_ITERS=8 NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1
export NANO_ROPE_LOOP=1
export NANO_MOR=0 NANO_MOD=1 NANO_MOD_CAPACITY=0.5 NANO_MOD_NOISE=0.1
export NANO_HYPERLOOP=1 NANO_HYPERLOOP_STREAMS=4
export NANO_MUON=1 NANO_MUON_LR=0.02 NANO_MUON_WD=0.2
export NANO_BIGRAM=0 NANO_TRIGRAM=0 NANO_BEAM=0
export NANO_PRELUDE_LAYERS=${NANO_PRELUDE_LAYERS:-2} NANO_RECUR_LAYERS=${NANO_RECUR_LAYERS:-1} NANO_CODA_LAYERS=${NANO_CODA_LAYERS:-2}
if [ "${RA15_SMOKE:-0}" = "1" ]; then
  export NANO_MAX_STEPS=30 NANO_EVAL_EVERY=30 NANO_LOG_EVERY=5; TAG=smoke
else
  export NANO_MAX_STEPS=1200 NANO_EVAL_EVERY=150 NANO_LOG_EVERY=25; TAG=1200
fi
mkdir -p logs checkpoints /tmp/triton_cache_ra15_exp
sudo -n chown -R david-barnes:david-barnes checkpoints 2>/dev/null || true
ONLY="${RA15_ONLY:-R0 R1 R2 R3}"
run() { # arm loops name
  case " $ONLY " in *" $1 "*) ;; *) return 0;; esac
  echo "########## RA15 $1: fixed_loops=$2 ${TAG} ##########"
  NANO_FIXED_LOOPS=$2 NANO_CKPT_TAG=ra15_$1_$3${TAG} python training/nano_mythos_train.py 2>&1 | tee logs/ra15_$1_$3${TAG}.log
  echo "$1 DONE"
}
run R0 0 adaptive
run R1 2 fixed2
run R2 4 fixed4
run R3 8 fixed8
echo "RA15 RUNS COMPLETE"
