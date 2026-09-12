#!/bin/bash
# RA-08 confirmation: matched 1200-step comparison.
# Tests whether the small 800-step win (-0.0014) compounds with training time
# (RA-06's real win grew monotonically: -0.0035 -> -0.0070). If the RoPE gap
# grows at 1200, it is a genuine late-stage mechanism, not noise.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark

COMMON=(
  NANO_SEQ_LEN=512
  NANO_MICRO_BATCH=8
  NANO_GRAD_ACCUM=8
  NANO_DEVICE_BATCH=8
  NANO_MAX_STEPS=1200
  NANO_EVAL_EVERY=150
  NANO_LOG_EVERY=10
  NANO_LEARNING_RATE=0.0003
  NANO_WARMUP_STEPS=100
  NANO_GRAD_CLIP=1.0
  NANO_MAX_LOOP_ITERS=8
  NANO_PARCAE_E_NORM=1
  NANO_PARCAE_DEPTH_SAMPLE=1
  NANO_PARCAE_INIT=1
)

echo "########## C0: full Parcae (no RoPE) 1200 ##########"
env "${COMMON[@]}" NANO_ROPE_LOOP=0 NANO_CKPT_TAG=ra08_C0_baseline1200 bash run_ra08_exp.sh

echo "########## C1: full Parcae + RoPE loop-index 1200 ##########"
env "${COMMON[@]}" NANO_ROPE_LOOP=1 NANO_CKPT_TAG=ra08_C1_rope1200 bash run_ra08_exp.sh

echo "RA08 1200-STEP CONFIRMATION COMPLETE"
