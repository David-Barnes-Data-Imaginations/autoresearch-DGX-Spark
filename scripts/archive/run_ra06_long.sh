#!/bin/bash
# RA-06 longer matched comparison: baseline vs full Parcae at 800 steps.
# Tests whether the stability advantage manifests as training progresses past 250 steps.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark

COMMON=(
  NANO_SEQ_LEN=512
  NANO_MICRO_BATCH=8
  NANO_GRAD_ACCUM=8
  NANO_DEVICE_BATCH=8
  NANO_MAX_STEPS=800
  NANO_EVAL_EVERY=100
  NANO_LOG_EVERY=10
  NANO_LEARNING_RATE=0.0003
  NANO_WARMUP_STEPS=100
  NANO_GRAD_CLIP=1.0
  NANO_MAX_LOOP_ITERS=8
)

echo "########## L0: baseline 800 steps ##########"
env "${COMMON[@]}" NANO_PARCAE_E_NORM=0 NANO_PARCAE_DEPTH_SAMPLE=0 NANO_PARCAE_INIT=0 NANO_CKPT_TAG=ra06_L0_baseline800 bash run_ra06_exp.sh

echo "########## L3: full Parcae 800 steps ##########"
env "${COMMON[@]}" NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1 NANO_CKPT_TAG=ra06_L3_full800 bash run_ra06_exp.sh

echo "########## RA-06 LONGER COMPARISON COMPLETE ##########"
