#!/bin/bash
# RA-06 ablation driver: runs A0-A3 sequentially on Spark 1 (single GPU).
# Fixed-step head-to-head so val_bpb + stability are directly comparable.
set -e
cd /home/david-barnes/autoresearch-DGX-Spark

COMMON=(
  NANO_SEQ_LEN=512
  NANO_MICRO_BATCH=8
  NANO_GRAD_ACCUM=8
  NANO_DEVICE_BATCH=8
  NANO_MAX_STEPS=250
  NANO_EVAL_EVERY=100
  NANO_LOG_EVERY=10
  NANO_LEARNING_RATE=0.0003
  NANO_WARMUP_STEPS=100
  NANO_GRAD_CLIP=1.0
  NANO_MAX_LOOP_ITERS=8
)

echo "########## A0: baseline (orig LTI init, no e-norm, no depth sample) ##########"
env "${COMMON[@]}" NANO_PARCAE_E_NORM=0 NANO_PARCAE_DEPTH_SAMPLE=0 NANO_PARCAE_INIT=0 NANO_CKPT_TAG=ra06_a0_baseline bash run_ra06_exp.sh

echo "########## A1: +Parcae LTI init (rho=0.95) ##########"
env "${COMMON[@]}" NANO_PARCAE_E_NORM=0 NANO_PARCAE_DEPTH_SAMPLE=0 NANO_PARCAE_INIT=1 NANO_CKPT_TAG=ra06_a1_init bash run_ra06_exp.sh

echo "########## A2: +e-norm (e=LN(P(s))) ##########"
env "${COMMON[@]}" NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=0 NANO_PARCAE_INIT=1 NANO_CKPT_TAG=ra06_a2_enorm bash run_ra06_exp.sh

echo "########## A3: full Parcae (init + e-norm + depth sample) ##########"
env "${COMMON[@]}" NANO_PARCAE_E_NORM=1 NANO_PARCAE_DEPTH_SAMPLE=1 NANO_PARCAE_INIT=1 NANO_CKPT_TAG=ra06_a3_full bash run_ra06_exp.sh

echo "########## ALL RA-06 ABLATIONS COMPLETE ##########"
