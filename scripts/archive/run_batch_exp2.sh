#!/bin/bash
set -e

# Batch experiment runner for DGX Spark - Session 6
# Tests configurations not yet explored in the speedrun sweep
# Uses run_exp_spark.sh which runs a COPY of train.py (safe, no in-place patching)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "BATCH EXPERIMENTS 30-34 ON DGX SPARK"
echo "========================================"

# Experiment 30: WEIGHT_DECAY=0.2 (higher than 0.1)
echo ""
echo ">>> Running Experiment 30: wd_020"
bash run_exp_spark.sh "wd_020" "s/WEIGHT_DECAY = 0.1/WEIGHT_DECAY = 0.2/"

# Experiment 31: ADAM_BETAS=(0.7, 0.95) (lower beta1 from 0.8)
echo ""
echo ">>> Running Experiment 31: betas_07"
bash run_exp_spark.sh "betas_07" "s/ADAM_BETAS = (0.8, 0.95)/ADAM_BETAS = (0.7, 0.95)/"

# Experiment 32: WARMDOWN_RATIO=0.05 (shorter warmdown from 0.1)
echo ""
echo ">>> Running Experiment 32: warmdown_005"
bash run_exp_spark.sh "warmdown_005" "s/WARMDOWN_RATIO = 0.1/WARMDOWN_RATIO = 0.05/"

# Experiment 33: EMBEDDING_LR=0.60 (lower from 0.65)
echo ""
echo ">>> Running Experiment 33: emb_lr_060"
bash run_exp_spark.sh "emb_lr_060" "s/EMBEDDING_LR = 0.65/EMBEDDING_LR = 0.60/"

# Experiment 34: MATRIX_LR=0.035 (lower from 0.04)
echo ""
echo ">>> Running Experiment 34: matrix_lr_035"
bash run_exp_spark.sh "matrix_lr_035" "s/MATRIX_LR = 0.04/MATRIX_LR = 0.035/"

echo ""
echo "========================================"
echo "ALL BATCH EXPERIMENTS COMPLETE"
echo "========================================"
