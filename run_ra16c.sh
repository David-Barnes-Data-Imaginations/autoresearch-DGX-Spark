#!/bin/bash
# RA-16c: depth/loop ratio around the adopted 1:3:1 x3 baseline (rel FLOPs = P + R*L + C; baseline 11), seed 42, 1200 steps.
# R0 1:3:1 x3 (baseline rerun, 11) | R1 1:2:1 x5 (12, +9%) | R2 1:4:1 x2 (10, -9%) | R3 1:1:1 x9 (11, matched).
# Ends "RA16C COMPLETE". Usage: nohup bash run_ra16c.sh > logs/ra16c.log 2>&1 &
cd /home/david-barnes/autoresearch-DGX-Spark
NANO_ARMS_SMOKE=1 ARMS="s16c_P1R4C1:2:4:1:1:-:0 s16c_P1R1C1:9:1:1:1:-:0" bash run_nano_arms.sh || { echo "SMOKE FAILED"; exit 1; }
ARMS="ra16c_P1R3C1_L3:3:3:1:1:-:0 ra16c_P1R2C1_L5:5:2:1:1:-:0 ra16c_P1R4C1_L2:2:4:1:1:-:0 ra16c_P1R1C1_L9:9:1:1:1:-:0" bash run_nano_arms.sh
echo "RA16C COMPLETE"
