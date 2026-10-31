#!/bin/bash
# RA-16b seed-2 stability check: FLOP-matched control 2:1:2 x7 (R0) then 1:3:1 x3 (R1), NANO_SEED=2, 1200 steps.
# Ends "RA16B SEED2 COMPLETE". Usage: nohup bash run_ra16b_seed2.sh > logs/ra16b_seed2.log 2>&1 &
cd /home/david-barnes/autoresearch-DGX-Spark
export NANO_SEED=2
NANO_ARMS_SMOKE=1 ARMS="s16b_seed2:3:3:1:1:-:0" bash run_nano_arms.sh || { echo "SMOKE FAILED"; exit 1; }
grep -q "NANO_SEED=2" logs/s16b_seed2_smoke.log || { echo "SEED NOT APPLIED"; exit 1; }
ARMS="ra16b_P2R1C2_L7_s2:7:1:2:2:-:0 ra16b_P1R3C1_L3_s2:3:3:1:1:-:0" bash run_nano_arms.sh
echo "RA16B SEED2 COMPLETE"
