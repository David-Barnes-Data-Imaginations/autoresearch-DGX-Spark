#!/bin/bash
# RA-16 part 2 + RA-16b: smoke prelude=1 config, then 1200-step arms. Ends "RA16 PART2 COMPLETE".
cd /home/david-barnes/autoresearch-DGX-Spark
NANO_ARMS_SMOKE=1 ARMS="s16b_P1R3C1:3:3:1:1:-:0" bash run_nano_arms.sh || { echo "SMOKE FAILED"; exit 1; }
ARMS="ra16b_P1R3C1_L3:3:3:1:1:-:0 ra16b_P2R1C2_L7:7:1:2:2:-:0 ra16_L8R2:8:2:2:2:-:0 ra16b_P2R5C2_L2:2:5:2:2:-:0" bash run_nano_arms.sh
echo "RA16 PART2 COMPLETE"
