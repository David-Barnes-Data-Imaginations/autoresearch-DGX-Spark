#!/bin/bash
set -e

# Run a training experiment on Spark 2 (GB10 GPU node)
# Usage: bash run_exp_spark.sh "experiment_name" "sed_expr1" "sed_expr2" ...
# Patches a COPY of train.py, runs training, leaves original untouched

EXPERIMENT_NAME="$1"
shift

echo "========================================"
echo "EXPERIMENT: $EXPERIMENT_NAME"
echo "========================================"

IMAGE="nvcr.io/nvidia/pytorch:25.12-py3"
WORKSPACE="/home/david-barnes/autoresearch-DGX-Spark"
DATA_CACHE="/home/david-barnes/.cache/autoresearch"

# Write sed expressions to a file (one expression per line, without -e prefix)
# sed -f reads expressions from a file, one per line
SED_FILE=$(mktemp /tmp/hermes-sed.XXXXXX)
for expr in "$@"; do
    echo "$expr" >> "$SED_FILE"
done

# Create a patch script to run inside Docker
PATCH_SCRIPT=$(mktemp /tmp/hermes-patch.XXXXXX)
cat > "$PATCH_SCRIPT" << 'PATCH_EOF'
set -e
pip install -q rustbpe tiktoken pyarrow requests 2>&1 | tail -1
cp train.py train_exp.py
sed -i -f /tmp/sed_exprs.txt train_exp.py
echo '--- Config ---'
grep -E '^EMBEDDING_LR|^SCALAR_LR|^MATRIX_LR|^UNEMBEDDING_LR|^WARMUP_RATIO|^softcap|^WEIGHT_DECAY' train_exp.py
python train_exp.py 2>&1
rm -f train_exp.py
PATCH_EOF

docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size 64gb --oom-score-adj 1000 \
    -v "$DATA_CACHE":"$DATA_CACHE" \
    -v "$WORKSPACE":"$WORKSPACE" \
    -v "$SED_FILE":/tmp/sed_exprs.txt \
    -v "$PATCH_SCRIPT":/tmp/patch_script.sh \
    -w "$WORKSPACE" \
    -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e NCCL_P2P_DISABLE=1 \
    -e TORCH_CUDA_ARCH_LIST=12.0 \
    -e HOME=/home/david-barnes \
    "$IMAGE" \
    bash /tmp/patch_script.sh 2>&1

rm -f "$SED_FILE" "$PATCH_SCRIPT"

echo ""
echo "EXPERIMENT $EXPERIMENT_NAME COMPLETE"
