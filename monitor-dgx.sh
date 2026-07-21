#!/bin/bash
# Monitoring script for DGX Spark training

CONTAINER_NAME="autoresearch_training"

echo "=== DGX Spark Training Monitor ==="
echo ""

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Container '$CONTAINER_NAME' is not running."
    echo "Start training first with: ./run-dgx.sh"
    exit 1
fi

echo "Monitoring container: $CONTAINER_NAME"
echo "Press Ctrl+C to stop monitoring"
echo ""

# Monitor in real-time
watch -n 1 'echo "=== Docker Stats ===" && docker stats --no-stream "$CONTAINER_NAME" && echo "" && echo "=== GPU Usage ===" && nvidia-smi'
