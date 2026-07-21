# Quick Start Guide for DGX Spark

## Step 1: Launch Docker Container

```bash
./run-dgx.sh
```

You should see output like:
```
=== DGX Spark Autoresearch Launcher ===
Image: nvcr.io/nvidia/pytorch:25.12-py3
Container: autoresearch_training

Starting DGX Spark container with optimized settings...
```

## Step 2: Inside the Container

Once you're in the container bash prompt:

```bash
# Install dependencies (one-time)
uv sync

# Download data (one-time, start with 5 shards for testing)
uv run prepare.py --num-shards 5

# Run training (~5 minutes)
uv run train.py
```

## Step 3: Monitor from Host (Optional)

In a separate terminal on the host machine:

```bash
# Watch container resource usage
docker stats autoresearch_training

# Or use the monitoring script
./monitor-dgx.sh
```

## Expected Output

Training should start and show output like:
```
Vocab size: 8,192
Model config: {...}
Parameter counts:
  wte:                    33,554,432
  value_embeds:          ...
Estimated FLOPs per token: ...
Time budget: 300s
Gradient accumulation steps: 4
step 00001 (0.3%) | loss: 9.876543 | ...
```

## Troubleshooting

### Container won't start
- Check Docker is running: `docker info`
- Check NVIDIA runtime: `docker info | grep NVIDIA`
- Verify GPU: `nvidia-smi`

### OOM errors during training
- Reduce `DEVICE_BATCH_SIZE` in `train.py` (line 450) to 4 or 2
- Reduce `DEPTH` in `train.py` (line 449) to 2 or 3

### Very slow training
- Verify `pin_memory=True` in `prepare.py` line 326
- Ensure `non_blocking=True` in `prepare.py` line 355

### System freeze
- Ensure `--oom-score-adj 1000` is in `run-dgx.sh`
- This is already set in the script

## Next Steps

Once the initial test runs successfully:
1. Increase shards: `uv run prepare.py --num-shards 50`
2. Increase model depth: Change `DEPTH = 6` in `train.py`
3. Start autonomous research: Follow instructions in `program.md`
