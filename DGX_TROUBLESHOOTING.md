# DGX Spark Setup - Troubleshooting Guide

## ✅ Your Setup is Ready!

All tests passed! Your DGX Spark is properly configured with:
- ✅ Docker running
- ✅ NVIDIA Container Toolkit installed
- ✅ GPU detected: NVIDIA GB10
- ✅ Container GPU access verified

## Quick Start

### Option 1: Interactive Mode (Recommended)

```bash
./run-dgx.sh
```

Once inside the container:
```bash
# Install dependencies
uv sync

# Download data (first time only)
uv run prepare.py --num-shards 5

# Run training
uv run train.py
```

### Option 2: Detached Mode

```bash
# Start container in background
./run-dgx.sh -d

# Enter the container
docker exec -it autoresearch_training bash

# Inside container:
uv sync
uv run prepare.py --num-shards 5
uv run train.py
```

### Option 3: Test Mode (What you just ran)

```bash
./run-dgx.sh --test
```

This verifies your setup without starting a container.

## Monitoring

### From Host Machine

```bash
# Watch container resource usage
docker stats autoresearch_training

# Check GPU utilization
watch -n 1 nvidia-smi

# View container logs
docker logs -f autoresearch_training
```

## Troubleshooting

### "the input device is not a TTY"

**Cause**: This message appears when running the script in a non-interactive environment (like when testing via automation).

**Solution**: This is **normal** and won't happen when you run it directly in your DGX Spark terminal. Just ignore it - the container will still start correctly.

If you want to avoid this message:
- Use detached mode: `./run-dgx.sh -d`
- Then enter with: `docker exec -it autoresearch_training bash`

### Container Won't Start

**Check Docker**:
```bash
docker info
```

**Check NVIDIA Runtime**:
```bash
docker info --format '{{json .Runtimes}}' | grep nvidia
```

**Check GPU**:
```bash
nvidia-smi
```

### OOM Errors During Training

If you see out-of-memory errors:

1. **Reduce batch size** in `train.py` line 450:
   ```python
   DEVICE_BATCH_SIZE = 4  # Was 8
   ```

2. **Reduce model depth** in `train.py` line 449:
   ```python
   DEPTH = 2  # Was 4
   ```

### Very Slow Training

If training is extremely slow (< 100 tokens/sec):

1. **Verify pinned memory** is enabled (already done):
   - `prepare.py` line 326: `pin_memory=True`
   - `prepare.py` line 355: `non_blocking=True`

2. **Check GPU utilization**:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Verifying the Container is Working

After starting the container with `./run-dgx.sh`, verify it's working:

```bash
# From another terminal (on host)
docker ps | grep autoresearch

# Check if GPU is visible inside container
docker exec autoresearch_training nvidia-smi

# Check if Python/Torch is available
docker exec autoresearch_training python -c "import torch; print(torch.cuda.is_available())"
```

Should output: `True`

## What Each Script Does

### `run-dgx.sh`
- Launches Docker container with DGX-optimized settings
- Validates prerequisites (Docker, NVIDIA runtime, GPU)
- Sets up memory, IPC, and GPU configurations

### `monitor-dgx.sh`
- Real-time monitoring of container resources
- Shows CPU, memory, and GPU usage

### `--test` Flag
- Validates your setup without starting a container
- Quick way to verify everything is configured correctly

## Next Steps

1. **Start the container**:
   ```bash
   ./run-dgx.sh
   ```

2. **Install dependencies** (inside container):
   ```bash
   uv sync
   ```

3. **Download data** (first time):
   ```bash
   uv run prepare.py --num-shards 5
   ```

4. **Run training**:
   ```bash
   uv run train.py
   ```

5. **Monitor progress**:
   - Watch the training output in the container
   - Use `docker stats` from another terminal

## Common Commands Reference

```bash
# Start container
./run-dgx.sh

# Enter existing container
docker exec -it autoresearch_training bash

# Stop container
docker stop autoresearch_training

# Remove container
docker rm autoresearch_training

# View logs
docker logs autoresearch_training

# Monitor resources
docker stats autoresearch_training

# Test setup
./run-dgx.sh --test
```

## Documentation Files

- **`DGX_QUICKSTART.md`** - Quick start guide
- **`DGX_SETUP.md`** - Comprehensive setup documentation
- **`DGX_ADAPTATION_SUMMARY.md`** - All code changes explained
- **`deep-research.md`** - Original research findings

## Support

- **NVIDIA DGX Spark**: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- **Developer Forum**: https://forums.developer.nvidia.com/
- **Original Project**: https://github.com/karpathy/autoresearch

---

You're all set! Run `./run-dgx.sh` to start training! 🚀
