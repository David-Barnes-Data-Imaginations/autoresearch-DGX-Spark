# Running Autoresearch on NVIDIA DGX Spark

This guide explains how to run the autoresearch project on your NVIDIA DGX Spark workstation.

## Prerequisites

1. **NVIDIA Container Toolkit** installed:
   ```bash
   # Follow official installation guide:
   # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
   ```

2. **Docker** running and configured to use NVIDIA runtime

3. **DGX Spark** with GB10 Grace Blackwell Superchip accessible

## Quick Start

### 1. Launch Docker Container

```bash
chmod +x run-dgx.sh
./run-dgx.sh
```

This command starts a Docker container with:
- `--ipc=host`: Critical for dataloader communication
- `--gpus all`: Full access to GB10 GPU
- `--oom-score-adj 1000`: Prevents system-wide freezes on OOM
- `--shm-size 64gb`: Prevents dataloader shared memory issues
- `--ulimit memlock=-1`: Unlimited locked memory
- Environment variables optimized for DGX Spark

### 2. Inside the Container

Once inside the container, set up the environment:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source uv
source ~/.local/bin/env

# Install dependencies
uv sync
```

### 3. Prepare Data (First Time Only)

For initial testing with fewer shards:

```bash
uv run prepare.py --num-shards 5
```

For full dataset:

```bash
uv run prepare.py --num-shards -1  # Downloads all 6542 shards
```

### 4. Run Training

```bash
uv run train.py
```

## DGX-Specific Optimizations

### Memory Management

The DGX Spark uses unified memory architecture (128GB LPDDR5X shared between CPU and GPU). Key adaptations:

1. **Reduced Batch Size**: `DEVICE_BATCH_SIZE = 8` (down from 128)
   - Prevents OOM crashes
   - Maintains stability during long training runs

2. **Reduced Model Depth**: `DEPTH = 4` (down from 8)
   - ~50% memory reduction
   - Still provides meaningful research results

3. **Pinned Memory**: Enabled in dataloader for faster H2D transfers
   - Avoids 50× slowdown from pageable memory

4. **Non-blocking Transfers**: `non_blocking=True` for GPU copies

### Environment Variables

The script sets these automatically:

```bash
export NCCL_P2P_DISABLE=1        # Disable P2P on unified memory
export TORCH_CUDA_ARCH_LIST=12.0 # GB10 architecture (SM_121)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Monitoring

### Real-time Container Stats

```bash
# From host machine (outside container)
docker stats autoresearch_training
```

### GPU Utilization

```bash
# Inside container
watch -n 1 nvidia-smi
```

### Check Container Logs

```bash
docker logs -f autoresearch_training
```

## Troubleshooting

### System Freezes ("Zombie" State)

**Symptom**: SSH unresponsive, monitor frozen, system requires hard reset

**Solution**: Ensure `--oom-score-adj 1000` is set in `run-dgx.sh`

This ensures the kernel kills the container before vital system processes.

### CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce `DEVICE_BATCH_SIZE` further (try 4 or 2)
2. Reduce `DEPTH` in `train.py` (try 2 or 3)
3. Reduce `MAX_SEQ_LEN` in `prepare.py` (try 1024)

### Slow Training (50× slowdown)

**Symptom**: Training extremely slow, especially during data loading

**Causes**:
- Pageable CPU memory instead of pinned memory
- Missing `non_blocking=True` in transfers

**Solution**: Verify these are set in `prepare.py`:
```python
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
gpu_buffer.copy_(cpu_buffer, non_blocking=True)
```

### Dataloader Errors

**Symptom**: Shared memory errors, dataloader crashes

**Solution**: Increase shared memory:
```bash
--shm-size 64gb
```

### NCCL Communication Issues

**Symptom**: NCCL warnings or errors in logs

**Solution**: Disable P2P (already set in `run-dgx.sh`):
```bash
export NCCL_P2P_DISABLE=1
```

## Performance Expectations

### Single DGX Spark

- **Model**: DEPTH=4, ~10M parameters
- **Batch Size**: 8 per step
- **Effective Batch**: 2^16 = 65,536 tokens (with gradient accumulation)
- **Training Time**: 5 minutes per experiment
- **Expected Throughput**: ~500-1000 tokens/sec (varies by model size)

### Comparison to H100

The DGX Spark (GB10) has different characteristics than H100:
- **Memory**: Unified 128GB vs H100's 80GB dedicated
- **Bandwidth**: LPDDR5X vs HBM3
- **Compute**: Blackwell architecture with SM_121

Expect ~30-50% of H100 throughput for this workload, but with better memory capacity for larger models.

## Multi-Node Training (Future)

When your NVLink cable arrives, you can train across both DGX Sparks:

```bash
# Node 1 (master)
export MASTER_ADDR=node1_ip
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=0

# Node 2
export MASTER_ADDR=node1_ip
export MASTER_PORT=29500
export WORLD_SIZE=2
export RANK=1

# Use PyTorch DDP with NCCL backend
NCCL_IB_DISABLE=0 NCCL_IB_HCA=rocep1s0f1 uv run train.py --distributed
```

## Configuration Reference

### Key Parameters in `train.py`

```python
DEPTH = 4              # Model depth (reduce if OOM)
ASPECT_RATIO = 64      # Model dimension = depth * ASPECT_RATIO
DEVICE_BATCH_SIZE = 8  # Per-step batch size (reduce if OOM)
TOTAL_BATCH_SIZE = 2**16  # Effective batch with gradient accumulation
```

### Key Parameters in `prepare.py`

```python
MAX_SEQ_LEN = 2048     # Sequence length (reduce to 1024 if needed)
TIME_BUDGET = 300      # 5-minute training budget
EVAL_TOKENS = 40 * 524288  # Validation tokens
```

## Next Steps

1. **Initial Test**: Run with 5 shards and DEPTH=4
2. **Monitor**: Watch for OOM or system freezes
3. **Iterate**: Adjust batch size/depth based on stability
4. **Scale Up**: Increase shards and model size once stable
5. **Research**: Begin autonomous experimentation with `program.md`

## Additional Resources

- [NVIDIA DGX Spark Product Page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [DGX Spark Developer Forum](https://forums.developer.nvidia.com/)
