# 🚀 DGX Spark Training - Getting Started

## Quick Start (3 Steps)

### Step 1: Launch Docker Container

```bash
./run-dgx.sh
```

This starts a Docker container with all DGX Spark optimizations:
- Pinned memory for fast data transfers
- OOM protection to prevent system freezes
- 64GB shared memory for dataloaders
- GB10 architecture support (SM_121)

### Step 2: Install Dependencies (First Time Only)

Inside the container:

```bash
uv sync
```

### Step 3: Run Training

```bash
# Download minimal data for testing (5 shards)
uv run prepare.py --num-shards 5

# Run a 5-minute training experiment
uv run train.py
```

## What Changed for DGX Spark?

### Model Configuration
- **DEPTH**: 8 → 4 (50% memory reduction)
- **DEVICE_BATCH_SIZE**: 128 → 8 (prevents OOM)
- **TOTAL_BATCH_SIZE**: 2^19 → 2^16 (maintains stability)

### Code Optimizations
- ✅ Pinned memory enabled (`pin_memory=True`)
- ✅ Non-blocking transfers (`non_blocking=True`)
- ✅ OOM detection and recovery
- ✅ NCCL P2P disabled for unified memory
- ✅ GB10 architecture targeting (SM_121)

### Docker Configuration
- `--ipc=host`: Critical for dataloader
- `--gpus all`: Full GPU access
- `--oom-score-adj 1000`: Prevents system freezes
- `--shm-size 64gb`: Dataloader stability

## Expected Output

```
Vocab size: 8,192
Model config: {'sequence_len': 2048, 'vocab_size': 8192, 'n_layer': 4, ...}
Parameter counts:
  wte:                    8,388,608
  value_embeds:          4,194,304
  lm_head:               8,388,608
  transformer_matrices:  18,874,368
  scalars:               8
  total:                 39,845,896
Estimated FLOPs per token: 4.718593e+08
Time budget: 300s
Gradient accumulation steps: 4
step 00001 (0.3%) | loss: 9.876543 | lrm: 0.00 | dt: 1250ms | tok/sec: 4,096 | mfu: 12.5% | epoch: 1 | remaining: 300s
...
```

## Monitoring

### From Host Machine (Separate Terminal)

```bash
# Watch container resource usage
docker stats autoresearch_training

# Check GPU utilization
watch -n 1 nvidia-smi
```

### Inside Container

```bash
# Monitor training in real-time
# (Already shown in training output)
```

## Troubleshooting

### Container Won't Start

**Check Docker**:
```bash
docker info
```

**Check NVIDIA Runtime**:
```bash
docker info | grep NVIDIA
# Should show: Runtimes: runc nvidia
```

**Check GPU**:
```bash
nvidia-smi
```

### OOM During Training

If you see:
```
OOM detected at step 42. Reducing batch size.
Reduce DEVICE_BATCH_SIZE to 4 and restart
```

**Solution**: Edit `train.py` line 450:
```python
DEVICE_BATCH_SIZE = 4  # Was 8
```

### Very Slow Training (< 100 tokens/sec)

**Check pinned memory** in `prepare.py` line 326:
```python
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
```

**Verify non-blocking** in `prepare.py` line 355:
```python
gpu_buffer.copy_(cpu_buffer, non_blocking=True)
```

### System Freeze (SSH Unresponsive)

This means OOM handling failed. Ensure `run-dgx.sh` has:
```bash
--oom-score-adj 1000
```

This forces the kernel to kill the container before vital system processes.

## Scaling Up

Once the initial test runs successfully:

### 1. Increase Data
```bash
uv run prepare.py --num-shards 50  # More training data
```

### 2. Increase Model Size (if stable)
Edit `train.py` line 449:
```python
DEPTH = 6  # Was 4
```

### 3. Increase Batch Size (if stable)
Edit `train.py` line 450:
```python
DEVICE_BATCH_SIZE = 16  # Was 8
```

### 4. Full Dataset
```bash
uv run prepare.py --num-shards -1  # All 6542 shards
```

## Autonomous Research

Once everything is stable:

```bash
# Follow instructions in program.md
# Your AI agent can now start experimenting!
```

The agent will:
1. Modify `train.py` to test new architectures
2. Run 5-minute experiments
3. Evaluate results (val_bpb)
4. Iterate on successful changes

## Performance Expectations

### Single DGX Spark

| Metric | Value |
|--------|-------|
| Model Size | ~40M parameters |
| Throughput | 500-1000 tokens/sec |
| Time/Experiment | 5 minutes |
| Experiments/Hour | 60-80 |

### Comparison to H100

| Aspect | DGX Spark | H100 |
|--------|-----------|------|
| Memory | 128GB unified | 80GB dedicated |
| Throughput | 30-50% | 100% |
| Model Size | Can run larger | Limited by VRAM |

## When NVLink Arrives

For distributed training across both DGX Sparks:

```bash
# Enable NCCL over 200GbE
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1

# Use PyTorch DDP
uv run train.py --distributed
```

Expected: ~1.9x speedup over single Spark

## Documentation

- **`QUICKSTART.md`** - This file (quick reference)
- **`DGX_SETUP.md`** - Comprehensive setup guide
- **`DGX_ADAPTATION_SUMMARY.md`** - All changes explained
- **`deep-research.md`** - Original research findings

## Support

- **NVIDIA DGX Spark**: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- **Developer Forum**: https://forums.developer.nvidia.com/
- **Original Project**: https://github.com/karpathy/autoresearch

---

**Ready to train?** Just run `./run-dgx.sh` and let's get started! 🎉
