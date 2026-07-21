# Research Synthesis: Running Karpathy’s `autoresearch` on NVIDIA DGX Sparks

## Executive Summary
Andre Karpathy’s `autoresearch` project is designed for single-GPU H100 environments but is compatible with the NVIDIA DGX Spark’s GB10 Grace Blackwell Superchip. The primary challenge is the DGX Spark’s **unified memory architecture**, where CPU and GPU share 128GB of LPDDR5X memory. Unlike traditional GPUs, memory exhaustion on the Spark often causes system-wide "zombie" freezes rather than clean CUDA OOM errors [1][4]. To run `autoresearch` successfully, you must enforce strict Docker container configurations, modify the training loop to use pinned memory, and configure OpenCode to manage the environment. Two DGX Sparks can be clustered for distributed training using NCCL over 200GbE interconnects [5][10].

## 1. Critical Docker Container Configuration
The DGX Spark requires all training workloads to run within Docker containers due to its unified memory architecture [2]. Standard `docker run` commands will likely fail due to shared memory limits and OOM handling.

**Required Flags:**
*   `--ipc=host`: Critical for dataloader communication processes [3].
*   `--gpus all`: Grants full access to the GB10 Superchip [7].
*   `--oom-score-adj 1000`: Ensures the kernel kills the container before vital system processes to prevent system freezes [4].
*   `--shm-size 64gb`: Increases shared memory beyond the default 64MB to prevent dataloader failures [6].

**Recommended Command:**
```bash
docker run --ipc=host --gpus all --oom-score-adj 1000 --shm-size 64gb \
  -v $(pwd):/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:25.12-py3
```
*Note: Pin specific container tags (e.g., `25.12-py3`) for reproducibility [3].*

## 2. Code Modifications for `autoresearch`
The `autoresearch` project (approx. 630 lines) requires specific adjustments to handle the GB10 architecture and unified memory constraints [8].

### A. Dependency Updates
Update `pyproject.toml` to support the Blackwell architecture (SM_121) and ARM64:
*   **PyTorch:** `>=2.9.0` (Includes ARM64 optimizations) [11].
*   **Triton:** `>=3.5.0` (Supports SM_121) [11].

### B. Memory Management
To avoid the "pathological slowdown" (50× slower H2D copies) and OOM crashes:
1.  **Pin Memory:** Enable `pin_memory=True` in the DataLoader to ensure accelerated transfers [11].
2.  **Pre-allocation:** Instantiate the model and optimizer *before* loading the dataset to prevent memory fragmentation during startup [4].
3.  **Batch Size:** Limit batch size to **≤6** in Docker containers to maintain stability [4].

**Modified Training Loop Snippet:**
```python
# Pre-allocate model
model = GPT().cuda()
optimizer = torch.optim.AdamW(model.parameters())

# Data Loading with pinned memory
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

# Training with OOM protection
for batch in dataloader:
    try:
        inputs = batch['input'].to(device, non_blocking=True)
        loss = model(inputs)
        loss.backward()
        optimizer.step()
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("OOM detected, reducing batch size")
            break
        raise
```

### C. Performance Optimization
*   **Gradient Checkpointing:** Enable for larger models to reduce memory footprint [4].
*   **Tensor Parallelism:** Set `--tensor-parallel-size 1` for single Spark workloads [3].

## 3. Multi-Node Training (2 DGX Sparks)
To utilize both DGX Sparks for larger models or faster training:

*   **Interconnect:** Use the 200GbE ConnectX-7 ports for RoCE communication [5].
*   **Framework:** Use PyTorch DistributedDataParallel (DDP) with the `nccl` backend [5].
*   **NCCL Configuration:**
    ```bash
    export NCCL_IB_DISABLE=0
    export NCCL_IB_HCA=rocep1s0f1
    export NCCL_SOCKET_IFNAME=eth0
    ```
*   **Scaling:** Dual Spark setup can provide up to 1.9x performance scaling over a single node [5].

## 4. Troubleshooting & Monitoring
| Issue | Symptom | Solution |
| :--- | :--- | :--- |
| **System Freeze** | SSH unresponsive, monitor frozen | Use `--oom-score-adj 1000` and monitor `systemd-oomd` [4]. |
| **Slow Training** | 50× slowdown in data loading | Ensure `pin_memory=True` and `non_blocking=True` [11]. |
| **Docker Restart Loop** | Container exits immediately after login | Check `docker logs` and increase `--shm-size` [3]. |
| **OOM Crash** | `RuntimeError: CUDA out of memory` | Reduce batch size to ≤6; pre-allocate model [4]. |

**Monitoring Commands:**
```bash
# Monitor container memory usage
docker stats autoresearch_training

# Check NCCL communication logs
tail -f /var/log/syslog | grep NCCL
```

## Sources
1.  **NVIDIA DGX Spark Official Product Page**: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
2.  **LMSYS Blog: NVIDIA DGX Spark In-Depth Review**: https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/
3.  **Daily.co: Training Smart Turn on NVIDIA DGX Spark**: https://www.daily.co/blog/training-smart-turn-on-the-nvidia-dgx-spark/
4.  **NVIDIA Developer Forums: DGX Spark becomes unresponsive ("zombie")**: https://forums.developer.nvidia.com/t/dgx-spark-becomes-unresponsive-zombie-instead-of-throwing-cuda-oom/353752
5.  **NVIDIA NCCL Playbook (Two Sparks)**: https://build.nvidia.com/spark/nccl
6.  **Stack Overflow: Docker shm-size**: https://stackoverflow.com/questions/30210362/how-to-increase-the-size-of-the-dev-shm-in-docker-container
7.  **NVIDIA Container Runtime Documentation**: https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html
8.  **karpathy/autoresearch GitHub**: https://github.com/karpathy/autoresearch
10. **NVIDIA Developer Forums: Training on DGX Spark**: https://forums.developer.nvidia.com/t/training-transformer-on-dgx-spark/347674
11. **NVIDIA Developer Forums: Pathological Slowdown**: https://forums.developer.nvidia.com/t/dgx-spark-arm64-cuda-13-pathological-slowdown-for-many-small-h2d-copies-from-pageable-cpu-memory-50x-vs-pinned-impacts-pytorch-model-load-patt/354506

---
**Deep Research Execution Plan Saved At:** `/home/sparky/.sparky/agent-workspace/docs/deep-research/plans/plan_20260308_172032.json`