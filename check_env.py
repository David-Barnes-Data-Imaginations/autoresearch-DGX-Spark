import torch
import subprocess
import sys

props = torch.cuda.get_device_properties(0)
total_gb = props.total_mem / 1e9
print(f"GPU: {props.name}")
print(f"Total memory: {total_gb:.0f} GB")

# Check what other processes are using
nvidia_smi = subprocess.check_output(
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader",
    shell=True
).decode().strip()
used_mb = int(nvidia_smi.split()[0])
avail = total_gb * 1024 - used_mb
print(f"Used by other procs: {used_mb/1024:.0f} GB")
print(f"Available for training: {avail/1024:.0f} GB")

# Check flash attention availability
try:
    import flash_attn
    print(f"flash_attn version: {flash_attn.__version__}")
except ImportError:
    print("flash_attn: NOT installed")

try:
    import triton
    print(f"triton version: {triton.__version__}")
except ImportError:
    print("triton: NOT installed")

# Check torch.compile support
print(f"torch.compile available: {hasattr(torch, 'compile')}")
print(f"CUDA arch: {props.major}.{props.minor}")

# Check for FP8 support
try:
    x = torch.empty(1, dtype=torch.float8_e4m3fn, device='cuda')
    print("FP8 e4m3fn: SUPPORTED")
except:
    print("FP8 e4m3fn: NOT supported")

try:
    x = torch.empty(1, dtype=torch.float8_e5m2, device='cuda')
    print("FP8 e5m2: SUPPORTED")
except:
    print("FP8 e5m2: NOT supported")

# Check uv
uv_path = subprocess.run("which uv", shell=True, capture_output=True, text=True)
print(f"uv: {'installed' if uv_path.returncode == 0 else 'NOT installed'}")
