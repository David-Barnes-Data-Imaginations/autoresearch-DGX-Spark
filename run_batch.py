"""
In-process batch experiment runner.
Imports train.py's model/optimizer code directly and runs multiple configs
without subprocess overhead. Much more reliable output capture.
"""
import os, sys, gc, time, math, re
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# Import prepare.py utilities
sys.path.insert(0, "/home/david-barnes/autoresearch-DGX-Spark")
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---- Load train.py code ----
TRAIN_CODE_PATH = "/home/david-barnes/autoresearch-DGX-Spark/train.py"

def load_train_module(patch_old=None, patch_new=None):
    """Load train.py, optionally patch, and extract just the classes/functions."""
    with open(TRAIN_CODE_PATH, "r") as f:
        code = f.read()

    if patch_old and patch_new:
        if patch_old not in code:
            raise ValueError(f"Patch target not found: {patch_old[:60]}")
        code = code.replace(patch_old, patch_new, 1)

    ns = {}
    exec(code, ns)
    return ns

# Load base namespace once
print("Loading base model code...")
ns = load_train_module()

GPTConfig = ns["GPTConfig"]
GPT = ns["GPT"]
MuonAdamW = ns["MuonAdamW"]
flash_attn_func = ns["flash_attn_func"]

# Experiments: (name, patch_old, patch_new, notes)
EXPERIMENTS = [
    ("baseline", None, None, "DEPTH=4 AR=64 SCALAR_LR=0.5 baseline confirm"),
    ("scalar_lr_0.8", "SCALAR_LR = 0.5", "SCALAR_LR = 0.8", "SCALAR_LR sweep: 0.5->0.8"),
    ("scalar_lr_1.0", "SCALAR_LR = 0.5", "SCALAR_LR = 1.0", "SCALAR_LR sweep: 0.5->1.0"),
    ("scalar_lr_1.2", "SCALAR_LR = 0.5", "SCALAR_LR = 1.2", "SCALAR_LR sweep: 0.5->1.2"),
    ("scalar_lr_1.5", "SCALAR_LR = 0.5", "SCALAR_LR = 1.5", "SCALAR_LR sweep: 0.5->1.5"),
    ("matrix_lr_0.05", "MATRIX_LR = 0.04", "MATRIX_LR = 0.05", "MATRIX_LR sweep: 0.04->0.05"),
    ("matrix_lr_0.06", "MATRIX_LR = 0.04", "MATRIX_LR = 0.06", "MATRIX_LR sweep: 0.04->0.06"),
    ("weight_decay_0.05", "WEIGHT_DECAY = 0.1", "WEIGHT_DECAY = 0.05", "WD halved: 0.1->0.05"),
    ("adam_b1_0.85", "ADAM_BETAS = (0.8, 0.95)", "ADAM_BETAS = (0.85, 0.95)", "Adam beta1: 0.8->0.85"),
    ("warmup_0.02", "WARMUP_RATIO = 0.0", "WARMUP_RATIO = 0.02", "Small warmup 0.02"),
    ("warmup_0.05", "WARMUP_RATIO = 0.0", "WARMUP_RATIO = 0.05", "Small warmup 0.05"),
]

BASELINES = {
    "ASPECT_RATIO": 64, "HEAD_DIM": 128, "WINDOW_PATTERN": "SSSL",
    "TOTAL_BATCH_SIZE": 2**19, "EMBEDDING_LR": 0.6, "UNEMBEDDING_LR": 0.004,
    "MATRIX_LR": 0.04, "SCALAR_LR": 0.5, "WEIGHT_DECAY": 0.1,
    "ADAM_BETAS": (0.8, 0.95), "WARMUP_RATIO": 0.0, "WARMDOWN_RATIO": 0.1,
    "FINAL_LR_FRAC": 0.0, "DEPTH": 4, "DEVICE_BATCH_SIZE": 8,
}

def get_hp(name, patch_old, patch_new):
    """Extract hyperparams from patched train.py namespace."""
    with open(TRAIN_CODE_PATH, "r") as f:
        code = f.read()
    if patch_old and patch_new:
        code = code.replace(patch_old, patch_new, 1)

    hp = dict(BASELINES)
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        for key in ["ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN", "TOTAL_BATCH_SIZE",
                     "EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR",
                     "WEIGHT_DECAY", "ADAM_BETAS", "WARMUP_RATIO", "WARMDOWN_RATIO",
                     "FINAL_LR_FRAC", "DEPTH", "DEVICE_BATCH_SIZE"]:
            if line.startswith(key + " =") or line.startswith(key + " ="):
                try:
                    val = eval(line.split("=", 1)[1].split("#")[0].strip())
                    hp[key] = val
                except:
                    pass
    return hp

def run_experiment(name, patch_old, patch_new, notes):
    """Build and train a model with the given config."""
    hp = get_hp(name, patch_old, patch_new)

    depth = hp["DEPTH"]
    aspect = hp["ASPECT_RATIO"]
    head_dim = hp["HEAD_DIM"]
    total_bs = hp["TOTAL_BATCH_SIZE"]
    device_bs = hp["DEVICE_BATCH_SIZE"]

    # Build model config
    base_dim = depth * aspect
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads,
        n_embd=model_dim, window_pattern=hp["WINDOW_PATTERN"],
    )

    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    tokens_per_fwdbwd = device_bs * MAX_SEQ_LEN
    grad_accum_steps = total_bs // tokens_per_fwdbwd

    optimizer = model.setup_optimizer(
        unembedding_lr=hp["UNEMBEDDING_LR"],
        embedding_lr=hp["EMBEDDING_LR"],
        scalar_lr=hp["SCALAR_LR"],
        adam_betas=hp["ADAM_BETAS"],
        matrix_lr=hp["MATRIX_LR"],
        weight_decay=hp["WEIGHT_DECAY"],
    )

    model = torch.compile(model, dynamic=False)

    # Fresh dataloader for each experiment (same data, but reset state)
    train_loader = make_dataloader(tokenizer, device_bs, MAX_SEQ_LEN, "train", pin_memory=True)
    x, y, epoch = next(train_loader)

    # Schedules
    def get_lr_multiplier(progress):
        wr = hp["WARMUP_RATIO"]
        if progress < wr:
            return progress / wr if wr > 0 else 1.0
        elif progress < 1.0 - hp["WARMDOWN_RATIO"]:
            return 1.0
        else:
            cooldown = (1.0 - progress) / hp["WARMDOWN_RATIO"]
            return cooldown * 1.0 + (1 - cooldown) * hp["FINAL_LR_FRAC"]

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return hp["WEIGHT_DECAY"] * (1 - progress)

    # Training loop
    t0_train = time.time()
    smooth_loss = 0
    total_t = 0
    step = 0

    print(f"  Params: {num_params:,} | Steps budget: ~{TIME_BUDGET/3:.0f} | GAS: {grad_accum_steps}", flush=True)

    gc.collect(); gc.freeze(); gc.disable()

    while True:
        torch.cuda.synchronize()
        t_step = time.time()

        for micro_step in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        progress = min(total_t / TIME_BUDGET, 1.0)
        lrm = get_lr_multiplier(progress)
        mm = get_muon_momentum(step)
        wd = get_weight_decay(progress)

        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = mm
                group["weight_decay"] = wd
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        if train_loss_f > 100:
            print(f"  FAIL: loss exploded at step {step}")
            return None

        torch.cuda.synchronize()
        dt = time.time() - t_step
        if step > 10:
            total_t += dt

        ema_beta = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * train_loss_f
        debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
        tok_s = int(total_bs / dt)

        print(f"\r  step {step:05d} ({progress*100:.1f}%) | loss: {debiased_loss:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/s: {tok_s:,}", end="", flush=True)
        step += 1

        if step > 10 and total_t >= TIME_BUDGET:
            break

    print()  # newline
    total_tokens = step * total_bs

    # Eval
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, device_bs)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    result = {
        "name": name, "status": "OK",
        "val_bpb": f"{val_bpb:.6f}", "steps": str(step),
        "training_s": f"{total_t:.1f}", "vram_mb": f"{peak_vram_mb:.0f}",
        "tok_s": str(tok_s), "notes": notes,
    }
    print(f"  -> val_bpb={val_bpb:.6f} steps={step} time={total_t:.1f}s tok/s={tok_s:,} vram={peak_vram_mb:.0f}MB")
    return result

def main():
    print("=" * 120)
    print("In-Process Batch Experiment Runner")
    print("=" * 120)

    baseline_val = 1.865391
    results = []
    t_total = time.time()

    for name, patch_old, patch_new, notes in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Experiment: {name} | {notes}")
        print(f"{'='*60}")
        try:
            r = run_experiment(name, patch_old, patch_new, notes)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": name, "status": f"ERROR: {e}", "notes": notes})

    print(f"\n{'='*120}")
    print(f"Total wall time: {time.time() - t_total:.0f}s")
    print(f"\n{'='*120}")
    print(f"{'Name':<25} {'val_bpb':<10} {'Steps':<6} {'Time(s)':<8} {'tok/s':<10} {'VRAM_MB':<8} {'diff':<8} {'Notes'}")
    print("-" * 120)

    ok = [r for r in results if r["status"] == "OK"]
    for r in sorted(ok, key=lambda r: float(r["val_bpb"])):
        diff = float(r["val_bpb"]) - baseline_val
        marker = "Y" if diff < 0 else " "
        print(f"  {marker} {r['name']:<23} {r['val_bpb']:<10} {r['steps']:<6} {r['training_s']:<8} {r['tok_s']:<10} {r['vram_mb']:<8} {diff:+.4f}   {r['notes']}")

    if ok:
        best = min(ok, key=lambda r: float(r["val_bpb"]))
        print(f"\nBest: {best['name']} val_bpb={best['val_bpb']} (baseline={baseline_val})")

    # TSV
    print("\n---TSV---")
    print("name\tval_bpb\tsteps\ttraining_s\ttok_s\tvram_mb\tnotes")
    for r in results:
        if r["status"] == "OK":
            print(f"{r['name']}\t{r['val_bpb']}\t{r['steps']}\t{r['training_s']}\t{r['tok_s']}\t{r['vram_mb']}\t{r['notes']}")

if __name__ == "__main__":
    main()
