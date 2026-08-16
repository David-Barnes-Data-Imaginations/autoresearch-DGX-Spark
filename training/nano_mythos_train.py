"""
Nano-Mythos: Minimal RDT (Recurrent-Depth Transformer) for architecture validation.

A ~50M parameter RDT model to pre-train for ~24 hours on DGX Spark, validating:
1. The recurrent loop improves convergence vs. a standard transformer
2. ACT halting converges (easy positions stop early)
3. LTI-stable injection prevents hidden state explosion across loops
4. Loss curve shape matches theoretical expectations

Uses the same architectural components as OpenMythos but scaled down:
- GQA attention (simpler than MLA for small models)
- Dense SwiGLU FFN (no MoE — too complex for 50M params)
- LTI-stable injection (spectral radius < 1 by construction)
- ACT halting (adaptive computation time)
- Loop-index RoPE embedding
- Depth-wise LoRA adapter

Usage:
    python training/nano_mythos_train.py
"""
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import gc
import time
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Nano-Mythos Configuration
# ---------------------------------------------------------------------------

@dataclass
class NanoMythosConfig:
    """Minimal RDT config for ~50M parameter architecture validation."""
    vocab_size: int = 8192          # matched to prepare.py tokenizer
    dim: int = 256                  # model hidden dimension
    n_heads: int = 8                # attention heads
    n_kv_heads: int = 4             # GQA: fewer KV heads
    max_seq_len: int = 512          # context length (reduced from 2048)
    max_loop_iters: int = 8         # T — recurrent loop depth
    prelude_layers: int = 2         # standard transformer blocks before loop
    coda_layers: int = 2            # standard transformer blocks after loop
    rope_theta: float = 10000.0
    act_threshold: float = 0.99     # ACT halting threshold
    lora_rank: int = 4              # depth-wise LoRA rank
    dropout: float = 0.0            # no dropout for pretraining


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex-valued RoPE rotation matrices."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to query or key tensors."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# ---------------------------------------------------------------------------
# GQA Attention (simplified — no KV cache for training)
# ---------------------------------------------------------------------------

class GQAttention(nn.Module):
    """Grouped Query Attention with Flash Attention / SDPA fallback."""
    def __init__(self, cfg: NanoMythosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # Expand KV heads for GQA
        k = k.repeat_interleave(self.groups, dim=2)
        v = v.repeat_interleave(self.groups, dim=2)

        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SwiGLU FFN (dense, no MoE for small model)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: output = down(silu(gate(x)) * up(x))."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Standard Transformer Block (for prelude/coda)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GQA attention and SwiGLU FFN."""
    def __init__(self, cfg: NanoMythosConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = GQAttention(cfg)
        self.ffn = SwiGLU(cfg.dim, cfg.dim * 4)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + self.resid_drop(self.attn(self.attn_norm(x), freqs_cis, mask))
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Loop-index RoPE embedding
# ---------------------------------------------------------------------------

def loop_index_embedding(h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Inject sinusoidal loop-index signal into the first loop_dim channels of h."""
    freqs = 1.0 / (theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim))
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """Depth-wise LoRA: shared A/B matrices with per-loop scale vector."""
    def __init__(self, dim: int, rank: int, max_loops: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)
        self.scale = nn.Embedding(max_loops, rank)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))
        down = self.down(x) * s
        return down @ self.B


# ---------------------------------------------------------------------------
# LTI-stable injection (spectral radius < 1 by construction)
# ---------------------------------------------------------------------------

class LTIInjection(nn.Module):
    """
    Stable input injection for recurrent update:
        h_{t+1} = A·h_t + B·e + Transformer(h_t, e)

    Guarantees ρ(A) < 1 via ZOH discretization:
        A_continuous = Diag(-exp(log_A))
        A_discrete   = exp(Δt · A_continuous) = exp(-exp(log_dt + log_A))
    All values strictly in (0, 1).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT halting
# ---------------------------------------------------------------------------

class ACTHalting(nn.Module):
    """Adaptive Computation Time halting mechanism."""
    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block (one set of weights, looped T times)
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    """
    Core recurrent block — a single TransformerBlock looped T times.

    At each loop iteration t:
        1. loop_index_embedding: inject sinusoidal loop-index signal into h
        2. TransformerBlock:    compute attention + FFN on normalized (h + e)
        3. LoRAAdapter:         apply depth-wise LoRA delta
        4. LTIInjection:        stable update h = A·h + B·e + transformer_out
        5. ACTHalting:          accumulate per-position halting probabilities
    """
    def __init__(self, cfg: NanoMythosConfig):
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = cfg.dim // 8  # fraction of channels receiving loop-index embedding

    def forward(self, h: torch.Tensor, e: torch.Tensor, freqs_cis: torch.Tensor, mask=None, n_loops=None) -> torch.Tensor:
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            trans_out = self.block(combined, freqs_cis, mask)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            p = self.act(h)  # (B, T)
            still_running = ~halted

            # ACT remainder trick
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            if halted.all():
                break

        return h_out


# ---------------------------------------------------------------------------
# Full Nano-Mythos Model
# ---------------------------------------------------------------------------

class NanoMythos(nn.Module):
    """
    Nano-Mythos: Minimal Recurrent-Depth Transformer.

    Architecture: Input → Prelude (2 blocks) → Recurrent Block (T=8 loops) → Coda (2 blocks) → Output
    """
    def __init__(self, cfg: NanoMythosConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs)

        self.prelude = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.prelude_layers)])
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.coda_layers)])

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(self, input_ids: torch.Tensor, n_loops: int = None) -> torch.Tensor:
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        for layer in self.prelude:
            x = layer(x, freqs_cis, mask)

        e = x  # encoded input frozen for injection every loop
        x = self.recurrent(x, e, freqs_cis, mask, n_loops)

        for layer in self.coda:
            x = layer(x, freqs_cis, mask)

        return self.head(self.norm(x))

    def num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

# Model architecture
NANO_CFG = NanoMythosConfig(
    vocab_size=8192,
    dim=256,
    n_heads=8,
    n_kv_heads=4,
    max_seq_len=512,
    max_loop_iters=8,
    prelude_layers=2,
    coda_layers=2,
    rope_theta=10000.0,
    act_threshold=0.99,
    lora_rank=4,
    dropout=0.0,
)

# Training hyperparameters
SEQ_LEN = 512
MICRO_BATCH = 16
GRAD_ACCUM = 64
TOTAL_BATCH_SIZE = MICRO_BATCH * SEQ_LEN * GRAD_ACCUM  # 524,288 tokens per step
TIME_BUDGET = 86400  # 24 hours
DEVICE_BATCH_SIZE = MICRO_BATCH
EVAL_EVERY_N_STEPS = 500
LOG_EVERY_N_STEPS = 10

# Optimizer
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.95
WARMUP_STEPS = 100
GRAD_CLIP = 1.0

# GB10 (RTX 5070-class) peak BF16 FLOPS
GB10_BF16_PEAK_FLOPS = 40e12


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# Use MAX_SEQ_LEN from prepare.py for evaluation, but our model uses seq_len=512
# Override the model's max_seq_len to match
cfg = NanoMythosConfig(
    vocab_size=vocab_size,
    dim=256,
    n_heads=8,
    n_kv_heads=4,
    max_seq_len=MAX_SEQ_LEN,  # use the prepare.py MAX_SEQ_LEN for eval compatibility
    max_loop_iters=8,
    prelude_layers=2,
    coda_layers=2,
    rope_theta=10000.0,
    act_threshold=0.99,
    lora_rank=4,
    dropout=0.0,
)

with torch.device("meta"):
    model = NanoMythos(cfg)
model.to_empty(device=device)
model._init_weights()

num_params = model.num_params()
print(f"Model: NanoMythos")
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
print(f"Config: {asdict(cfg)}")

# Estimate FLOPs per token (rough: 6 * params + attention)
num_flops_per_token = 6 * num_params  # simplified estimate
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
grad_accum_steps = GRAD_ACCUM

print(f"Seq len: {MAX_SEQ_LEN}")
print(f"Micro batch: {DEVICE_BATCH_SIZE}")
print(f"Grad accum: {grad_accum_steps}")
print(f"Total batch: {TOTAL_BATCH_SIZE:,} tokens")
print(f"Time budget: {TIME_BUDGET}s ({TIME_BUDGET/3600:.0f}h)")
print(f"Steps per epoch: ~{tokens_per_fwdbwd * grad_accum_steps / 1e6:.1f}M tokens/step")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2),
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,
)

model = torch.compile(model, dynamic=False)

# Dataloader
train_loader = make_dataloader(
    tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", pin_memory=True
)
x, y, epoch = next(train_loader)


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0
total_steps = TIME_BUDGET // (MAX_SEQ_LEN * DEVICE_BATCH_SIZE * grad_accum_steps * 0.01)  # rough estimate
# Better: estimate steps from time budget
# Each step takes roughly: (MAX_SEQ_LEN * DEVICE_BATCH_SIZE * grad_accum_steps) / throughput
# We'll just run until time budget expires

gc.collect()
gc.freeze()
gc.disable()

print(f"\nStarting training loop...")
print(f"Expected steps: ~{TIME_BUDGET // 100}s / ~100s per step ≈ {TIME_BUDGET // 100} steps")
print(f"Expected total tokens: ~{TIME_BUDGET * 250000 / 1e9:.1f}B tokens\n")

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        try:
            with autocast_ctx:
                logits = model(x, n_loops=cfg.max_loop_iters)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1,
                )
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nOOM detected at step {step}. Reducing batch size.")
                DEVICE_BATCH_SIZE = max(1, DEVICE_BATCH_SIZE // 2)
                raise SystemExit(f"Reduce DEVICE_BATCH_SIZE to {DEVICE_BATCH_SIZE} and restart")
            raise

    # LR schedule
    lr = get_lr(step, 999999)
    for group in optimizer.param_groups:
        group["lr"] = lr

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail
    if train_loss_f > 100:
        print("FAIL: loss exploded")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * total_training_time / TIME_BUDGET
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / GB10_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {lr:.6f} | "
        f"dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}% | "
        f"epoch: {epoch} | remaining: {remaining:.0f}s",
        end="",
        flush=True,
    )

    # Periodic eval
    if step > 0 and step % EVAL_EVERY_N_STEPS == 0:
        model.eval()
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)
        model.train()
        print(f"\n  [EVAL] step {step} | val_bpb: {val_bpb:.6f} | train_loss: {debiased_smooth_loss:.6f}")

    step += 1

    # Time's up
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Save checkpoint
checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "nano_mythos_final.pt")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "config": asdict(cfg),
    "step": step,
    "val_bpb": val_bpb,
    "total_tokens": total_tokens,
}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = (
    100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10)
    / total_training_time / GB10_BF16_PEAK_FLOPS
    if total_training_time > 0 else 0
)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"model_dim:        {cfg.dim}")
print(f"max_loop_iters:   {cfg.max_loop_iters}")
print(f"act_threshold:    {cfg.act_threshold}")