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
import sys

# Add parent directory to path so we can import prepare.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    # --- RA-06: Parcae LTI Stability (paper Sec 4.1 / 4.2) ---
    parcae_e_norm: bool = True      # e = LN(P(s)): normalize injected prelude output (loss-spike fix)
    parcae_depth_sample: bool = True  # per-sequence Poisson depth sampling during training
    # --- RA-08: RoPE Loop-Index Embedding ---
    rope_loop: bool = False          # rotate full h by 2D complex RoPE over loop index t
    rope_loop_theta: float = 10000.0


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
# RA-08: RoPE loop-index embedding (2D complex rotary over recurrence depth)
# Paper: "A Mechanistic Analysis of Looped Reasoning Language Models"
# Rotates the FULL hidden state by loop iteration t so shared weights see a
# distinct phase each loop (extraction -> composition -> verification).
# Mitigation (plan): RMSNorm immediately after rotation to keep norms stable.
# ---------------------------------------------------------------------------

def apply_rope_loop_index(h: torch.Tensor, loop_t: int, theta: float = 10000.0) -> torch.Tensor:
    """Rotate h (B, T, D) by loop index t using a 2D complex RoPE over depth.

    Conjugate-pair layout: channels (2i, 2i+1) form complex pairs, so the
    rotated output is contiguous and norms are preserved exactly.
    """
    dim = h.shape[-1]
    half = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=h.device) / half))
    angle = loop_t * freqs  # (half,)
    x = h.float().reshape(*h.shape[:-1], half, 2)
    re, im = x[..., 0], x[..., 1]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    out_re = re * cos_a - im * sin_a
    out_im = re * sin_a + im * cos_a
    return torch.stack([out_re, out_im], dim=-1).reshape(h.shape).to(h.dtype)


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
        # Original init (rho(A) ~= 0.368). apply_parcae_init() can raise it to ~0.95.
        self.log_A = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    @torch.no_grad()
    def spectral_radius(self) -> float:
        """rho(A) = max_i |A_i| (diagonal A -> max abs of A). Logged each step (RA-06)."""
        return float(self.get_A().abs().max().item())

    @torch.no_grad()
    def apply_parcae_init(self):
        """Set log_A/log_dt so rho(A) ~= 0.95 (Parcae band top).
        rho = exp(-exp(log_dt+log_A)); solve log_dt+log_A = ln(-ln(0.95)) = -2.97."""
        self.log_A.data.fill_(0.03)
        self.log_dt.data.fill_(-3.0)

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
        # RA-08: RoPE loop-index rotation over full dim (optional)
        self.rope_loop = cfg.rope_loop
        self.rope_loop_theta = cfg.rope_loop_theta
        self.rope_loop_norm = RMSNorm(cfg.dim) if cfg.rope_loop else None

    def forward(self, h: torch.Tensor, e: torch.Tensor, freqs_cis: torch.Tensor, mask=None, n_loops=None, seq_depths=None) -> torch.Tensor:
        """n_loops: int (max loops, used at eval/inference).
        seq_depths: optional (B,) long tensor of per-sequence depths for this training
        step (Parcae per-sequence depth sampling). Sequence b updates only while t < seq_depths[b]."""
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        if seq_depths is not None:
            seq_depths = seq_depths.to(device=h.device, dtype=torch.long)
        else:
            seq_depths = torch.full((B,), n_loops, device=h.device, dtype=torch.long)

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)
        loops_used_sum = 0  # track average loop iterations for ACT metrics

        for t in range(n_loops):
            # A position runs this iteration only if not halted AND its sequence hasn't hit its depth
            still_running = ~halted & (seq_depths > t).unsqueeze(-1)  # (B, T)
            if self.rope_loop:
                # RA-08: full-dim 2D complex RoPE rotation by loop index t,
                # then RMSNorm (plan's failure mitigation for norm disruption).
                h_loop = self.rope_loop_norm(apply_rope_loop_index(h, t, self.rope_loop_theta))
            else:
                h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            trans_out = self.block(combined, freqs_cis, mask)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            p = self.act(h)  # (B, T)

            # ACT remainder trick (only for sequences still running)
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

            # RA-06 depth-completion term: a sequence reaching its sampled depth
            # before ACT-halting must still contribute its remaining probability to
            # h_out (the ACT trick sums to 1 only across the FULL depth). Without
            # this, short-depth sequences would emit ~zero output.
            finished_at_t = (seq_depths == (t + 1)).unsqueeze(-1) & still_running
            if finished_at_t.any():
                completion_mask = finished_at_t & ~halted  # exclude ACT-halted (already full-remainder)
                completion_weight = ((1.0 - cumulative_p).clamp(min=0)) * completion_mask.float()
                h_out = h_out + completion_weight.unsqueeze(-1) * h

            # Track how many positions used this loop iteration
            loops_used_sum += still_running.float().sum().item()

            if not (~halted & (seq_depths > (t + 1)).unsqueeze(-1)).any():
                break

        # ACT stats
        total_positions = B * T
        avg_loops = loops_used_sum / max(1, total_positions) if loops_used_sum > 0 else float(n_loops)
        self.last_avg_loops = avg_loops  # store for logging
        self.last_avg_depth = float(seq_depths.float().mean().item())  # RA-06: mean sampled depth
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
        self.e_inj_norm = RMSNorm(cfg.dim)  # Parcae (RA-06): normalize injected prelude output e
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

    def forward(self, input_ids: torch.Tensor, targets=None, reduction="mean", n_loops: int = None, seq_depths=None):
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        for layer in self.prelude:
            x = layer(x, freqs_cis, mask)

        e = x  # encoded input frozen for injection every loop
        if self.cfg.parcae_e_norm:
            # Parcae (paper Sec 4.1): e = LN(P(s)) stabilizes late-stage training
            e = self.e_inj_norm(e)
        # Parcae per-sequence depth sampling: only active during training; eval uses the
        # full n_loops so val_bpb stays comparable across configs.
        use_seq_depths = seq_depths if (seq_depths is not None and self.training and self.cfg.parcae_depth_sample) else None
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, seq_depths=use_seq_depths)

        for layer in self.coda:
            x = layer(x, freqs_cis, mask)

        logits = self.head(self.norm(x))

        if targets is not None:
            # Apply softcap for stable training (matching speedrun baseline)
            softcap = 10
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits

    def num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

# Training hyperparameters
SEQ_LEN = 512
MICRO_BATCH = 16
GRAD_ACCUM = 64
TOTAL_BATCH_SIZE = MICRO_BATCH * SEQ_LEN * GRAD_ACCUM  # 524,288 tokens per step
TIME_BUDGET = 86400  # 24 hours
DEVICE_BATCH_SIZE = MICRO_BATCH
EVAL_EVERY_N_STEPS = 200
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
# RA-06 experiment override layer (env vars; defaults preserve original 24h run)
# ---------------------------------------------------------------------------
def _env_float(key, default):
    v = os.environ.get(key)
    return float(v) if v not in (None, "") else default

def _env_int(key, default):
    v = os.environ.get(key)
    return int(float(v)) if v not in (None, "") else int(default)

def _env_bool(key, default):
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return v.lower() in ("1", "true", "yes", "on")

SEQ_LEN = _env_int("NANO_SEQ_LEN", SEQ_LEN)
MICRO_BATCH = _env_int("NANO_MICRO_BATCH", MICRO_BATCH)
GRAD_ACCUM = _env_int("NANO_GRAD_ACCUM", GRAD_ACCUM)
TOTAL_BATCH_SIZE = MICRO_BATCH * SEQ_LEN * GRAD_ACCUM  # recomputed
DEVICE_BATCH_SIZE = _env_int("NANO_DEVICE_BATCH", MICRO_BATCH)
TIME_BUDGET = _env_int("NANO_TIME_BUDGET", TIME_BUDGET)
EVAL_EVERY_N_STEPS = _env_int("NANO_EVAL_EVERY", EVAL_EVERY_N_STEPS)
LOG_EVERY_N_STEPS = _env_int("NANO_LOG_EVERY", LOG_EVERY_N_STEPS)
LEARNING_RATE = _env_float("NANO_LEARNING_RATE", LEARNING_RATE)
WARMUP_STEPS = _env_int("NANO_WARMUP_STEPS", WARMUP_STEPS)
GRAD_CLIP = _env_float("NANO_GRAD_CLIP", GRAD_CLIP)
MAX_STEPS = _env_int("NANO_MAX_STEPS", 0)  # 0 = run to TIME_BUDGET; else hard stop
MAX_LOOP_ITERS = _env_int("NANO_MAX_LOOP_ITERS", 8)
PARCAE_E_NORM = _env_bool("NANO_PARCAE_E_NORM", True)
PARCAE_DEPTH_SAMPLE = _env_bool("NANO_PARCAE_DEPTH_SAMPLE", True)
PARCAE_INIT = _env_bool("NANO_PARCAE_INIT", True)  # raise rho(A) init to ~0.95 (Parcae band top)
ROPE_LOOP = _env_bool("NANO_ROPE_LOOP", False)  # RA-08: RoPE loop-index rotation
ROPE_LOOP_THETA = _env_float("NANO_ROPE_LOOP_THETA", 10000.0)
CKPT_TAG = os.environ.get("NANO_CKPT_TAG", "final")
# Estimate total steps from the time budget so the LR schedule decays to ~0 by
# the end of a short run (the original hardcoded 999999 for the 24h run).
STEP_TIME_EST = 8.0  # seconds per step (recurrent block, seq 512)
EXPECTED_STEPS = max(50, int(TIME_BUDGET / STEP_TIME_EST))
if MAX_STEPS > 0:
    EXPECTED_STEPS = max(50, MAX_STEPS)  # LR schedule targets the hard stop
print(f"[RA-06 OVERRIDES] seq={SEQ_LEN} mb={MICRO_BATCH} ga={GRAD_ACCUM} "
      f"time={TIME_BUDGET}s eval_every={EVAL_EVERY_N_STEPS} lr={LEARNING_RATE} "
      f"loops={MAX_LOOP_ITERS} e_norm={PARCAE_E_NORM} depth_sample={PARCAE_DEPTH_SAMPLE} "
      f"rope_loop={ROPE_LOOP} rope_theta={ROPE_LOOP_THETA} tag={CKPT_TAG}")


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
# The model's max_seq_len must be >= MAX_SEQ_LEN (2048) because evaluate_bpb
# uses MAX_SEQ_LEN for the eval dataloader. Training uses SEQ_LEN=512.
cfg = NanoMythosConfig(
    vocab_size=vocab_size,
    dim=256,
    n_heads=8,
    n_kv_heads=4,
    max_seq_len=MAX_SEQ_LEN,  # must cover eval seq_len (2048 from prepare.py)
    max_loop_iters=MAX_LOOP_ITERS,
    prelude_layers=2,
    coda_layers=2,
    rope_theta=10000.0,
    act_threshold=0.99,
    lora_rank=4,
    dropout=0.0,
    parcae_e_norm=PARCAE_E_NORM,
    parcae_depth_sample=PARCAE_DEPTH_SAMPLE,
    rope_loop=ROPE_LOOP,
    rope_loop_theta=ROPE_LOOP_THETA,
)

with torch.device("meta"):
    model = NanoMythos(cfg)
model.to_empty(device=device)
# Re-initialize the freqs_cis buffer on the correct device (to_empty leaves it uninitialized)
freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta).to(device=device)
model.register_buffer("freqs_cis", freqs, persistent=False)
model._init_weights()
# RA-06: raise the LTI injection init so rho(A) ~= 0.95 (Parcae band top).
if PARCAE_INIT and hasattr(model.recurrent, "injection"):
    model.recurrent.injection.apply_parcae_init()
    print(f"[RA-06] Parcae LTI init applied: rho(A) = {model.recurrent.injection.spectral_radius():.4f}")
else:
    _rho0 = model.recurrent.injection.spectral_radius() if hasattr(model.recurrent, "injection") else float("nan")
    print(f"[RA-06] original LTI init (no Parcae init): rho(A) = {_rho0:.4f}")

num_params = model.num_params()
print(f"Model: NanoMythos")
print(f"Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
print(f"Config: {asdict(cfg)}")

# Estimate FLOPs per token (rough: 6 * params + attention)
# For recurrent models, the recurrent block runs n_loops times, so multiply by that factor
# Only the recurrent block params are counted multiple times
recurrent_params = sum(p.numel() for p in model.recurrent.parameters() if p.requires_grad)
non_recurrent_params = num_params - recurrent_params
num_flops_per_token = 6 * (non_recurrent_params + recurrent_params * cfg.max_loop_iters)
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * SEQ_LEN
grad_accum_steps = GRAD_ACCUM

print(f"Seq len: {SEQ_LEN}")
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
    tokenizer, DEVICE_BATCH_SIZE, SEQ_LEN, "train", pin_memory=True
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
# Run until time budget expires (steps estimated from throughput)
print(f"\nStarting training loop...")

gc.collect()
gc.freeze()
gc.disable()

print(f"\nStarting training loop...")
print(f"Expected steps: ~{TIME_BUDGET // 20}s / ~20s per step ≈ {TIME_BUDGET // 20} steps")
print(f"Expected total tokens: ~{TIME_BUDGET * 25000 / 1e9:.1f}B tokens\n")

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    # RA-06: per-sequence depth sampling (Poisson, mu = max_loop_iters).
    # Only during training; each sequence in the micro-batch gets its own loop depth.
    if cfg.parcae_depth_sample:
        _seq_depths = torch.poisson(cfg.max_loop_iters * torch.ones(x.shape[0])).long().clamp(min=1)
    else:
        _seq_depths = None
    for micro_step in range(grad_accum_steps):
        try:
            with autocast_ctx:
                loss = model(x, targets=y, reduction="mean", n_loops=cfg.max_loop_iters, seq_depths=_seq_depths)
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
    lr = get_lr(step, EXPECTED_STEPS)
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
    avg_loops = getattr(model.recurrent, 'last_avg_loops', cfg.max_loop_iters)
    avg_depth = getattr(model.recurrent, 'last_avg_depth', cfg.max_loop_iters)
    # RA-06: log spectral radius rho(A) of the LTI injection matrix.
    _lti = model.recurrent.injection if hasattr(model.recurrent, 'injection') else None
    rho_A = _lti.spectral_radius() if (_lti is not None and hasattr(_lti, 'spectral_radius')) else float('nan')

    print(
        f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {lr:.6f} | "
        f"dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | mfu: {mfu:.1f}% | "
        f"avg_loops: {avg_loops:.1f}/{cfg.max_loop_iters} | avg_depth: {avg_depth:.1f} | rhoA: {rho_A:.3f} | "
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

    # RA-06: hard step stop (for fixed-step head-to-head ablations)
    if MAX_STEPS > 0 and step >= MAX_STEPS:
        break

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
checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", f"nano_mythos_{CKPT_TAG}.pt")
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "config": asdict(cfg),
    "step": step,
    "val_bpb": val_bpb,
    "total_tokens": total_tokens,
    "avg_loops": getattr(model.recurrent, 'last_avg_loops', cfg.max_loop_iters),
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
print(f"avg_loops:        {getattr(model.recurrent, 'last_avg_loops', cfg.max_loop_iters):.2f}/{cfg.max_loop_iters}")