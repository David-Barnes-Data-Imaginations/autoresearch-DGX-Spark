# Autonomous Research Plan: Advanced Recurrent-Depth & Looped Transformer Architectures

**Target System**: Local Hermes AI Research Assistant (`Qwen 3.8 - 27B` @ FP8)
**Execution Environment**: NVIDIA DGX Spark (GB10 / SM_12.1, 128GB Unified Memory, PyTorch, bfloat16 mixed precision)
**Base Repository**: [`autoresearch-OpenMythos-KellerJordanSpeedrun`](file:///c:/Users/david/Documents/recursion)
**Objective**: Systematically evaluate, implement, and benchmark 12 novel research avenues derived from state-of-the-art 2025–2026 Recurrent-Depth, Looped Transformer, and Latent Reasoning literature.

---

## Executive Summary & Workflow Protocol

Your goal is to iteratively modify the codebase in `open_mythos/` and `training/`, run training experiments via `training/exp_train.py` or `training/dgx_spark_train.py`, record quantitative metrics, and validate architectural hypotheses.

### System Execution Rules for Hermes
1. **OOM & Unified Memory Safety**: Always enforce `NCCL_P2P_DISABLE=1` and keep per-GPU micro-batch sizes bounded ($\le 8$ for sequence length 2048/4096).
2. **Spectral Radius Enforcement**: In all looped variants, log spectral radius $\rho(A) = \max|\lambda_i(A)|$ after every epoch. If $\rho(A) \ge 1.0$, abort run and apply LTI log-space clamping.
3. **Automated Verification Command**:
   ```bash
   python training/smoke_test_train.py --steps 50 --batch_size 4
   ```
4. **Baseline Metrics (OpenMythos 3B Baseline)**:
   - **Validation Loss Target**: $\le 2.45$ on `FineWeb-Edu` (10B token sample).
   - **Training Throughput**: $\ge 12,500$ tokens/sec per GPU.
   - **Peak Memory**: $\le 42$ GB.

---

## Research Avenue 1: Mixture-of-Recursions (MoR) — Dynamic Token-Level Recurrent Depths & Selective KV-Caching

- **Paper Reference**: *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation* ([arXiv:2507.10524](https://arxiv.org/pdf/2507.10524))
- **Core Hypothesis & Mechanism**: Static loop counts ($T$ iterations for all tokens) waste compute on easy tokens and under-compute hard tokens. MoR introduces a lightweight `TokenDepthRouter` that routes individual tokens to specific target recursion depths $T_i \in \{1, 2, \dots, T_{\max}\}$. Tokens that reach their assigned depth halt early and stop appending to the KV cache, reducing both FLOPs and memory footprint.

### Implementation Logic
1. **Module Creation**: Create `open_mythos/mor.py` containing `TokenDepthRouter`.
2. **Router Architecture**:
   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F

   class TokenDepthRouter(nn.Module):
       def __init__(self, dim: int, max_depths: int = 4):
           super().__init__()
           self.proj = nn.Linear(dim, max_depths)
           self.max_depths = max_depths

       def forward(self, x: torch.Tensor):
           # x: (B, T, dim) -> logits: (B, T, max_depths)
           logits = self.proj(x)
           probs = F.softmax(logits, dim=-1)
           # Sample depth assignment (or argmax during inference)
           depth_indices = torch.argmax(probs, dim=-1) + 1 # 1 to max_depths
           return depth_indices, probs
   ```
3. **Integration into `RecurrentBlock` (`open_mythos/main.py`)**:
   - Before the recurrence loop, calculate `target_depths` per token from the Prelude output $e$.
   - At loop iteration $t$, construct a mask `active_mask = (target_depths >= t)`.
   - Update only active tokens: `h[active_mask] = A * h[active_mask] + B * e[active_mask] + block_out[active_mask]`.
   - Freeze KV cache entries for inactive tokens after their designated depth step.

### Hyperparameters
- `max_loop_iters`: 16
- `router_depth_choices`: `[4, 8, 12, 16]`
- `aux_depth_loss_weight`: $0.01$ (penalizes high average token depth)

### Validation Criteria
- **Validation Loss**: Equal or lower loss ($\Delta \text{Val Loss} \le 0.00$) compared to uniform $T=16$ baseline.
- **Compute Efficiency**: $\ge 35\%$ reduction in total FLOPs per batch.
- **Throughput Gain**: $\ge 1.4\times$ speedup in token generation throughput.

### Automated Failure Mitigation
- *Issue*: Router collapses to assigning all tokens to minimum depth $T=1$.
- *Fix*: Add load-balancing entropy loss: $\mathcal{L}_{\text{balance}} = \sum_{k} (p_k - \frac{1}{K})^2$ where $p_k$ is the batch fraction assigned to depth $k$.

---

## Research Avenue 2: Hyperloop Transformers — Matrix-Valued Residual Streams & Multi-State Hyper-Connections

- **Paper Reference**: *Hyperloop Transformers* ([arXiv:2604.21254](https://arxiv.org/abs/2604.21254))
- **Core Hypothesis & Mechanism**: Standard residual streams pass a 1D vector $h_t \in \mathbb{R}^D$ across loops. Hyperloop expands this into a multi-stream matrix state $H_t \in \mathbb{R}^{K \times D}$ (where $K$ is the number of hyper-streams, e.g., $K=4$). Inter-loop hyper-connections learn dynamic linear combinations of previous loop states $\{H_0, H_1, \dots, H_{t-1}\}$, allowing complex representation trajectories without adding heavy layer parameters.

### Implementation Logic
1. **Module Creation**: In `open_mythos/main.py`, create `HyperConnectionBlock`.
2. **Matrix State Representation**:
   - Initialize $H_0 \in \mathbb{R}^{B \times T \times K \times D}$ by projecting Prelude output $e$ across $K$ streams: $H_0 = \text{stack}([W_k e]_{k=1}^K)$.
3. **Hyper-Connection Recurrence**:
   $$\tilde{H}_{t} = \sum_{j=0}^{t-1} \alpha_{j,t} H_j + \text{TransformerBlock}\left(\sum_{k=1}^K \beta_k H_{t-1, k}\right)$$
   ```python
   class HyperConnectionBlock(nn.Module):
       def __init__(self, dim: int, num_streams: int = 4):
           super().__init__()
           self.num_streams = num_streams
           self.mix_weights = nn.Parameter(torch.randn(num_streams, num_streams) * 0.02)
           self.stream_proj = nn.Linear(dim * num_streams, dim)

       def forward(self, stream_states: torch.Tensor, block_out: torch.Tensor):
           # stream_states: (B, T, K, D)
           # Mix streams across K dimension
           mixed = torch.einsum('btkd,kl->btld', stream_states, self.mix_weights)
           # Add block output to primary stream
           mixed[:, :, 0, :] += block_out
           return mixed
   ```

### Hyperparameters
- `num_streams (K)`: 4
- `dim`: 2048
- `max_loop_iters`: 12

### Validation Criteria
- **Model Quality**: Achieve identical validation loss to a $2\times$ parameter non-looped transformer.
- **Parameter Overhead**: Parameter increase must be $< 3\%$ compared to standard OpenMythos.
- **Quantization Resilience**: Post-training FP8/INT4 quantization accuracy degradation $< 0.5\%$.

### Automated Failure Mitigation
- *Issue*: Multi-stream hidden states explode across loop iterations.
- *Fix*: Apply LayerNorm over stream dimension $K$ after each hyper-connection update: `mixed = RMSNorm(dim)(mixed)`.

---

## Research Avenue 3: LT2 Linear-Time Looped Transformers — Hybrid Subquadratic Attention in Continuous Latent Space

- **Paper Reference**: *LT2: Linear-Time Looped Transformers* ([arXiv:2605.20670](https://arxiv.org/pdf/2605.20670))
- **Core Hypothesis & Mechanism**: Standard softmax attention inside a looped transformer incurs quadratic time complexity $O(T \cdot N^2)$ over sequence length $N$ and loop depth $T$. LT2 replaces quadratic attention inside the loop with an interleaved hybrid of Linear Gated Attention (GDN) and Dynamic Sparse Attention (DSA), maintaining $O(N)$ linear time per loop iteration.

### Implementation Logic
1. **Module Creation**: Create `open_mythos/lt2_attention.py`.
2. **Linear Gated Attention (GDN)**:
   $$Q = W_Q x, \quad K = W_K x, \quad V = W_V x, \quad G = \text{sigmoid}(W_G x)$$
   $$S_t = S_{t-1} + K_t^\top V_t, \quad O_t = G_t \odot (Q_t S_t)$$
3. **Hybrid Block Structure inside `RecurrentBlock`**:
   - Alternate loop layers: Even iterations ($t=0, 2, 4\dots$) use Linear Gated Attention (GDN); Odd iterations ($t=1, 3, 5\dots$) use 1 Full Attention step or Sparse Window Attention.

### Hyperparameters
- `attn_type`: `"lt2_hybrid"`
- `linear_state_dim`: 128
- `sparse_window_size`: 256

### Validation Criteria
- **Context Scaling**: Linear memory and time scaling up to sequence length $N = 32,768$.
- **Perplexity**: Loss within $0.02$ of full quadratic attention OpenMythos baseline.
- **Decoding Speed**: $\ge 2.2\times$ faster autoregressive token generation speed.

### Automated Failure Mitigation
- *Issue*: Linear attention state $S_t$ numerically underflows or overflows during fp16/bf16 recurrent steps.
- *Fix*: Normalize linear recurrent state $S_t \leftarrow S_t / (\max(|S_t|) + 1e-6)$ after every 4 loop steps.

---

## Research Avenue 4: Mixture-of-Depths (MoD) Attention — Capacity-Constrained Token Bypassing per Loop

- **Paper Reference**: *Mixture-of-Depths Attention* ([arXiv:2603.15619](https://arxiv.org/abs/2603.15619))
- **Core Hypothesis & Mechanism**: In any given loop iteration, not all tokens require heavy Attention and MoE computation. MoD enforces a fixed compute budget per loop by routing only the top $P\%$ (e.g. 50%) of tokens through the Attention/MoE block, while the remaining $(100-P)\%$ tokens bypass the block via a direct residual connection.

### Implementation Logic
1. **Module Creation**: Create `open_mythos/mod_router.py`.
2. **Top-K Capacity Routing**:
   ```python
   class MoDRouter(nn.Module):
       def __init__(self, dim: int, capacity_factor: float = 0.5):
           super().__init__()
           self.router = nn.Linear(dim, 1)
           self.capacity_factor = capacity_factor

       def forward(self, x: torch.Tensor):
           # x: (B, T, D)
           B, T, D = x.shape
           k = int(T * self.capacity_factor)
           weights = torch.sigmoid(self.router(x)).squeeze(-1) # (B, T)
           topk_weights, topk_indices = torch.topk(weights, k=k, dim=-1) # (B, k)
           return topk_indices, topk_weights
   ```
3. **Block Execution**:
   - Gather top $k$ tokens per batch sequence.
   - Run `Attention` and `MoE` only on gathered tokens `x_topk`.
   - Scatter outputs back to original sequence tensor; non-topk tokens keep their input value.

### Hyperparameters
- `capacity_factor`: $0.5$ (50% tokens processed per loop)
- `max_loop_iters`: 16 (achieves 16 loops of depth using 8 loops worth of compute)

### Validation Criteria
- **Compute Reduction**: Exactly 50% FLOPS reduction per loop block.
- **Performance**: Matches full compute baseline accuracy with $2\times$ deeper loop iterations.

### Automated Failure Mitigation
- *Issue*: Router selecting static token positions regardless of sequence context.
- *Fix*: Inject router noise during training: `router_logits += torch.randn_like(router_logits) * 0.1`.

---

## Research Avenue 5: Relaxed Recursive Transformers — Layer-Wise & Rank-Adaptive Depth LoRA

- **Paper Reference**: *Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA* ([arXiv:2410.20672](https://arxiv.org/pdf/2410.20672))
- **Core Hypothesis & Mechanism**: Pure weight tying across loops limits expressiveness, but un-tied layers ruin parameter efficiency. Relaxed Recursive Transformers add a small depth-wise LoRA adapter $W_{\text{eff}}(t) = W_{\text{shared}} + A_t B_t$ where rank $r(t)$ grows dynamically with recurrence depth $t$.

### Implementation Logic
1. **Modification to `LoRAAdapter` in `open_mythos/main.py`**:
   - Replace constant rank `rank` with adaptive rank table $r(t) = r_0 + \lfloor \alpha \cdot t \rfloor$.
   - Apply separate LoRA projection matrices to Prelude, Recurrent Attention (Q, K, V), Recurrent MoE, and Coda.
   ```python
   class DynamicDepthLoRA(nn.Module):
       def __init__(self, dim: int, max_loops: int, base_rank: int = 8, rank_step: int = 4):
           super().__init__()
           self.adapters = nn.ModuleList([
               nn.Sequential(
                   nn.Linear(dim, base_rank + t * rank_step, bias=False),
                   nn.Linear(base_rank + t * rank_step, dim, bias=False)
               ) for t in range(max_loops)
           ])

       def forward(self, x: torch.Tensor, loop_t: int):
           t_idx = min(loop_t, len(self.adapters) - 1)
           return self.adapters[t_idx](x)
   ```

### Hyperparameters
- `base_rank (r0)`: 8
- `rank_step (alpha)`: 4
- `max_loops`: 16

### Validation Criteria
- **Perplexity Improvement**: $\ge 0.08$ decrease in validation perplexity vs strictly tied weight baseline.
- **Parameter Overhead**: Parameter increase $< 5\%$ of total model size.

### Automated Failure Mitigation
- *Issue*: Depth LoRA weights dominate the shared weight representations during early training.
- *Fix*: Scale LoRA output by $\frac{\gamma}{r(t)}$ where $\gamma = 0.1$.

---

## Research Avenue 6: Parcae Spectral Radius Regularization & Advanced LTI State-Space Stability

- **Paper Reference**: *Parcae: Scaling Laws for Stable Looped Language Models* ([arXiv:2604.12946](https://arxiv.org/abs/2604.12946))
- **Core Hypothesis & Mechanism**: Recurrent hidden state updates $h_{t+1} = A h_t + B e + \text{Block}(h_t)$ diverge when the spectral radius $\rho(A) \ge 1.0$. Parcae guarantees $\rho(A) < 1.0$ by construction using a Continuous-to-Discrete Zero-Order Hold (ZOH) parametrization and dynamic step-size scheduling.

### Implementation Logic
1. **Refinement of `LTIInjection` in `open_mythos/main.py`**:
   $$\rho(A) = \max_i |A_i|, \quad A_i = \exp(-\exp(\log \Delta t + \log A_i))$$
2. **Log-Space Bounds & Regularization Loss**:
   ```python
   def compute_lti_loss(self):
       A = self.get_A()
       rho = torch.max(torch.abs(A))
       # Penalty if spectral radius exceeds safety bound 0.95
       penalty = torch.relu(rho - 0.95)**2
       return penalty
   ```
3. **Adaptive Discretization Step Size Schedule**:
   $$\Delta t(t) = \Delta t_0 \cdot (1 + \cos(\pi \cdot t / T_{\max}))$$

### Hyperparameters
- `log_A_init`: $-1.0$
- `log_dt_init`: $-2.0$
- `spectral_penalty_weight`: $0.1$

### Validation Criteria
- **Stability Guarantee**: Zero loss spikes across 50,000 training steps at high learning rate ($lr = 3\times 10^{-3}$).
- **Spectral Radius Bound**: $\rho(A)$ strictly bounded in $[0.70, 0.95]$ throughout training.

### Automated Failure Mitigation
- *Issue*: `log_dt + log_A` underflows in fp16.
- *Fix*: Clamp log arguments: `(self.log_dt + self.log_A).clamp(-15.0, 15.0)`.

---

## Research Avenue 7: Latent Reasoning & Continuous Space CoT Supervision — Test-Time Compute Extrapolation

- **Paper Reference**: *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach* ([arXiv:2502.05171](https://arxiv.org/abs/2502.05171)) & *Training LLMs to Reason in Continuous Latent Space* ([arXiv:2412.06769](https://arxiv.org/abs/2412.06769))
- **Core Hypothesis & Mechanism**: Looped transformers perform implicit chain-of-thought in continuous latent space. Supervising only the final Coda output causes intermediate loop representations to collapse. Adding auxiliary cross-entropy supervision at intermediate loop steps $t \in \{T/4, T/2, 3T/4\}$ forces progressive refinement and enables test-time loop depth extrapolation ($T_{\text{test}} = 2 \times T_{\text{train}}$).

### Implementation Logic
1. **Module Creation**: Create `AuxiliaryCodaHead` in `open_mythos/main.py`.
2. **Intermediate Loss Computation**:
   ```python
   class AuxiliaryCodaHead(nn.Module):
       def __init__(self, coda_block, lm_head):
           super().__init__()
           self.coda = coda_block
           self.lm_head = lm_head

       def forward(self, h_t, freqs_cis):
           coda_out = self.coda(h_t, freqs_cis)
           return self.lm_head(coda_out)
   ```
3. **Loss Objective**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{final\_coda}} + \sum_{t \in \{4, 8, 12\}} \lambda_t \mathcal{L}_{\text{CE}}(\text{AuxHead}(h_t), y)$$
   where $\lambda_t = 0.1 \cdot (t / T)$.

### Hyperparameters
- `train_loop_iters`: 8
- `test_loop_iters`: 16 or 32
- `aux_loss_weight`: 0.1

### Validation Criteria
- **Depth Extrapolation Accuracy**: Accuracy on multi-hop math/logic tasks improves monotonically as test loop count scales from $T=8 \rightarrow 16 \rightarrow 32$.
- **Latent Trajectory Smoothness**: Cosine distance $\|h_t - h_{t-1}\|$ decays smoothly without oscillating.

### Automated Failure Mitigation
- *Issue*: Intermediate loss dominates early training, preventing deep feature formation.
- *Fix*: Warm up $\lambda_t$ linearly over the first 2,000 training steps.

---

## Research Avenue 8: Looped Depth-Sharing with Loop-Index Positional Embeddings (Raschka / OpenAI Looped Depth)

- **Paper Reference**: *A Mechanistic Analysis of Looped Reasoning Language Models* ([arXiv:2604.11791](https://arxiv.org/abs/2604.11791)) & Sebastian Raschka *Looped Depth-Sharing* ([Blog](https://sebastianraschka.com/llm-architecture-gallery/looped-depth-sharing/))
- **Core Hypothesis & Mechanism**: Standard looped depth applies identical weights across loops without knowing which stage of reasoning the model is in. Replacing static scalar loop embeddings with 2D Complex Rotary Embeddings over recurrence depth ($t$) allows the same layer weights to execute distinct phase operations (e.g. extraction $\rightarrow$ composition $\rightarrow$ verification) across loop iterations.

### Implementation Logic
1. **Function Creation**: Create `RoPELoopEmbedding` in `open_mythos/main.py`.
2. **RoPE Rotation over Recurrence Depth**:
   ```python
   def apply_rope_loop_index(h: torch.Tensor, loop_t: int, theta: float = 10000.0):
       # h: (B, T, D)
       dim = h.shape[-1]
       half_dim = dim // 2
       freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32, device=h.device) / half_dim))
       angle = loop_t * freqs # (half_dim,)
       cos_emb = torch.cos(angle).repeat(2) # (dim,)
       sin_emb = torch.sin(angle).repeat(2)
       
       # Rotate h along feature dimension
       h_rot = torch.cat([-h[..., half_dim:], h[..., :half_dim]], dim=-1)
       return h * cos_emb + h_rot * sin_emb
   ```

### Hyperparameters
- `rope_loop_theta`: 10000.0
- `max_loop_iters`: 16

### Validation Criteria
- **Mechanistic Differentiation**: Probing internal attention maps shows distinct attention patterns at $t=1$ (broad context search) vs $t=16$ (focused token target).
- **Validation Loss**: $\ge 0.05$ loss reduction over static sinusoidal embedding baseline.

### Automated Failure Mitigation
- *Issue*: High rotation frequencies disrupt pre-trained hidden state norms.
- *Fix*: Apply RMSNorm immediately after `apply_rope_loop_index`.

---

## Research Avenue 9: Universal Transformer Halting — Dynamic ACT Threshold & Ponder Loss Regularization

- **Paper Reference**: *Universal Transformers* ([arXiv:1807.03819](http://arxiv.org/abs/1807.03819)) & *Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers* ([arXiv:2604.07822](https://arxiv.org/abs/2604.07822))
- **Core Hypothesis & Mechanism**: Without dynamic halting, models overthink simple tokens, leading to state drift and degradation. Upgrading `ACTHalting` with per-token Ponder Loss $\mathcal{L}_{\text{ponder}} = \tau \sum_i N(i)$ enables automatic early halting for simple tokens while reserving deep recurrence for complex tokens.

### Implementation Logic
1. **Enhancement to `ACTHalting` in `open_mythos/main.py`**:
   ```python
   class DynamicACTHalting(nn.Module):
       def __init__(self, dim: int, threshold: float = 0.99):
           super().__init__()
           self.halt_linear = nn.Linear(dim, 1)
           self.threshold = threshold

       def forward(self, h, cum_probs, halted_mask):
           p = torch.sigmoid(self.halt_linear(h)).squeeze(-1) # (B, T)
           new_cum_probs = cum_probs + p
           new_halted = new_cum_probs >= self.threshold
           # Compute remainder for exact probability weighting
           remainder = torch.clamp(1.0 - cum_probs, min=0.0)
           effective_p = torch.where(new_halted & ~halted_mask, remainder, p)
           return effective_p, new_cum_probs, new_halted
   ```
2. **Ponder Loss Objective**:
   $$\mathcal{L}_{\text{ponder}} = \tau \cdot \frac{1}{B \cdot T} \sum_{i=1}^B \sum_{j=1}^T N_{i,j}$$
   where $N_{i,j}$ is the fractional loop count before halting, and $\tau = 0.001$.

### Hyperparameters
- `act_threshold`: 0.99
- `ponder_tau`: 0.001

### Validation Criteria
- **Average Ponder Steps**: Easy tokens halt in $\approx 3$ loops; hard reasoning tokens run up to $16$ loops.
- **Overthinking Prevention**: Zero degradation in accuracy when max loop iterations increase from $16 \rightarrow 32$.

### Automated Failure Mitigation
- *Issue*: Ponder loss penalty is too high, causing all tokens to halt at step 1.
- *Fix*: Dynamic tau annealing: start at $\tau = 0.0$, scale to $0.001$ over 5,000 steps.

---

## Research Avenue 10: Reasoning with Latent Thoughts — Multi-Trajectory Continuous Latent Beam Search

- **Paper Reference**: *Reasoning with Latent Thoughts — On the Power of Looped Transformers* ([arXiv:2502.17416](https://arxiv.org/abs/2502.17416))
- **Core Hypothesis & Mechanism**: Single continuous hidden state paths can commit to wrong reasoning directions. Latent Beam Search forks the hidden state $h_t$ into $B=2$ continuous latent streams at an intermediate loop step $t_0$, processes both in parallel, scores their likelihood via an auxiliary value head, and prunes/merges them back into a single state at the Coda.

### Implementation Logic
1. **Module Creation**: Create `open_mythos/latent_beam.py`.
2. **Beam Branching Protocol**:
   - At loop step $t_{\text{split}} = T / 2$, create stream $A = h_t$ and stream $B = h_t + \delta \cdot \text{MLP}_{\text{perturb}}(h_t)$.
   - Process streams independently through remaining loop iterations: $h_{t, A}$ and $h_{t, B}$.
   - At loop $T$, compute confidence scores $s_A, s_B = \text{softmax}(V(h_{T,A}), V(h_{T,B}))$.
   - Merge states: $h_{\text{final}} = s_A \cdot h_{T,A} + s_B \cdot h_{T,B}$.

### Hyperparameters
- `beam_width`: 2
- `split_loop_step`: 8
- `perturb_scale (delta)`: 0.05

### Validation Criteria
- **Problem Solving Accuracy**: $\ge 12\%$ accuracy boost on complex reasoning benchmarks (GSM8K, MATH, HumanEval-style synthesis).
- **Latent Divergence**: Streams $A$ and $B$ explore distinct representation sub-manifolds before converging.

### Automated Failure Mitigation
- *Issue*: Streams $A$ and $B$ become identical ($\text{cos\_sim} \approx 1.0$).
- *Fix*: Add orthogonality loss penalty: $\mathcal{L}_{\text{orth}} = (\frac{h_A \cdot h_B}{\|h_A\| \|h_B\|})^2$.

---

## Research Avenue 11: The Recurrent Transformer — Key-Value Recycled Memory & Efficient Autoregressive Decoding

- **Paper Reference**: *The Recurrent Transformer: Greater Effective Depth and Efficient Decoding* ([arXiv:2604.21215](https://arxiv.org/abs/2604.21215))
- **Core Hypothesis & Mechanism**: Recalculating Key ($K$) and Value ($V$) projections from scratch at every loop iteration $t$ creates severe memory bandwidth bottlenecks during autoregressive decoding. KV Memory Recycling updates $K$ and $V$ incrementally using an exponential moving average over loop steps: $K_t = \alpha K_{t-1} + (1-\alpha) W_K h_t$.

### Implementation Logic
1. **Modification to `GQAttention` & `MLAttention` in `open_mythos/main.py`**:
   ```python
   def forward_recycled_kv(self, x, freqs_cis, prev_k=None, prev_v=None, alpha=0.7):
       B, T, _ = x.shape
       new_k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
       new_v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
       new_k = apply_rope(new_k, freqs_cis)
       
       if prev_k is not None:
           k = alpha * prev_k + (1.0 - alpha) * new_k
           v = alpha * prev_v + (1.0 - alpha) * new_v
       else:
           k, v = new_k, new_v
           
       return k, v
   ```

### Hyperparameters
- `kv_recycle_alpha`: 0.7
- `max_loop_iters`: 16

### Validation Criteria
- **Inference Latency**: $\ge 2.5\times$ speedup in per-token generation latency.
- **Memory Bandwidth Reduction**: Memory read/write volume reduced by $\approx 60\%$.
- **Validation Loss**: Perplexity delta $\le +0.01$ vs baseline.

### Automated Failure Mitigation
- *Issue*: High $\alpha$ causes stale KV representations during early loops.
- *Fix*: Dynamic alpha schedule: $\alpha(t) = \alpha_{\max} \cdot (1 - e^{-t / 2})$.

---

## Research Avenue 12: Nanbeige Agentic Compact Architecture — Fine-Grained MoE Expert Allocation & Hyperparameter Calibration

- **Paper Reference**: *Nanbeige4.2-3B: Unlocking Agentic Capabilities in a Compact Model* ([arXiv:2607.22083](https://arxiv.org/abs/2607.22083))
- **Core Hypothesis & Mechanism**: Compact 3B parameter models operating on unified memory (NVIDIA DGX Spark) require hyper-optimized expert allocation to maximize domain specialization without exceeding memory bandwidth. Setting 128 fine-grained routed experts (top-8 active per token) combined with 4 always-on shared experts yields the agentic multi-step capabilities of a 14B model within a 3B footprint.

### Implementation Logic
1. **Create Pre-configured Scale in `open_mythos/variants.py`**:
   ```python
   def mythos_nanbeige_3b() -> MythosConfig:
       return MythosConfig(
           vocab_size=32000,
           dim=2560,
           n_heads=20,
           n_kv_heads=4, # GQA ratio 5:1
           max_seq_len=8192,
           max_loop_iters=16,
           prelude_layers=2,
           coda_layers=2,
           attn_type="mla",
           kv_lora_rank=512,
           q_lora_rank=1280,
           qk_rope_head_dim=64,
           qk_nope_head_dim=128,
           v_head_dim=128,
           n_experts=128,          # Fine-grained experts
           n_shared_experts=4,     # Always-on shared experts
           n_experts_per_tok=8,    # Top-8 routed
           expert_dim=320,         # Compact inner expert dim
           lora_rank=16,
           rope_theta=500000.0,
       )
   ```
2. **DGX Spark Execution Command (`run_mythos_spark.sh`)**:
   ```bash
   torchrun --nproc_per_node=1 training/dgx_spark_train.py \
       --config nanbeige_3b \
       --batch_size 8 \
       --gradient_accumulation_steps 4 \
       --learning_rate 2e-3 \
       --max_steps 20000
   ```

### Hyperparameters
- `dim`: 2560
- `n_experts`: 128 (routed), 4 (shared)
- `n_experts_per_tok`: 8
- `batch_size`: 8 (micro), 32 (effective)

### Validation Criteria
- **Tool Use & Function Calling Accuracy**: Multi-turn agentic tool calling accuracy $\ge 84\%$.
- **DGX Spark Throughput**: $\ge 18,000$ tokens/sec in bf16 mixed precision.
- **Memory Footprint**: Peak VRAM $\le 36$ GB on DGX Spark.

### Automated Failure Mitigation
- *Issue*: MoE routing imbalance causes VRAM spikes on specific expert buffers.
- *Fix*: Enable aux-loss-free router bias adjustment: `router_bias -= 0.01 * (expert_load - mean_load)`.

---

## Master Experiment Matrix for Hermes Agent

| Avenue ID | Architecture Feature | Primary Paper Source | Key Metric Goal | Failure Recovery Action |
|---|---|---|---|---|
| **RA-01** | Mixture-of-Recursions (MoR) | arXiv:2507.10524 | 35% FLOPs reduction | Add load-balancing entropy loss |
| **RA-02** | Hyperloop Multi-Stream | arXiv:2604.21254 | Equal loss at 50% params | LayerNorm over stream dim K |
| **RA-03** | LT2 Hybrid Attention | arXiv:2605.20670 | 2.2x decoding speedup | Normalize linear state S_t |
| **RA-04** | Mixture-of-Depths (MoD) | arXiv:2603.15619 | 50% compute per loop | Add router logit noise |
| **RA-05** | Rank-Adaptive Depth LoRA | arXiv:2410.20672 | -0.08 Val Perplexity | Scale output by gamma / r(t) |
| **RA-06** | Parcae LTI Stability | arXiv:2604.12946 | rho(A) in [0.70, 0.95] | Clamp log space parameters |
| **RA-07** | Latent CoT Supervision | arXiv:2502.05171 | 2x test depth scaling | Warm up intermediate loss weight |
| **RA-08** | RoPE Loop-Index Embedding | arXiv:2604.11791 | Distinct loop attention maps | RMSNorm after loop RoPE |
| **RA-09** | Dynamic ACT Ponder Loss | arXiv:1807.03819 | Zero overthinking decay | Anneal ponder penalty tau |
| **RA-10** | Continuous Latent Beam Search | arXiv:2502.17416 | +12% MATH accuracy | Add orthogonality loss penalty |
| **RA-11** | Recycled KV Memory | arXiv:2604.21215 | 2.5x inference speedup | Dynamic alpha schedule |
| **RA-12** | Nanbeige 3B Compact MoE | arXiv:2607.22083 | 18k tokens/sec on Spark | Aux-loss-free bias update |

---

## Implementation Order & Milestone Plan

1. **Phase 1 (Stability & Fundamentals)**: Implement **RA-06 (Parcae LTI)** and **RA-08 (RoPE Loop-Index)** first to ensure the core recurrent loop is 100% stable.
2. **Phase 2 (Compute & Memory Efficiency)**: Implement **RA-01 (MoR)**, **RA-04 (MoD)**, and **RA-11 (Recycled KV)** to maximize throughput on DGX Spark.
3. **Phase 3 (Subquadratic Scaling & Expressiveness)**: Implement **RA-03 (LT2)**, **RA-05 (Depth LoRA)**, and **RA-02 (Hyperloop)**.
4. **Phase 4 (Deep Reasoning & Agentic Capabilities)**: Implement **RA-07 (Latent Reasoning)**, **RA-09 (Dynamic ACT)**, **RA-10 (Latent Beam Search)**, and **RA-12 (Nanbeige 3B)**.

---
*End of Autonomous Research Plan for Local Hermes AI Agent.*
