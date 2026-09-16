# autoresearch-DGX-Spark

> An adaptation of Karpathy's `autoresearch`, re-targeted to the **NVIDIA DGX Spark** (GB10 Grace Blackwell Superchip) and repurposed from 5-minute speedrun tuning into a long-term autonomous research program: implementing **Recursive Depth Transformer (RDT)** avenues and the new **Test-Time Training (TTT)** methodologies — working toward a theoretical re-implementation of what the forthcoming 10T-parameter GPT model (codenamed *"Bel"*) may be using, along with the RDT architecture from the new *Astra* model.

This repository is now the **canonical master copy** of the project. (An earlier fusion repo, `autoresearch-OpenMythos-KellerJordanSpeedrun`, is retained only as history.)

## Lineage

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the original: give an AI agent a small but real LLM training setup, let it experiment autonomously overnight (5-minute budget, `val_bpb` metric), keep-or-discard.
- **DGX Spark adaptation** — Docker-based execution with unified-memory optimizations (pinned memory, OOM protection, ARM64 support), plus venv-based training routes.
- **OpenMythos × Keller-Jordan speedrun fusion** — Nano-Mythos validation models, looped/recurrent transformer blocks, ACT halting metrics.
- **This repo** — diverged far enough through the research program below that it is now its own project.

## Research goal

Frontier-model reports point at three converging directions: **fast weights / two-speed learning**, **test-time training with dynamic weight adaptation**, and **weight consolidation for continual learning** — wrapped in closed-loop verifier self-improvement. This repo tests all three, at small scale, one avenue at a time, keeping only what validates:

- **Phase A — Recursive Depth Transformers** (`docs/research_plan/recursion_research_plan.md`): 12 avenues (see below). In progress.
- **Phase B — Fast weights & TTT** (`docs/TTT/hermes_research_plan.md`): 4 vetted, agent-feasible avenues (see below). Queued until Phase A completes.

## How it works

An autonomous researcher (Hermes cron agent, daily 10:00) works through avenues **one at a time**:

1. Implements the avenue's module in `open_mythos/` or `training/` (Nano-Mythos ~9M-param validation setup; headline figures from the papers are scaled down proportionally).
2. Validates against the avenue's criteria — full training runs where applicable, targeted benchmarks (retrieval accuracy, throughput, VRAM, forgetting) where not. The agent chooses the method to fit the avenue.
3. **5-strike rule**: 5 consecutive non-improving tests ends the avenue (adopted-with-wins | neutral | exhausted | not-implementable).
4. Every experiment is appended to `SESSION_NOTES.md` (with per-phase progress trackers) and `results.tsv`, then committed and pushed. Adopted wins carry into the next avenue's baseline unless they conflict.

No 1B/1.5B scale-up runs without explicit owner approval.

## Results so far

Current validation baseline: **val_bpb 2.304970 @1200 steps** (full Parcae + RoPE loop-index + MoD cap-0.5). Full history in `SESSION_NOTES.md`.

## Phase A avenues — Recursive Depth Transformers

- **RA-01 Mixture-of-Recursions (MoR)** — routes each token through different recursion depths via expert-choice gating with selective KV caching, so easy tokens skip loop iterations.
  Status: ✅ completed, neutral — redundant with the existing ACT per-token halting; not adopted.
- **RA-02 Hyperloop Transformers** — replaces vector residual streams with matrix-valued states and multi-state hyper-connections, letting loop iterations share richer memory.
  Status: ⏳ not started.
- **RA-03 LT2 Linear-Time Looped Transformers** — hybrid subquadratic attention operating in continuous latent space, aiming for linear-time looped reasoning.
  Status: ⏳ not started.
- **RA-04 Mixture-of-Depths (MoD)** — capacity-constrained token bypassing per loop: only a fixed fraction (cap-0.5) of tokens take the full-depth path.
  Status: ✅ completed, adopted — val_bpb 2.304970 @1200 plus −19% train time and −18% VRAM.
- **RA-05 Relaxed Recursive Transformers** — layer-wise and rank-adaptive depth LoRA adapters that relax strict weight-tying across loop iterations.
  Status: ⏳ not started.
- **RA-06 Parcae spectral stability** — spectral-radius regularization and LTI state-space stabilization so looped weights stay contractive (ρ(A) < 1) through depth.
  Status: ✅ completed, adopted — ρ(A) pinned at 0.950, val_bpb 2.559975 @800.
- **RA-07 Latent reasoning & continuous-space CoT** — supervises chain-of-thought in continuous latent space to extrapolate test-time compute.
  Status: ⏳ not started.
- **RA-08 Loop-index positional embeddings** — RoPE-style rotary encodings over recurrence depth so shared loop weights differentiate their per-iteration roles.
  Status: ✅ completed, adopted — val_bpb 2.306980 @1200 with a compounding late-training win.
- **RA-09 Universal Transformer halting** — dynamic Adaptive Computation Time thresholds with ponder-loss regularization for per-token loop exit.
  Status: ⏳ not started.
- **RA-10 Reasoning with Latent Thoughts** — multi-trajectory continuous latent beam search, keeping several reasoning paths alive through the loop.
  Status: ⏳ not started.
- **RA-11 Recurrent Transformer memory** — key-value recycled memory for efficient autoregressive decoding across loop iterations.
  Status: ⏳ not started.
- **RA-12 Nanbeige compact agentic architecture** — fine-grained MoE expert allocation and hyperparameter calibration for small-but-agentic models.
  Status: ⏳ not started.

## Phase B avenues — Fast weights & Test-Time Training (vetted core)

- **TTT-05 Infini-Attention** — segments sequences into chunks with local masked attention while compressing history into a fixed-size linear memory matrix plus normalization vector.
  Simplest and most robust avenue; pure chunked matrix ops, no custom kernels. Queued first.
- **TTT-01 Delta-rule fast weights** — replaces quadratic attention with an associative fast-weight matrix updated via the Widrow–Hoff delta rule, writing only residual error to avoid saturation.
  Must use chunk-parallel recurrence — token-by-token Python loops would bottleneck Spark memory bandwidth.
- **TTT-06 TTT-Linear recurrent layers** — reformulates recurrent state as weights of an internal linear model trained online via self-supervised reconstruction, using the parallel "dual form".
  TTT-Linear only; TTT-MLP (multi-step non-linear autograd per chunk) is explicitly out of scope.
- **TTT-02 Titans test-time memory** — sliding-window attention paired with a long-term associative memory modulated by gradient surprise and dynamic forgetting gates.
  Memory restricted to a linear projection so surprise gradients stay cheap outer products; deep-MLP memory is out of scope.

## Repository layout

```
prepare.py / train.py      — upstream autoresearch core (data prep, base training)
training/                  — Nano-Mythos trainers (nano_mythos_train.py, dgx_spark_train.py, smoke test)
open_mythos/               — architecture modules (on the Spark OpenMythos clone; mirrored here via experiments)
docs/research_plan/        — Phase A plan + paper PDFs
docs/TTT/                  — Phase B plan + paper summaries
run_nano_mythos_spark.sh   — canonical training runner | run_exp_spark.sh — sweep runner
scripts/archive/           — retired one-off runners (reference only, do not execute)
SESSION_NOTES.md           — append-only experiment log + progress trackers
results.tsv                — every experiment row
```

## Running on the DGX Spark

```bash
ssh spark
cd ~/autoresearch-DGX-Spark
# Docker route (preferred):
bash run_nano_mythos_spark.sh
# Venv route (fallback, e.g. during driver mismatches):
# source ~/OpenMythos/.venv/bin/activate && TRITON_CACHE_DIR=/tmp/triton_cache_raXX python training/nano_mythos_train.py
```

Mandatory environment: `NCCL_P2P_DISABLE=1`, `TORCH_CUDA_ARCH_LIST=12.0`, `HOME=/home/david-barnes` (inside Docker), micro-batch ≤ 8 at seq 2048/4096. Smoke test first: `python training/smoke_test_train.py --steps 50 --batch_size 4`.

## Attribution

Core concept, training harness, and the opening quote belong to Andrej Karpathy's `autoresearch` (MIT). Speedrun lineage via Keller Jordan's modded-nanoGPT work; Nano-Mythos/OpenMythos architecture from its contributors. Everything in `docs/`, `training/nano_mythos_train.py`, and the research program above is this project's own work.
