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

- **Phase A — Recursive Depth Transformers** (`docs/research_plan/recursion_research_plan.md`): 12 avenues (mixture-of-recursions, Parcae LTI stability, RoPE loop-index embeddings, mixture-of-depths, latent reasoning, ACT ponder loss, …).
- **Phase B — Fast weights, TTT & consolidation** (`docs/TTT/hermes_research_plan.md`): 12 avenues — delta-rule fast weights, Titans neural memory, Infini-attention, TTT-linear/MLP layers, in-place TTT, streaming TTT-LoRA, CLS replay, DualNet, Progress & Compress, O-LoRA, CDL-Prompt, verifier-anchored self-improvement.

## How it works

An autonomous researcher (Hermes cron agent, daily 10:00) works through avenues **one at a time**:

1. Implements the avenue's module in `open_mythos/` or `training/` (Nano-Mythos ~9M-param validation setup; headline figures from the papers are scaled down proportionally).
2. Validates against the avenue's criteria — full training runs where applicable, targeted benchmarks (retrieval accuracy, throughput, VRAM, forgetting) where not. The agent chooses the method to fit the avenue.
3. **5-strike rule**: 5 consecutive non-improving tests ends the avenue (adopted-with-wins | neutral | exhausted | not-implementable).
4. Every experiment is appended to `SESSION_NOTES.md` (with per-phase progress trackers) and `results.tsv`, then committed and pushed. Adopted wins carry into the next avenue's baseline unless they conflict.

No 1B/1.5B scale-up runs without explicit owner approval.

## Results so far

| Avenue | Result | Status |
|--------|--------|--------|
| RA-06 Parcae LTI stability | ρ(A) pinned 0.950, val_bpb 2.559975 @800 (baseline 2.566979) | ✅ adopted |
| RA-08 RoPE loop-index embedding | val_bpb 2.306980 @1200 (baseline 2.314457, compounding win) | ✅ adopted |
| RA-01 Mixture-of-Recursions | Neutral — redundant with existing ACT halting | ➖ not adopted |

Current validation baseline: **val_bpb 2.306980 @1200 steps** (full Parcae + RoPE loop-index). Full history in `SESSION_NOTES.md`.

## Repository layout

```
prepare.py / train.py      — upstream autoresearch core (data prep, base training)
training/                  — Nano-Mythos trainers (nano_mythos_train.py, dgx_spark_train.py, smoke test)
open_mythos/               — architecture modules (on the Spark OpenMythos clone; mirrored here via experiments)
docs/research_plan/        — Phase A plan + paper PDFs
docs/TTT/                  — Phase B plan + paper summaries
run_nano_mythos_spark.sh   — Spark Docker runner | run_raXX_*.sh — per-avenue runners
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
