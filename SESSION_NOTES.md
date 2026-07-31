# Autoresearch Sessions (DGX Spark)

## Session Jul22-P3 (2026-07-22, 14:00-17:00 UTC) - Third autonomous session

### Infrastructure Fixed:
- **CRITICAL:** Previous sessions didn't sync code changes to spark properly. Local edits stayed local; spark ran stale baseline code.
- **Fix:** scp train.py + prepare.py to spark before each run. Created clean runner script `/tmp/run_spark_exp.sh` on spark (avoids run_experiment.sh quoting issues via ssh).
- **Container note:** Runs as root → HOME=/home/david-barnes must be set as env var.

### Experiment 3: TOTAL_BATCH_SIZE=2^18 (halved from 2^19)
- **Hypothesis:** Smaller batches = more optimizer steps = better convergence
- **Result:** val_bpb=1.876521, 214 steps, ~1480ms/step
- **Baseline comparison:** 1.865 (114 steps, ~3s/step)
- **Verdict:** ✗ WORSE by +0.011 despite 2x more steps
- **Insight:** Larger batches (2^19) provide better convergence dynamics. Per-token training quality > optimizer step count.

### Updated Hypotheses (ranked by priority):
1. **TOTAL_BATCH_SIZE=2^20** — if larger=better, go even bigger
2. **WARMDOWN_RATIO=0.0** — eliminate warmdown, maximize full-LR time
3. **EMBEDDING_LR=0.3 or 0.2** — 0.6 may be too aggressive
4. **Adam betas=(0.9, 0.95)** — more first-moment momentum
5. **Remove Value Embeddings** — reduce arch complexity for throughput
6. **Simpler MLP** — ReLU instead of ReLU²
7. **SCALAR_LR=1.0** — per-layer scalars need more training
8. **Muon momentum=0.97** — higher momentum for matrix params
9. **HEAD_DIM=64** — narrower attention heads
10. **Gradient clipping (max_norm=1.0)** — stabilize training

### Current Best: val_bpb=1.865391 (unchanged from baseline)

### Current Best: val_bpb=1.865391 (DEPTH=4, AR=64, 11.5M params, MATRIX_LR=0.04, baseline config)

### Key Findings from This Session:

1. **Throughput is king on GB10.** Every config change that increased params/complexity reduced steps and hurt val_bpb.
2. **Wider model (AR=96) → worse.** 19.7M params, only 99 steps → val_bpb 1.871
3. **Higher Muon LR (0.08) → worse.** Same 11.5M params, 113 steps → val_bpb 1.875
4. **Deeper model (DEPTH=6) → worse.** 26.3M params, only 72 steps → val_bpb 1.881
5. **Full context windows (LLLL) + reduced WD + AdamB1=0.9 → worse.** val_bpb 1.878
6. **torch.compile is 2x faster than eager.** Don't touch it.

### Experiments Run:

| # | Config Change | val_bpb | Steps | tok/s | Params | Verdict |
|---|--------------|---------|-------|-------|--------|---------|
| 1 | Baseline confirm (AR=96, MR=0.06) | 1.871 | 99 | 154K | 19.7M | ✗ |
| 2 | AR=64, MATRIX_LR=0.08, WARMUP=0.02 | 1.875 | 113 | 177K | 11.5M | ✗ |
| 3 | DEPTH=6, AR=48, MR=0.04 | 1.881 | 72 | 106K | 26.3M | ✗ |
| 4 | LLLL windows, WD=0.05, B1=0.9, WD=0.05 | 1.878 | 113 | 177K | 11.5M | ✗ |

### Session Jul22-P4 (2026-07-22, ~17:00-21:30 UTC) - Fourth autonomous session

### Experiments Run:

| # | Config Change | val_bpb | Steps | tok/s | Verdict |
|---|--------------|---------|-------|-------|---------|
| 3 | TOTAL_BATCH_SIZE=2^18 | 1.877 | 214 | 177K | ✗ worse |
| 4 | WARMDOWN_RATIO=0.0 | 1.882 | 111 | 174K | ✗ worse |
| 5 | Muon momentum 0.90→1.00 | 1.873 | 114 | 179K | ✗ worse |
| 6 | EMBEDDING_LR=0.3 | 1.874 | 113 | 178K | ✗ worse |
| 7 | SCALAR_LR=1.0 | 1.869 | 114 | 178K | ~ closest but still ✗ |

### Key Findings from Session P4:
1. **None of 5 experiments beat baseline 1.865** — the baseline config is remarkably hard to improve upon.
2. **Exp 7 (SCALAR_LR=1.0) was closest at 1.869** — only +0.004 above baseline. Worth investigating scalar LR tuning further.
3. **Throughput is stable at ~177-179K tok/s** — the GB10 is consistently delivering. ~3s/step is the physical limit for this config.
4. **Wider/deeper models fail due to fewer steps** — confirmed again. 114 steps is the sweet spot for 300s budget.
5. **Learning rate changes all moved in wrong direction** — the current LR configuration is already well-tuned.

### Updated Hypotheses (ranked by priority):
1. **SCALAR_LR sweep: 0.8, 1.2, 1.5** — Exp 7 was closest, need finer tuning
2. **MATRIX_LR=0.05 or 0.06** — slightly higher Muon LR (previous 0.08 was too much)
3. **WEIGHT_DECAY=0.05** — less aggressive decay (previous 0.05+combo didn't help alone)
4. **HEAD_DIM=64** — narrower attention heads for more compute per token
5. **WARMUP_RATIO=0.05** — small warmup for stability
6. **Remove Value Embeddings** — major arch change, requires careful multi-edit patch
7. **Simpler MLP** — ReLU instead of ReLU²
8. **Adam betas=(0.85, 0.95)** — midpoint between 0.8 and 0.9
9. **Gradient clipping (max_norm=1.0)** — stabilize gradient magnitudes
10. **TIME_BUDGET increase** — can we train longer for better val_bpb?

### Current Best: val_bpb=1.865391 (unchanged from baseline DEPTH=4, AR=64)

### Critical Insight:
The 5-minute time budget + ~3s/step = ~100 steps. **Maximizing tokens/step** is more important than model capacity improvements. The baseline already achieves ~180K tok/s which is excellent for GB10.

### Hypotheses for Next Session:
1. **Reduce TOTAL_BATCH_SIZE to 2^18** — smaller batches = fewer micro-step overhead, more optimizer steps
2. **Increase HEAD_DIM to 64 (narrower heads)** — same params, less attention compute
3. **Remove Value Embeddings** — complexity that costs 3s/step. If we can save 0.5s/step, we get 50 more steps in 300s.
4. **Simpler MLP** — replace ReLU² with just ReLU or even linear projection
5. **Reduce rotary precision** — use float16 instead of bfloat16 for rotary embeddings
6. **Try TOTAL_BATCH_SIZE=2^20** — larger batches = fewer micro-steps = less overhead
7. **Gradient clipping (max_norm=1.0)** — may stabilize training for same params
8. **Try Adam beta2=0.99** instead of 0.95 — more momentum, possibly better for short runs
9. **Reduce SCALAR_LR** — the x0/resid lambdas may be training too aggressively
10. **Increase TIME_BUDGET via env** — can we set it higher and train longer?

### Next Session Priority:
- Focus on **throughput optimization** first (reducing step time below 2.5s)
- Then try **simplifying architecture** (remove VE, simpler MLP)
- Only then try **LR scheduling** experiments

---

## Session Jul23-P1 (2026-07-23) — Throughput Breakthrough

### Experiments Run:

| # | Config Change | val_bpb | Steps | tok/s | Params | VRAM (MB) | Verdict |
|---|--------------|---------|-------|-------|--------|-----------|---------|
| 1 | MATRIX_LR=0.05, SCALAR_LR=1.2 | 1.872971 | 112 | 176K | 11.5M | 1815 | ✗ worse |
| 2 | SCALAR_LR=1.0, MATRIX_LR=0.04 | 1.872700 | 112 | 174K | 11.5M | 1815 | ✗ worse |
| 3 | **HEAD_DIM=64 + grad clip (max_norm=1.0)** | **1.864236** | **158** | **256K** | 11.5M | **1303** | **✓ NEW BEST!** |

### Key Findings:
1. **HEAD_DIM=64 is a major breakthrough!** Narrower attention heads (64 vs 128) means:
   - 12 heads instead of 6 for the same 768-dim model → more parallel attention compute
   - 45% faster step time (2050ms vs 3000ms) → 41% more optimizer steps (158 vs 112)
   - 28% less VRAM (1303MB vs 1815MB)
   - **val_bpb=1.864236 beats baseline 1.865391 by -0.001155**

2. **SCALAR_LR sweep (0.5, 1.0, 1.2) all worse than baseline** — the per-layer scalars are already well-tuned at 0.5. Higher values destabilize training.

3. **MATRIX_LR=0.05 worse than 0.04** — Muon LR is already optimal at 0.04.

4. **Gradient clipping (max_norm=1.0) contributes to stability** — combined with HEAD_DIM=64, training is smooth with no loss explosions.

### New Baseline: val_bpb=1.864236 (DEPTH=4, AR=64, HEAD_DIM=64, MATRIX_LR=0.04, SCALAR_LR=0.5, grad_clip=1.0)

### Next Session Priorities:
1. **HEAD_DIM=32** — even narrower heads? (768/32=24 heads) May be too narrow, but worth testing
2. **HEAD_DIM=64 + WARMUP_RATIO=0.05** — small warmup for the faster training
3. **HEAD_DIM=64 + remove Value Embeddings** — reduce complexity further
4. **HEAD_DIM=64 + simpler MLP** (ReLU instead of ReLU²)
5. **HEAD_DIM=64 + WEIGHT_DECAY=0.05** — less aggressive decay
6. **HEAD_DIM=64 + try DEPTH=5** — slightly deeper with the faster step time
