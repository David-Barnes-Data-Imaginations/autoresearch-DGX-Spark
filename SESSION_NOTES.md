# Autoresearch Sessions (DGX Spark)

## Session Jul22-P2 (2026-07-22, 11:00-13:00 UTC) - Second autonomous session

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
