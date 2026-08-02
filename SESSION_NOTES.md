# Session Notes — Speedrun Training

## Session Date: 2026-08-01

## Summary
Today's session focused on establishing a stable baseline and running a gradient clipping sweep. We also fixed the MFU (Model FLOPs Utilization) calculation which was incorrectly using H100 peak FLOPS instead of GB10.

## Baseline Results
- **Config**: DEPTH=4, HEAD_DIM=64, ASPECT_RATIO=64, 11.5M params
- **Optimizer**: MuonAdamW (Muon for 2D matrices, AdamW for embeddings/scalars)
- **Batch**: TOTAL_BATCH_SIZE=2^19 (~524K tokens/step)
- **Time budget**: 300s (5 min)

## Experiments Run

| # | Config Change | val_bpb | Steps | tok/s | MFU | Verdict |
|---|--------------|---------|-------|-------|-----|---------|
| 1 | Baseline (5 shards, grad_clip=1.0) | 1.864224 | 157 | 254K | 30.2% | ✓ baseline |
| 2 | WARMUP=0.02, MATRIX_LR=0.05 | 1.865084 | 158 | 253K | 30.5% | ✗ worse |
| 3 | 10 shards, EMBEDDING_LR=0.6 | 1.864132 | 157 | 254K | 30.3% | ✓ slight improvement |
| 4 | 15 shards, EMBEDDING_LR=0.7 | 1.864180 | 157 | 253K | 30.0% | ✗ worse |
| 5 | grad_clip=0.5 | 1.863679 | 157 | 254K | 30.2% | ✓ improvement |
| 6 | grad_clip=0.3 | 1.863344 | 158 | 254K | 30.4% | ✓ best so far |
| 7 | grad_clip=0.2 | 1.863337 | 157 | 255K | 30.3% | ✓ same as 0.3 |

## Session Date: 2026-08-02

### Additional Experiments

| # | Config Change | val_bpb | Steps | tok/s | MFU | Verdict |
|---|--------------|---------|-------|-------|-----|---------|
| 8 | HEAD_DIM=32 (narrower heads) | 1.863738 | 154 | 249K | 29.7% | ✗ worse |
| 9 | ADAM_BETAS=(0.9, 0.95) | 1.865795 | 157 | 250K | 30.3% | ✗ worse |
| 10 | ASPECT_RATIO=32 (smaller model) | 1.869546 | 198 | 258K | 15.4% | ✗ worse |
| 11 | DEPTH=5 (deeper model) | 1.867396 | 118 | 176K | 37.7% | ✗ worse |
| 12 | MATRIX_LR=0.05 | 1.864480 | 157 | 250K | 30.2% | ✗ worse |
| 13 | SCALAR_LR=0.6 | 1.863487 | 156 | 250K | 30.1% | ✗ worse |
| 14 | EMBEDDING_LR=0.7 | 1.863360 | 157 | 251K | 30.3% | ✓ slight improvement |
| 15 | EMBEDDING_LR=0.8 | 1.863418 | 157 | 251K | 30.3% | ✗ worse |
| 16 | EMBEDDING_LR=0.65 | 1.863348 | 158 | 251K | 30.3% | ✓ best so far |
| 17 | EMBEDDING_LR=0.75 | 1.863481 | 158 | 251K | 30.4% | ✗ worse |
| 18 | EMBEDDING_LR=0.65 + SCALAR_LR=0.55 | 1.863295 | 158 | 251K | 30.3% | ✓ **NEW BEST** |

### Key Findings (Aug 02)

1. **HEAD_DIM=32 is worse** — narrower heads gave fewer steps (154 vs 158) and didn't improve val_bpb. The GB10's compute units benefit from wider heads.
2. **ADAM_BETAS=(0.8, 0.95) is better** — standard (0.9, 0.95) betas were worse. Lower beta1 helps with the short training budget.
3. **Smaller model (ASPECT_RATIO=32) is worse** — 198 steps but much worse val_bpb. The model is too small to learn effectively.
4. **Deeper model (DEPTH=5) is worse** — fewer steps (118) despite higher MFU. More params = less steps.
5. **MATRIX_LR=0.04 is optimal** — 0.05 was worse.
6. **SCALAR_LR=0.55 with EMBEDDING_LR=0.65 is best** — small bump from 0.5 → 0.55 helped.
7. **EMBEDDING_LR sweet spot is 0.65-0.7** — 0.65 is best, 0.7 is close, 0.65+0.55 scalar is new best.

### Current Best Config (Aug 02)
```python
ASPECT_RATIO = 64
HEAD_DIM = 64
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = 2**19
EMBEDDING_LR = 0.65  # was 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.55  # was 0.5
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.1
FINAL_LR_FRAC = 0.0
DEPTH = 4
DEVICE_BATCH_SIZE = 8
GRAD_CLIP = 0.3
```

### Next Steps
- Try EMBEDDING_LR=0.65 with SCALAR_LR=0.525 (fine-tune)
- Try WARMUP_RATIO=0.01 (very tiny warmup)
- Try GRAD_CLIP=0.25
- Consider Phase 2: HRM/RDT architecture experimentation
