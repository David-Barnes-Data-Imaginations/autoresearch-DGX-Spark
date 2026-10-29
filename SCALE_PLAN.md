# SCALE_PLAN — nano_mythos looped transformer: small → mid → 1B

Written 2026-10-29. Hardware: DGX Spark GB10, 128 GB unified memory, ~40 TFLOPS BF16 peak each; we measure ~18–25% MFU on the nano model. Spark 2 is used ONLY at Stage 3 and only when David starts it.

## Compute model
FLOPs/token ≈ 6 · (P_nonrec + P_rec · L), where L = loops and P_rec = parameters of the shared block. Training FLOPs = that × tokens. Wall-clock = FLOPs / (40e12 · MFU · n_sparks).
Memory (Muon + AdamW for embeddings/scalars, BF16 weights with fp32 master): ≈ 16 B/param of state. Activations scale with L (each loop stores its activations) → activation checkpointing per loop iteration (recompute the shared block in backward) keeps activation memory ≈ one loop + prelude/coda.

## Stage 0 — nano (now, ~10M params, 1200 steps, ~10–40 min/arm)
Avenues RA-15 → RA-19 (see SESSION_NOTES Next session). Primary comparisons are compute-matched.
Go to Stage 1 when: a fixed-loop / depth config beats the adaptive baseline (2.289139) by ≥ 0.005 val_bpb at matched FLOPs, reproduced on a 2nd seed, OR is equal val_bpb with clearly lower run-to-run variance (the stability goal).

## Stage 1 — small (~50–125M params: dim 512–768, Dory-like 2:5:2 prelude:recur:coda, fixed loops 4–8)
~0.5–2B tokens per run. Example: 100M params with P_rec ≈ 55M, L = 4 → ≈ 6·(45M + 220M) ≈ 1.6 GFLOP/token; 1B tokens ≈ 1.6e18 FLOPs ≈ 11 h at 40 TF × 0.4 MFU on one Spark (~22 h at 20% MFU).
Tests: the Stage-0 winner vs an equal-FLOP non-looped model; loops 4/8 then 12/16 (David's request) ; ILSD/elastic loop counts if RA-17 wins.
Go to Stage 2 when: the looped model beats its equal-FLOP non-looped twin, and the gain does not shrink from Stage 0 → 1 (check the trend at 2 sizes).

## Stage 2 — mid (~300–400M params, one Spark, ~5–10B tokens, a few days)
Confirms scaling trend + stability at higher loop counts (12–16) with activation checkpointing; settles the LR / Muon settings for 1B.
Go to Stage 3 when: the trend holds across 3 sizes (nano, small, mid) and there are no loss spikes at the chosen loop count.

## Stage 3 — 1B (both Sparks, ~30 days, needs David to start Spark 2)
Shape (Dory-like): e.g. dim 2048, prelude 4 / recur 10 / coda 4 layers (~1.0B params, P_rec ≈ 0.5B), fixed L = 4 (or the Stage-2 winner).
FLOPs/token ≈ 6 · (0.5B + 0.5B · 4) = 15 GFLOP/token.
Two Sparks, 30 days, at 30% MFU: 2 × 40e12 × 0.3 × 2.59e6 s ≈ 6.2e19 FLOPs → ≈ 4B tokens. At 20% MFU ≈ 2.8B tokens.
So "1B in 30 days" means roughly 3–4B tokens at 4 loops (about 1/5 Chinchilla-optimal for the unique params, but ~5× more compute per token than a 1B non-looped model). Lower loops (L = 2) → ~1.7× more tokens. Muon kept (≈ AdamW memory or less).
Memory per Spark (data-parallel, full replica): states ~16 GB + activations with per-loop checkpointing at seq 2048 / micro-batch 4–8 ≈ 20–40 GB → fits in 128 GB. Two-node DDP over the existing Spark network link (NCCL); test the link bandwidth before Stage 3.
Kill criteria: loss spikes that need > 2 restarts, or val loss tracking worse than the Stage-2 extrapolation by > 5% at 10% of the budget.

## Pretraining data (Stage 2–3 only; Stage 0/1 keep the current prepare.py data)

### "Paul Graham's initiative from Olmo": ambiguous, confirm with David
- **Most likely meaning: Ai2 / Paul *Allen*.** Ai2 is the Allen Institute for AI, founded by Paul Allen, and it makes OLMo. Its open pretraining data is **Dolma 3**:
  - Dolma 3 Mix (6T): https://huggingface.co/datasets/allenai/dolma3_mix-6T. About 5.93T tokens, 23.7 TB uncompressed. Mix: Common Crawl 76.1%, olmOCR science PDFs 13.6%, Stack-Edu code 6.9%, FineMath-3+ 2.6%, arXiv 0.9%, Wikipedia 0.04%. Licence ODC-By ("intended for research and educational use").
  - **150B-token sample with the same mix: https://huggingface.co/datasets/allenai/dolma3_mix-150B-1025** (the practical choice for us).
  - Pool (9.31T, before mixing): https://huggingface.co/datasets/allenai/dolma3_pool. Mid-training "Dolmino" (100B, math/code/QA/thinking) and "Longmino" (50B, long context) are described at https://github.com/allenai/dolma3.
  - Paper: arXiv:2512.13961 (Olmo 3). As of 2026-10-29 I found no newer Ai2 dataset; Ai2's Oct 2026 release was Olmo-core 3 (training infrastructure), and its blog says the next Olmo will be trained on "our largest dataset".
- **Other readings I checked (no match):** I found no Paul Graham–linked dataset initiative connected to OLMo. Only community scrapes of PG's essays exist (e.g. https://huggingface.co/datasets/sgoel9/paul_graham_essays, about 230 essays, roughly 0.5M words, useless for pretraining). "PG" could also mean Project Gutenberg (public-domain books, part of the older Dolma v1.x mixes). → **Ask David** whether he means Ai2's Dolma 3 (Paul Allen) or something else he saw.

### NVIDIA open pretraining data (Nemotron)
| Dataset | Size | Content | Licence |
|---|---|---|---|
| nvidia/Nemotron-CC-v2 https://huggingface.co/datasets/nvidia/Nemotron-CC-v2 | 6.59T tokens | English CC 3.36T, synthetic CC rephrase 1.26T, Diverse QA 0.69T, translated QA 0.56T, math 0.21T, plus SFT-style sets | NVIDIA Data Access Agreement for Model Training |
| nvidia/Nemotron-CC-v2.1 https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1 | 2.54T new English tokens (~4.6 TB) | organic + translated + rephrased CC (Dec 2025); "use together with" v2 | same agreement |
| nvidia/Nemotron-CC-Math (v1; HQ subset "4plus") | 133B tokens | math from CC, LaTeX-normalised | same agreement (check card) |
| nvidia/Nemotron-CC-Code-v1 | 428B tokens | code pages from CC | same agreement |
| nvidia/Nemotron-Pretraining-Code-v1/v2 | ~340B new tokens (v2) plus metadata | GitHub code (metadata to rebuild) plus synthetic code | same agreement |
| nvidia/Nemotron-Pretraining-Specialized-v1 | (see card) | synthetic STEM reasoning, scientific coding, wiki rewrite | CC BY 4.0 (some subsets CC BY-SA / GFDL) |

Licence notes:
- The cards say the NVIDIA agreement "enables training of any AI model, including models released under proprietary or open source licenses" and that the datasets are "ready for commercial use". But you **may not redistribute** the data or make it subject to an open licence.
- A model trained on them "may be subject to" the Qwen, DeepSeek and Phi-4 licence terms, because those models generated the synthetic portions.
- Dolma 3 (ODC-By) only requires attribution.
- For a private research model, both are fine. If David later wants to release the 1B weights openly, review the Qwen/DeepSeek clauses first.

### Proposed 1B mix (about 3–4B tokens total, from SCALE_PLAN Stage 3; illustrative, to be tuned at Stage 1–2)
- 45% Dolma 3 mix-150B sample (web + science PDFs + code + math, already quality-mixed by Ai2)
- 25% Nemotron-CC-v2/v2.1 high-quality slices (organic HQ plus Diverse QA; skip the translated QA, since our model is English-only)
- 10% Nemotron-CC-Math-4plus / FineMath
- 10% code (Stack-Edu via Dolma, or Nemotron-CC-Code)
- 10% Nemotron-Pretraining-Specialized (STEM reasoning) in the last ~20% of training, as a "mid-training" anneal alongside Dolmino-style data
- Rationale: looped models are pitched as better at reasoning, so we over-weight math, code and STEM compared with Dolma's 76% web.

### Download / storage (Spark 1 disk: 916 GB, 434 GB free on 2026-10-29)
- Don't download whole datasets: 3–4B tokens of text is only ~15 GB raw text, or ~8 GB as uint16 token shards.
- Use `datasets` streaming (or `hf download` with `--include` on specific shards) to sample N tokens per source. Tokenize straight to `.bin` shards on Spark 1, then rsync the shards to Spark 2 (each node keeps its own copy, ~10 GB).
- The 128 GB unified memory is not involved, because tokenized shards are memory-mapped. Budget about 50 GB of disk per node, including raw jsonl.zst staging, which is deleted after tokenizing.
- The tokenizer is a separate decision: the current prepare.py vocab of 8192 is too small for 1B. Plan a 32k–64k BPE trained on a sample of the mix.

### Small-scale data test (before committing)
- Stage 1 data ablation: same model and steps, (a) the current prepare.py data (karpathy/climbmix-400b-shuffle) vs (b) a 1–2B-token sample of the proposed mix. Evaluate both on a held-out slice of each, plus a small downstream set (e.g. HellaSwag/ARC-easy/GSM8K-style few-shot) so the different eval data doesn't confound bpb.
- Go with the mix if it's no worse on (a)'s eval and better downstream.
