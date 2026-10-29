# Dory vs nano_mythos — loop-count & depth notes (RA-15/RA-16)

Written 2026-10-29. Sources: NVIDIA model card (fetched 2026-10-29), our SESSION_NOTES (2026-10-18 section), YouTube transcripts (auto-subs).

## What the Dory model card actually says
Source: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 (collection: https://huggingface.co/collections/nvidia/nemotron-labs-dory)
- "Looped (latent-recurrent) normalized-GPT with interleaved sliding-window attention (SWA) and full-attention layers", 4B params.
- "The layer stack is divided into three groups: Input (I), Recurrent (R) and Output (O). The shared Recurrent block (R) is applied multiple times between the Input and Output blocks, and the input is reinjected into every loop. In Dory, the Input and Output blocks have 8 layers each, while the Recurrent block has 20 layers."
- Loop count is an **inference** setting: `n_recurrent_loops` (vLLM) / `dory.num_loops` = 1, 2, 3 or 4 (llama.cpp). "We recommend two loops as the default ... Most of the accuracy gain from looping comes from the second loop."
- "Elastic Inference Compute: A single checkpoint serves different inference compute budgets ... with no retraining."
- `recurrent_kv_cache_mode`: `per_loop` or `last_loop` (one KV cache per recurrent attention layer; "accuracy ... stays close to the per-loop mode").
- Training: 6T tokens (NVFP4 for ~4.5T then BF16), 300B-token context extension to 256K, Megatron-LM; then SFT; then multi-environment GRPO (NeMo RL).
- **Not stated anywhere on the card:** the loop count(s) used during pre-training, whether loops were fixed or sampled per step, any halting/ACT mechanism, loss on intermediate loops, or stability tricks beyond the nGPT backbone. No paper found as of 2026-10-29.

## Inferences (ours, unverified)
- Benchmarks are reported at 1/2/4 loops from one checkpoint, so training must have exposed the model to several loop counts (fixed-per-run training generally degrades at other counts — see the ELT video below). Most likely: loop count sampled from a small set {1..4} during training, or trained at a fixed count with some anytime mechanism. Unknown which.
- No per-token halting: every token gets the same loop count (global, user-chosen).
- Depth ratio: 8 + 20×L + 8 → 36 / 56 / 96 effective layers at L = 1 / 2 / 4. The recurrent block is ~55% of unique layers and is itself deep (20 layers).

## Our design vs Dory
| | Dory 1.0 4B | nano_mythos baseline (ref 2.289139 @1200) |
|---|---|---|
| Prelude / recurrent / coda layers | 8 / 20 / 8 | 2 / **1** / 2 |
| Loop count (train) | not published | Poisson(mu=8) per sequence, cap 8, **plus ACT halting** (avg ~2.3–2.4 loops observed) |
| Loop count (inference) | fixed, user-set 1–4 | up to 8 with ACT halting |
| Per-token adaptivity | none published | ACT halting + MoD-0.5 token bypass per loop |
| Input reinjection | every loop | every loop (LTI injection, Parcae e-norm) |
| Backbone | nGPT (normalized), SWA/full interleave | RMSNorm pre-norm, full attn, Hyperloop HC K=4, Muon |
Key difference: our loop count is doubly stochastic (Poisson depth + learned halting), and RA-09 showed halting collapsing to ~1.5 loops. Dory exposes an explicit global loop count. Our looped block is also very shallow (1 layer) vs Dory's 20.

## What we changed (env-gated, default OFF = identical baseline)
- `NANO_FIXED_LOOPS=N` (RA-15): exactly N loops in train and eval; disables ACT halting (output = state after final loop) and Poisson depth sampling. MoD-0.5 still applies inside each loop.
- `NANO_RECUR_LAYERS=k` (RA-16): k shared transformer layers inside the looped block (default 1). Not wired into the RA-11 KV-recycle path.
- `NANO_PRELUDE_LAYERS`, `NANO_CODA_LAYERS` (default 2/2).
- Startup log prints `fixed_loops= prelude= recur_layers= coda= eff_depth=`.

## Video search (David's half-remembered video)
Method: yt-dlp YouTube search (6 queries) + auto-subtitles; YouTube rate-limited (HTTP 429) several downloads. Wayback not useful without a specific URL. David's X saved links only contain the Dory card link (@NeoAIForecast, 17 Oct).
Best candidate (fits "training with an explicitly specified loop count"):
1. **"ELT: One AI Model, Any Compute Budget — Elastic Looped Transformers Explained"**, AI Study Group, 2026-04-16, https://youtu.be/9FJqFvow2Us (Google DeepMind ELT paper, visual generation).
   - [00:01:26] "If you train a model to loop exactly eight times, it learns to produce good images after exactly eight loops. Run it four times and the output is a blurry mess. Run it 10 times and quality actually degrades."
   - [00:02:05] "During training, ELT runs two paths simultaneously through the same shared parameters. The first is the teacher path. It runs the full number of loops ... The second is the student path. It exits early at a randomly sampled shorter loop count. The student is trained to match both the ground truth labels and the teacher's output." (intra-loop self-distillation, ILSD)
   - [00:04:06] too few unique layers ("just one layer looped many times") hurts quality; looping beyond the trained count degrades.
Other relevant (weaker match):
2. "Can looping make AI smarter?", Liz on AI, 2026-10-23, https://youtu.be/txPPv8gdWBI — on Dory: [00:01:20] "Eight layers in, 20 layers it can run up to four times, and eight layers out ... the loop count isn't a different file, it's one setting you pass it when you start the model."
3. "Looped models explained in 8min..", Latent Chalk, 2026-10-28, https://youtu.be/5uG8grdqmLo — Ouro adaptive exit gate; [00:07:17] "every token gets the same number of loops, but shouldn't easy token really pay as much as a hard one?"
4. "Looped Transformers: Think Longer, Not Bigger", Dushyant, 2026-09-10, https://youtu.be/ydz3FNZhtdc — argues the opposite (learned halting) [00:01:42].
5. "LLMs Don't Need More Parameters. They Need Loops.", NeuroDump, https://youtu.be/pDsTcrRVNc0 — Ouro exit-gate maths [00:09:30–00:12:04].
Not transcribed (429): Dory videos by AI WITH Rithesh (D3YM5OzNt9s, 2026-10-18) and Simply AI (dIBcmz_r8iA, 2026-10-22); Sebastian Raschka "OpenAI Astra and Recurrent Depth / Looped Transformers" (KT4n-z_4QJU, 2026-09-02); "Build Recurrent-Depth Transformers with OpenMythos" (fqVRsu0TAnk).
Uncertain: no transcript matched an LLM pre-training recipe with a fixed loop count verbatim; ELT is the closest and is about image/video models.

## Applicable to our design
- ELT's warning matches our setup: a model trained at fixed N may be poor at other N. RA-15 evaluates at the trained N only (comparable to how R0 is evaluated at max 8).
- Candidate RA-17: ELT-style ILSD — train at fixed N with an extra student pass at a random shorter loop count, distilled to the full-N output. Gives Dory-like "elastic" loop counts without ACT.
- Depth: both ELT and Dory suggest a 1-layer recurrent block is too thin → RA-16 (recur_layers 2–4 at fixed loops).

## Scaling to 1B (groundwork, single Spark)
- GB10: 128 GB unified, ~40 TFLOPS BF16 peak; we see ~18–25% MFU on the nano model.
- 1B params with AdamW/Muon in BF16+fp32 master ≈ 16 bytes/param ≈ 16 GB states; activations dominate with loops (stored per loop unless checkpointed) → use activation checkpointing per loop and fixed loops (static graph, compile-friendly).
- Compute: 6·N·D with effective N = non-recurrent + recurrent×loops. A 1B Dory-shaped model at 2 loops on 20B tokens ≈ 6·1.6e9·2e10 ≈ 1.9e20 FLOPs ≈ ~55 days at 40 TF peak and 100% MFU — so a single Spark realistically supports ~1–2B tokens (days), or a smaller model. Fixed loop count makes this budget predictable; adaptive halting does not.

## Literature scan (2026-07-29 → 2026-10-29): summary
Full report: LITERATURE_SCAN_2026-10.md (200 items; summarised by Nemotron-3.5-Super-VL on Spark 2; relevance scores are inflated, since 152/200 were rated 8 or higher, so treat tiers loosely).
Most relevant to fixed loop counts (unverified LLM summaries; read the papers):
- Adaptive-Depth Is a Finishing Stage (arXiv 2610.14252): train at a fixed loop count for ~67–80% of steps, then switch to sampled depths. This is a direct candidate schedule for RA-18.
- Looping Beyond Twice (2610.01153): recipe for deeper loops: residual scaling, decaying input re-injection, per-loop routers.
- StateLoop (2610.35544): persistent per-token state across loops with a learned gate.
- Shared Memory in Looped Transformers (2610.02383): loop 1 writes the KV cache, later loops read it with a short window; a gradient highway.
- LoopFold (2610.35556): predictor/corrector to parallelise loops (latency; mostly relevant at scale).

## Auxiliary LLM resources (for LLM-heavy side tasks, only when David OKs)
- Spark 2 Nemotron via Hermes: vLLM server http://100.82.91.70:8000/v1, model id nemotron-3.5-super-vl (nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4), the same endpoint Hermes (Jetson) uses. Used for the 2026-10 literature scan with David\x27s OK; do not restart/reload it.
- Spark 1 alternatives: `sparkrun list` then `sparkrun run <modelcode>` boots other models (e.g. Qwen 3.8 27B FP8). Don\x27t boot during a training run.

## Dory config.json + vLLM implementation (published artifacts, read 2026-10-29)
Sources: config.json in the HF repo (cached on Spark 2) and vLLM dory.py (vllm-project/vllm PR #60149, as installed in the Spark 2 vllm_node container).
- num_hidden_layers 72 counts attention and MLP separately: n_input_layers 16 / n_recurrent_layers 40 / n_output_layers 16 sub-layers = 8 / 20 / 8 blocks (matches the card).
- hidden 2560, 16 heads / 4 KV heads, head_dim 256, SwiGLU 7168, vocab 131072, untied embeddings; default n_recurrent_loops 2.
- Attention: sliding window 1024 with every 4th attention layer full (3:1); RoPE theta 1e4 (SWA) vs 1e7 (full); qk-norm after RoPE; sigmoid output gate.
- nGPT: every residual update is a normalised slerp: h = norm(h + |alpha·scale|·(norm(out) − h)), learnable per-channel alpha (ngpt_alpha_init 0.05, scaled by 1/initializer_range 0.0177); learnable logit scale.
- **Loop / reinjection:** state = input_h; for each loop: h = norm(state + input_h); run the 20 recurrent blocks; state = norm(h). Then the output blocks. → additive reinjection of the prelude output with unit-norm states and no learned gate (cf. our LTI injection).
- KV cache per loop ("LoopCoder virtual layer indices") or last-loop.
- Still unpublished: training loop schedule, halting, LR/warmup.
- Testable ideas: (a) NANO_NGPT-style unit-norm states + slerp residual alpha as a loop stabiliser (RA-20 candidate); (b) Dory reinjection norm(state+e) vs our LTI injection (RA-21 candidate); (c) SWA:full 3:1 in the looped block.

See PLAN.md for the overall plan; DORY_SELF_PROBE.md for the self-knowledge probe.
