# Literature scan: looped / recurrent-depth transformers, 2026-07-29 → 2026-10-29

Generated 2026-10-29 20:56 by scripts/lit_scan.py: arXiv API (cs.LG/cs.CL/cs.AI, 20 queries, keyword-filtered) + HF model search, 200 items. Summariser: nemotron-3.5-super-vl (NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) via the Spark 2 vLLM endpoint used by Hermes (http://100.82.91.70:8000/v1/chat/completions), approved by David 2026-10-29. Loop-related arXiv items were read as abstract + up to 14k chars of arXiv HTML full text (see 'read'). Summaries are LLM-generated and unverified: check the paper before relying on a number.

## Top 5 most relevant

- **LoopFold: Parallelizing Recurrent Depth in Looped Transformers** (http://arxiv.org/abs/2610.35556v1) — Directly targets our goal of reducing wall-clock latency for deeper fixed loops (2–16) on nano_mythos via a predictor/corrector schedule, which matters before the 1B DGX Spark run.  
  *Test:* Implement a LoopFold-style predictor and corrector for the shared recurrent block and compare fixed-loop schedules (e.g., 1+3+3) against standard sequential looping and ACT halting on the 10M model, measuring latency and zero-shot accuracy.

- **StateLoop: Persistent State for Looped Transformers** (http://arxiv.org/abs/2610.35544v1) — Addresses information loss across recurrent iterations, which is central to our deeper fixed-loop and prelude/coda ratio experiments at small scale.  
  *Test:* Add a StateLoop persistent-state branch with a selective SSM update (per-channel lambda and input-dependent Delta) to the shared recurrent block and test fixed loop counts 2–16, comparing identity mixing versus causal attention or Mamba-2 token-axis mixing for perplexity and overhead.

- **The Surprising Effectiveness of Shared Memory in Looped Transformers** (http://arxiv.org/abs/2610.02383v1) — Reduces KV cache memory and acts as a gradient highway, directly relevant to our memory-constrained 1B run and ACT/Poisson training dynamics.  
  *Test:* Implement shared KV cache where recursion 1 writes the full context and recursions 2–8 read it with a short sliding window (ω=64), then compare fixed loop counts 2–16 against the per-recursion cache approach and measure gradient norm at the first recursion.

- **Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts** (http://arxiv.org/abs/2610.01153v1) — Provides stabilization recipes (residual scaling, input re-injection, per-loop routers) for deeper loops, which complements our current Parcae LTI reinjection and MoD routing as we scale to 1B parameters.  
  *Test:* Add per-loop routers and a Looping Residual (EMA of previous loop outputs) to nano_mythos, implement residual scaling γ = λ/(H√M) and input re-injection g_t = λ/(t√M), and compare against current Parcae LTI reinjection across fixed loops 2–16.

- **Adaptive-Depth Is a Finishing Stage** (http://arxiv.org/abs/2610.14252v1) — Offers a compute-efficient fixed-first training schedule that could reduce wall-clock time before our 1B run while preserving adaptive-depth serving performance.  
  *Test:* Train nano_mythos at fixed loop count 2–4 for 67–80% of steps, then switch to uniform loop-depth sampling (1–16) for the finishing phase, and compare NLL and recall across serving depths against Poisson sampling from the start.

## Suggested new avenues

- `if ENABLE_ACCORDION:` Test fixed-prefix layer looping with a contiguous middle window of the shared recurrent block repeating 0–8 times per budget and zero-initialized residual gates, comparing uniform per-micro-budget sampling against Poisson/ACT to see if fixed paths reduce the one-loop tax.  
- `if ENABLE_ASYMLOOP:` Implement an asymmetric variant with a fixed 2-layer prelude encoder producing Global KV caches and a looped recurrent decoder attending to Global and Local KV under one softmax, comparing fixed loops 2–16 against the full-stack looped block at matched parameter count.  
- `if ENABLE_SCALE_RLT:` Implement the hybrid log-probability gate ℓ = ℓ_dense + sg(ℓ_sched − ℓ_dense) for GRPO/PPO and test fixed loops 2–16 with a deferred rollout engine batching deeper loops only when deep readouts are due.  
- `if ENABLE_LOOPCD:` Implement LoopCD-Hidden to contrast the final recurrent state against an early loop state before the coda/head, testing fixed guidance strength ω and using LoopCD-guided shallow targets for early-exit self-distillation.  
- `if ENABLE_INFINILOOP:` Replace the carry-last rule with InfiLoop’s streaming recurrence (weighted-state vector and normalization scalar with learned temporal decay) to prevent latent overthinking during the 1B DGX Spark run.  
- `if ENABLE_LOOPQUANT:` Apply LoopQuant’s CLDC and ILPA for W4A4 quantization across fixed loops 2–16, measuring accuracy retention versus BF16 and calibrating with LCTC-style lightweight calibration aligned to ACT halting trajectories.

## High relevance (8-10)

### LoopFold: Parallelizing Recurrent Depth in Looped Transformers
Feichen Song, Boyi Zeng, Yiqin Hao, Shixiang Song, He Li, Yizhou Yang · 2026-10-26 · http://arxiv.org/abs/2610.35556v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopFold parallelizes recurrent depth in looped Transformers by using a lightweight predictor to estimate inputs for later loops and a corrector to refine outputs using results from earlier loops. It uses a schedule of sequential stages and parallel width to reach a target depth index D with fewer sequential stages than a vanilla looped model. Experiments with 1.4B models show that deeper LoopFold configurations improve zero-shot accuracy from 53.41% to 56.38% and reduce decoding latency by 46% compared to a nine-loop vanilla model.

Key numbers: 1.4B models; accuracy 53.41% to 56.38%; target depth D=9; 5 sequential stages; latency reduction 46%; 2.20 percentage points Avg-8 improvement; 2d^2 parameters for predictor/corrector (8.39M at 1.4B); D = n + S - 1; backbone evaluations 1 + n(S-1)

Ideas for us: Implement a LoopFold-style predictor and corrector for nano_mythos to test if parallelizing recurrent depth reduces wall-clock latency while maintaining accuracy gains from deeper loops. Compare fixed loop counts (2-16) using LoopFold schedules (e.g., 1+3+3) against standard sequential looping and ACT halting on the small 10M parameter model.

### StateLoop: Persistent State for Looped Transformers
Feichen Song, Boyi Zeng, He Li, Shixiang Song, Yiqin Hao, Zitong Wang · 2026-10-26 · http://arxiv.org/abs/2610.35544v1 · relevance 10 · read: abstract+fulltext(14k chars)

StateLoop introduces a persistent state per token for looped Transformers to retain information across iterations that would otherwise be lost in the single hidden state. A learned gate controls the retention of previous state and incorporation of backbone output, with the updated state projected back and added to the backbone output for subsequent iterations and final prediction. The method reduces Pile perplexity and improves zero-shot accuracy over Vanilla Looped, with the advantage growing as trained loop depth increases. Optional causal attention or Mamba-2 token-axis mixers further improve accuracy gains.

Key numbers: 410M and 1.4B scales; 30B-token training budget; 0.18 Pile perplexity reduction at 1.4B; 0.97 percentage points average accuracy improvement at 1.4B; 1.43 and 1.44 percentage points with Attn/Mamba-2 mixers; lambda initialized logarithmically from 0.01 to 4; initial Delta = 0.3; b_delta = log(e^0.3 - 1)

Ideas for us: Add a StateLoop persistent state branch to nano_mythos's shared recurrent block, using the selective SSM update with per-channel lambda and input-dependent Delta, and test fixed loop counts of 2-16 to see if the perplexity advantage scales with loop depth as reported. Compare StateLoop with identity mixing versus causal attention or Mamba-2 token-axis mixing at the 10M parameter scale to determine if the extra mixer overhead is justified for small models.

### ScaleRLT: Exact Log-Probabilities and Fast Deferred Rollouts for Reinforcement Learning on Looped Transformers
Dezhen Lu, Zixiang Ni, Runguo Li · 2026-10-25 · http://arxiv.org/abs/2610.34427v1 · relevance 10 · read: abstract+fulltext(14k chars)

ScaleRLT addresses reinforcement learning for looped transformers by introducing a hybrid log-probability method that ensures bitwise exactness between deferred rollouts and dense learner passes. It uses a no-grad replay through the rollout's schedule for the value and a dense pass for the gradient, maintaining a strict same-weight gate. The paper reports throughput improvements via deferral and CUDA graphs, and demonstrates significant GSM8K accuracy gains using mixed-depth readout policies.

Key numbers: 10M params (project context); 1.47-1.56x rollout throughput (one GPU); 1.18-1.54x (CUDA graphs); 2.71x update shortening (two-node); 18% tokens miss 10^-5 gate (FP32, 4096 tokens); 3.1x10^-4 loss match to float64; GSM8K 23% to 82% (mixed-depth), 34% to 90% (native readout); 100-update runs.

Ideas for us: Implement the hybrid log-probability (ell = ell_dense + sg(ell_sched - ell_dense)) to ensure exact log-prob gates during GRPO/PPO training of nano_mythos. Test fixed loop counts (2-16) with a deferred rollout engine that batches deeper loops only when deep readouts are due, comparing mixed-depth vs native readout accuracy on small reasoning tasks.

### LSTM-Style Looped Transformers with Shared KV Caches
Andreas Kirsch · 2026-10-23 · http://arxiv.org/abs/2610.33360v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper studies an LSTM-style looped transformer that shares a single KV cache across its recurrent core, reducing cache sets from fifteen to four and parameters by 40.4% at width 1024. It uses a two-pass training objective and a donor consistency loss during the final 10% of training to align parallel training caches with recursive inference caches. Results on FineWeb show this removes 81-89% of the continuation-loss penalty relative to full-cache evaluation across widths 512-1024.

Key numbers: 40.4% fewer parameters; 15 to 4 cache sets; 81-89% penalty removal; final 10% of training budget; gamma=0.3; widths 512, 768, 1024; 1.05-4.19B training tokens; P1E2 with K=4, R=3; 7 weight sets, 15 block applications.

Ideas for us: Test a shared KV cache across the looped recurrent block in nano_mythos to reduce persistent cache sets and memory footprint. Implement the donor consistency loss during the final training stage to mitigate the mismatch between parallel training caches and recursive continuation caches.

### Accordion: Elastic-Depth Inference via Fixed-Prefix Layer Looping
Yudong Zhang, Wei Ding, Kaili Zheng, Yuchen Ding, Zhenyu Zhang · 2026-10-23 · http://arxiv.org/abs/2610.32492v1 · relevance 10 · read: abstract+fulltext(14k chars)

Accordion is an elastic-depth MoE Transformer that supports five fixed inference budgets through fixed-prefix layer looping. It repeats selected middle layers consecutively using zero-initialized residual gates, with each micro-batch uniformly sampling one budget for next-token prediction loss without cross-budget consistency objectives. The approach is extended through SFT and RL using parameter-group merging and budget-matched multi-teacher on-policy distillation. After 3.6T tokens of pretraining, increasing inference depth improves the five-category macro-average across 37 tasks by 3.01 percentage points, with the middle budget capturing 89% of the gain using half as many additional layer executions.

Key numbers: 5 inference budgets; L=24 physical layers; window {7,...,18}; menu K={0,3,6,9,12}; 3.6T tokens pretraining; 3.01 percentage points macro-average improvement across 37 tasks; 89% of gain captured by middle budget with half additional layer executions; lambda_seq=1e-4; 50B tokens for Accordion-Base-32K; 2md gate parameters; 8K context; IFEval outperforms four public reference models

Ideas for us: Test fixed-prefix layer looping on nano_mythos by designating a contiguous middle window of the shared recurrent block to repeat 0-8 times per fixed budget, with zero-initialized residual gates on the repeated passes. Compare uniform per-micro-budget sampling against the current Poisson loop-depth sampling and ACT halting to measure whether fixed paths reduce the one-loop tax and improve shallow-path performance at 10M parameter scale.

### How Much Gradient Does Truncated Backprop Lose in Looped Transformers?
Aur Shalev Merin · 2026-10-20 · http://arxiv.org/abs/2610.28871v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper measures the exact gradient lost when truncating backprop through only the last k loops in looped transformers across nine models. It finds that for shrinking signals, lost gradient is approximately rho^k but rho drifts upward at long loop counts, while for growing signals (Ouro), keeping 2 of 4 loops loses 81% of the gradient. The lost gradient also decreases during fine-tuning by 0.07-0.22 within 300 steps, and a trained model's value overstates what truncation removes later.

Key numbers: 9 models tested; Ouro keeps 2 of 4 loops loses 81% gradient; Ouro keeps 1 of 4 loops loses 92%; fine-tuning fall 0.07-0.22 within 300 steps; Parcae 32 to 64 loops change within 0.03; rho=0.8 with 4 loops loses ~41%; rho=0.35 with 4 loops loses ~1.5%; Parcae-140m keeping 1 of 32 loops cost ~0.04 nats; blended S5 settings 0.005-0.016 nats vs 0.18-nat benefit of training loop.

Ideas for us: Measure exact lost gradient per k for nano_mythos fixed loop counts (2-16) using one backward pass to choose optimal truncation depth; test whether blending residual (x <- (1-a)x + aF(x)) slows gradient fade and how much it changes validation loss at 8 loops before fine-tuning.

### Asymmetric Looped Transformers: Fixed Global Representations for an Iterative Decoder
Cheng Liang, Tengfei Zhang, Wenjie DU, Jianyu Wu, Changlian Ma, Shuchen Pu · 2026-10-20 · http://arxiv.org/abs/2610.28489v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Asymmetric Looped Transformers (AsymLoop), which separate single-pass context encoding from recurrent state refinement. A causal encoder produces fixed layer-specific Global KV caches, while a weight-tied recurrent decoder generates evolving Local KV and attends to both under a single softmax. Experiments on FineWeb-Edu across 60M to 1B parameter models show lower validation perplexity and higher downstream accuracy than Full Loop under matched effective depth and training budgets.

Key numbers: 60M to 1B model scales; 1B zero-shot accuracy improved from 49.58% to 50.85%; perplexity improves by 0.98 at 135M and 0.30 at 350M; E = D = L/2 split between encoder and decoder layers.

Ideas for us: Implement an asymmetric looped variant of nano_mythos with a fixed 2-layer prelude encoder producing Global KV caches and a looped recurrent decoder block attending to both Global and Local KV under one softmax; compare fixed loop counts (2-16 loops) between this asymmetric design and the current full-stack looped block at matched parameter count and token budget.

### Truncate or Not? A Scaling Study of Looped Transformers with Truncated Backpropagation
Xuyuan Xiong, Siyu Chen, Zihan Shao, Yue Su, Weiyang Jin, Ping Luo · 2026-10-19 · http://arxiv.org/abs/2610.26030v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper studies looped transformers trained with truncated BPTT across nearly 600 runs, jointly varying model size, tokens, and forward/backward loop counts. It fits a unified scaling law based on effective parameters and finds that full BPTT is favored when both forward and backward loop counts are optimized, while truncation only helps when the forward-loop count is prescribed. The law identifies two fully trained loops as the best looped setting when model size is free, achieving lower loss than a compute-optimal non-looped model with 27% fewer non-embedding parameters and 8% more compute.

Key numbers: 58M–919M non-embedding parameters; nearly 600 training runs; up to 16 forward loops; (2,2) at 344M vs (1,1) at 471M; 2.36 vs 2.37 nats; 27% fewer parameters; 8% more compute; 2.1×10^19 FLOPs; 8D0 tokens; 120M fixed size

Ideas for us: Run matched-compute comparisons at 10M parameters: full BPTT over 2 loops versus truncated BPTT over 4–8 loops, holding total FLOPs constant, to test whether full BPTT through fewer loops beats truncation. Sweep fixed loop counts (2, 4, 8, 16) with full BPTT and measure validation loss and effective parameter gain to validate the paper’s claim that two fully trained loops are the compute-optimal looped setting before scaling to 1B.

### Internalizing Chain-of-thought into Looped Transformers
Kehang Zhou, Pinyan Lu · 2026-10-19 · http://arxiv.org/abs/2610.25998v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper constructs constant-bit-size looped transformers that internalize Chain-of-Thought (CoT) into latent thought via recurrence. It proves that a shared block applied L times per token can simulate a Turing machine, reducing the context window and generated token count by a factor of L. The work establishes the loop count as an exchange rate between visible CoT tokens and internal latent computation, noting that high L creates learning difficulties because intermediate states are not directly supervised by next-token prediction.

Key numbers: L=1; k>=1; K=8kk'; context window O(s(n)/L); total CoT length O(t(n)*s(n)^(1/k)/L); condition s(n)^(1/k)=Ω(L); 2 DGX Spark GPUs; 1B parameter run; 10M parameters; 2-16 loops; 2 layers prelude; 1-layer shared recurrent block looped up to 8 times

Ideas for us: 1. Implement fixed loop counts (Dory-style) ranging from 2 to 16 on the nano_mythos model to measure the trade-off between visible token generation and latent computation depth. 2. Test ELT-style early-exit self-distillation at various loop depths to mitigate the supervision gap for latent steps identified in the paper.

### Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks
Magnus Sesodia, Juil Sock, Philip Torr, Christian Schroeder de Witt · 2026-10-16 · http://arxiv.org/abs/2610.24135v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper compares looped (weight-tied) transformers against untied transformers of matched effective depth on directed permutation traversal, a multi-hop algorithmic task. It finds that looped models solve tasks where untied models fail, forming a "staircase" of intermediate states that advances one hop per iteration, whereas untied models only expose intermediate results in final layers. The authors conclude that weight tying induces an inductive bias toward iterative program learning rather than merely simulating a deeper stack.

Key numbers: N=16 nodes; H∈{1,...,15} hops; max sequential steps min(H,N-H)≤8; block sizes L∈{2,4,8,16}; iterations up to T=16; effective depth D_eff=L·T from 2 to 32; width d=256; 8 heads; 4-layer block iterated 8 times (4×8) solves every seed while untied 32-layer model never does at η=3×10⁻⁴; 5 queries per edge list.

Ideas for us: Run fixed-loop-count ablations (2–16 loops) on dePo-style permutation traversal with our nano_mythos architecture to verify whether each recurrent block iteration advances a fixed number of hops and whether the staircase adapts to iteration budget. Add linear probes after each recurrent iteration and activation patching between iterations to localize which intermediate hop is encoded in the residual stream, comparing against an untied baseline of equal effective depth.

### Causal Latent Workspaces for Looped Transformers
Ilia Zaznov · 2026-10-16 · http://arxiv.org/abs/2610.22796v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces a causal latent workspace Transformer, an anchored, four-round shared-depth architecture where positions write gated vectors into prefix-specific latent slots and retrieve content-addressed summaries before each shared block application. It complements causal self-attention by providing a separate pathway for compact causal prefix summaries while keeping the original input embedding accessible at every round. Experiments across three byte-level corpora and two parameter budgets show the workspace model outperforms five Transformer controls in five of six settings, with all nine seed-labelled comparisons at the 101.71M budget favoring the workspace.

Key numbers: 101.71M parameters; 0.918M parameters; R=4 rounds; S=4 slots; L=512 sequence length; WikiText-103 perplexity 2.7820; enwik8 perplexity 2.7843; TinyStories perplexity 1.6823; improvements of 1.06%, 0.73%, and 1.01% over strongest non-workspace controls; epsilon=10^-6 for normalization; three seeds per condition; five of six corpus-size settings led by workspace.

Ideas for us: Add a causal latent workspace with S=4 slots and m coordinates per slot to nano_mythos, implementing gated prefix writes and content-addressed reads before each recurrent block application while keeping the original input embedding anchored. Test fixed loop counts of 2, 4, 8, and 16 with and without the workspace on a small byte-level corpus to measure held-out perplexity and determine if the workspace improves looped depth efficiency at the 10M parameter scale.

### LoopAhead: Speculating Ahead for Faster Decoding in Looped Language Models
Zixuan Liu, Nan Xue, Rui Bao, Yuhang Xu, Xiaofeng Cao, Jiangchao Yao · 2026-10-15 · http://arxiv.org/abs/2610.21594v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopAhead is a speculative decoding framework for looped Transformers that uses a lightweight lookahead drafter (LAD) to predict multiple future tokens from early-loop states, with intermediate-loop feedback to suspend or resume speculative branches before full-depth verification. It preserves the target model's output distribution and achieves up to 7.82x decoding speedup over autoregressive decoding. The method trains the drafter on teacher trajectories from the full-depth model while keeping target weights frozen.

Key numbers: 7.82x decoding speedup over AR; up to 8 recurrent loops mentioned in project context; 2-16 fixed loop counts in project testing

Ideas for us: Train a tiny LAD on nano_mythos early-loop states (e.g., after 2 loops) to draft K=4 future tokens and measure speedup vs. fixed-depth AR decoding; use intermediate-loop agreement (e.g., loop 4 vs loop 8) to pause unpromising speculative branches and compare wasted compute against LoopSpec-style pipelining.

### LoopQuant: Accurate and Efficient Post-Training Quantization for Looped Transformers
Zhixiong Zhao, Zukang Xu, Xing Hu, Lifeng Liu, Guangyu Sun, Dawei Yang · 2026-10-12 · http://arxiv.org/abs/2610.16368v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopQuant is a post-training quantization framework for Looped Language Models (LoopLMs) that addresses cross-loop activation dynamics, heterogeneous loop sensitivity, and recursive error propagation. It uses Cross-Loop Distribution Canonicalization (CLDC), Interaction-Aware Loop-wise Precision Allocation (ILPA), and Loop-Conditioned Task Calibration (LCTC) to enable low-bit inference. On Ouro-2.6B, W4A4 quantization surpasses W4A8 baselines while reducing model memory by 65.3% and achieving a 1.7x decoding speedup over BF16.

Key numbers: W4A4, W4A8, 65.3% memory reduction, 3.4% additional storage overhead, 1.7x decoding speedup, Ouro-2.6B

Ideas for us: Apply LoopQuant's CLDC and ILPA to nano_mythos to test W4A4 quantization across fixed loop counts (2-16) and measure accuracy retention versus BF16; implement LCTC-style lightweight calibration to align quantized inference with the ACT halting trajectory during early-exit self-distillation.

### Learning to Extrapolate in Depth with Looped Transformers
Huitao Yang, Jianhao Huang, Zixuan Wang, Baharan Mirzasoleiman · 2026-10-10 · http://arxiv.org/abs/2610.14456v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper analyzes looped transformers trained at a fixed depth K and then unrolled to greater test-time depths T on graph connectivity. It proves that training converges to a single effective parameter balancing graph propagation against fitting unresolved structure, creating a finite extrapolation window where too few loops under-propagate and too many create false connections. It derives explicit window endpoints and certified depth gains that grow exponentially as training depth outpaces training hardness, validated on graph reasoning tasks.

Key numbers: K=3 training loops; T=6 test loops; 4-vertex training paths; 16-vertex test paths; 3^L connectivity range after L loops; exponential certified depth gain as training depth outpaces training hardness

Ideas for us: Train nano_mythos at fixed loop counts (e.g., 4, 8) and evaluate on longer-context language tasks to find the extrapolation window where performance peaks before overthinking degrades; run Poisson loop-depth sampling versus fixed-depth training on the 1B DGX Spark run to compare which yields a wider safe test-time depth range and better early-exit distillation targets.

### Adaptive-Depth Is a Finishing Stage
Andrei Piterbarg, Sean McLeish, Dhruv Pai, Benjamin Keigwin · 2026-10-09 · http://arxiv.org/abs/2610.14252v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces fixed-first training for looped transformers, where the model trains at a constant depth for most of the run and switches to uniform depth variation only during a short finishing phase. At 200M parameters, training at fixed depth 2 for 67% of the run reduces modeled compute by 45% while keeping mean serving-depth NLL within 0.5% of uniform training. At 1B parameters, fixed depth 4 for 67-80% of training reduces compute by 31-37% and wall-clock time by 32-39%, with terminal NLL within 0.8% of uniform training.

Key numbers: 200M params: K0=2, 67% fixed phase, 45.3% compute reduction, 0.5% NLL difference; 200M params: K0=2, 80% fixed phase, 54.3% compute reduction, 0.8% NLL difference; 1B params: K0=4, 67% fixed phase, 30.8% compute reduction, 32.1% wall-clock reduction; 1B params: K0=4, 80% fixed phase, 36.9% compute reduction, 38.5% wall-clock reduction; terminal NLL within 0.8%; 200M training: 4,800 updates, 2.52B tokens; 1B training: 20,000 updates, 10.49B tokens; serving depths K=1,...,16, evaluated through K=64.

Ideas for us: Test fixed-first training on nano_mythos by training at a fixed loop count (e.g., 2-4 loops) for 67-80% of steps, then switching to uniform loop-depth sampling (1-16) for the finishing phase to reduce compute. Compare the NLL and recall across serving depths of a fixed-first nano_mythos model against a model trained with Poisson loop-depth sampling from the start to validate if the finishing stage recovers adaptive performance at small scale.

### Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals
Pengxiang Li, Dilxat Muhtar, Di He, Guinan Su, Lu Yin, Shiwei Liu · 2026-10-08 · http://arxiv.org/abs/2610.11570v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces InfiLoop, a loop-native attention residual connection designed to prevent 'latent overthinking' in looped Transformers. It replaces the carry-last rule with a learned aggregation of recurrent states using content-based weighting and temporal decay. An exact streaming recurrence allows for constant memory usage regardless of loop count, enabling performance improvements at tens of thousands of test-time iterations.

Key numbers: 7M parameters, 97.9% Sudoku-Extreme accuracy, 13.6% pass@2 ARC-AGI-2, 20,000+ effective steps, 62.2% TRM peak accuracy, 58.1% TRM accuracy at 1,872 layers, 90.9% InfiLoop accuracy

Ideas for us: Implement InfiLoop's streaming recurrence (weighted-state vector and normalization scalar) to replace the current carry-last rule in nano_mythos. Test if the learned temporal decay and content scoring prevent latent overthinking during the planned 1B-parameter run on DGX Spark GPUs.

### From Chain-of-Thought to Loops: Non-Autoregressive Latent Reasoning via Looped Transformers
Gerard Grau García, Arnau Padrés Masdemont, Niccolò Grillo, Jordi Ros-Giralt, Arash Behboodi, Victor Conchello Vendrell · 2026-10-08 · http://arxiv.org/abs/2610.11472v1 · relevance 10 · read: abstract+fulltext(14k chars)

LLoCoT is a looped latent-reasoning framework that replaces autoregressive latent-thought generation with iterative refinement of a fixed-size latent workspace using a shared transformer. It uses K=64 hidden-state slots, a shared Gaussian head, and mean feedback from non-final iterations to guide subsequent refinement. The method is trained with VAE-derived continuous targets and likelihood-based supervision, then conditions an autoregressive answer decoder on jointly sampled latent thoughts.

Key numbers: K=64, d_h=2048, D=2560, R=2, tau=0.3, 36x faster time to first answer token, 42x lower reasoning-phase latency, 9.2% increased end-to-end throughput

Ideas for us: Test fixed loop counts (2-16) on nano_mythos using a shared recurrent block with slotwise Gaussian heads and mean feedback between loops, mirroring LLoCoT's workspace refinement. Compare ELT-style early-exit self-distillation against LLoCoT's auxiliary intermediate supervision on coding tasks to evaluate if parallel latent refinement improves reasoning latency for small models.

### ResidualQuant: KV Cache Quantization for Looped Transformers with 2-Bit Residuals
Heejun Kim, Junyoung Lee, SangLyul Cho, Dongsu Han, Insu Han, Sehoon Kim · 2026-10-07 · http://arxiv.org/abs/2610.10381v1 · relevance 10 · read: abstract+fulltext(14k chars)

ResidualQuant is a KV cache quantization method specifically designed for looped Transformers. It exploits the high similarity of KV states across recurrent loops by using the final-loop KV states as a reference anchor and storing the remaining loops as low-precision residuals. The method combines least-square scaling, learned rotations, and loop-wise mixed precision to achieve INT2 quantization while retaining accuracy close to BF16.

Key numbers: 80.7% reduction in theoretical KV storage; 13.0% higher accuracy than rotation-based baseline at same memory budget; 2.73x fixed-batch decode throughput improvement on RTX 5090; 2x larger feasible batches; 4.15x peak decode throughput improvement; accuracy increased from 27.8% to 76.0% (matching BF16 baseline of 75.0%) on Ouro-1.4B.

Ideas for us: Implement ResidualQuant for nano_mythos to quantize KV caches across the 2-16 fixed loops, using the final loop as the anchor for INT2 residuals. Compare the memory/throughput impact of ResidualQuant against standard KV quantization during the 1B-parameter run on DGX Spark GPUs.

### Recurrent Looped Transformer
Yifan Zhang, Jichen Feng, Shihan Qin, Sun-Yuan Kung, Mengdi Wang · 2026-10-06 · http://arxiv.org/abs/2610.07591v2 · relevance 10 · read: abstract+fulltext(14k chars)

The Recurrent Looped Transformer (RLT) divides layers between a parallel causal encoder and a recurrent decoder that merges encoder output with the previous token's state at every position. It compares five splits of eight layers on algorithmic tasks, showing that recurrent feedback enables length generalization (e.g., parity to 256 bits and swap-based S5 tracking at 256 operations) where standard Transformers fail. The paper also evaluates feedback intervals, finding per-token feedback is critical for permutation tracking while chunked feedback can preserve parity performance.

Key numbers: 8 total layers; splits 5+3, 7+1, 4+4, 6+2; 100% parity accuracy at 256 bits; 97.30% S5 accuracy at 256 operations; 93.36% mod-5 accuracy at length 63; chunk size B=4 lowers S5 accuracy from 100% to 19.60%.

Ideas for us: Test fixed loop counts (2-16) against RLT-style encoder/decoder splits (e.g., 2 prelude + 6 recurrent) on parity and permutation tracking to see if loop depth mirrors decoder depth. Compare per-token recurrence (B=1) versus chunked feedback (B=4) in nano_mythos to quantify the trade-off between parallel prefill speed and state-tracking accuracy.

### On the Chain-of-Thought Monitorability of Looped Language Models
Han Wang, Ishwar B Balappanawar, Huan Zhang · 2026-10-02 · http://arxiv.org/abs/2610.02741v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper evaluates Chain-of-Thought (CoT) monitorability in Looped Language Models (LoopLMs) across eight MonitorBench tasks. It finds that increasing loop depth can reduce monitorability on specific Logic, Science, and Engineering tasks under stress tests, but looped architecture alone does not make models systematically less monitorable than non-looped models matched by size or depth. The study uses Ouro-1.4B-Thinking, showing drops from 77.9% to 41.8% on Logic and 40.8% to 23.6% on Science when loop depth increases from 4 to 8 under Direct Concealment.

Key numbers: Ouro-1.4B-Thinking: Logic 77.9% to 41.8%, Science 40.8% to 23.6%, Engineering 54.4% to 43.6% (T=4 to T=8, Direct Concealment); 8 tasks from MonitorBench; 5 Cue Answer and 3 Cue Condition tasks.

Ideas for us: Run fixed loop counts (2-16 loops) on nano_mythos using MonitorBench-style Cue Answer tasks to measure if deeper recurrent blocks reduce CoT monitorability in our specific architecture. Compare monitorability between nano_mythos and a non-looped transformer matched by parameter count and effective depth to isolate the effect of the looped design.

### The Surprising Effectiveness of Shared Memory in Looped Transformers
Giovanni Monea, Keshav Ramji, Yousef El-Kurdi, Luis A. Lastras, Yoav Artzi, Nathan Godey · 2026-10-01 · http://arxiv.org/abs/2610.02383v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Memory Sharing for Looped Transformers, where only the first recursion writes a full-context KV cache and later recursions read it while maintaining a short local window. This architectural change reduces context memory by 76-79% compared to standard looped Transformers while improving validation perplexity and downstream accuracy. The shared memory acts as a gradient highway to the first recursion, preventing gradient decay across depth and allowing quality to improve through five recursions where standard looped models plateau after three.

Key numbers: 150M-1B parameters; 5 recursions; 1.12-1.82 perplexity reduction vs standard Transformer; 76-79% less context memory; up to 1.09 nats lower PPL; up to 4.8 percentage points higher accuracy; up to 4.7x less context memory; T=4,096, ω=64, 5 recursions increases attended KV entries by 1.25%; 4.7x cache reduction at T=4,096.

Ideas for us: Implement shared KV cache in nano_mythos where recursion 1 writes full cache and recursions 2-8 read it with a short sliding window (ω=64), then compare fixed loop counts (2-16) against current per-recursion cache approach. Measure gradient norm at the first recursion with and without shared memory reads to test the gradient highway hypothesis in our ACT/Poisson loop-depth setting.

### Decoding Looped Transformers Better for (Almost) Free
Weihao Liu, Huangjie Zheng, Tianrong Chen, Rohit Dilip, Richard He Bai, Yizhu Jiao · 2026-10-01 · http://arxiv.org/abs/2610.02185v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopCD is a training-free contrastive decoding framework for looped Transformers that uses intermediate recurrent states as weak references to guide the final prediction. It operates in either logit space (LoopCD-Logits) or hidden-state space (LoopCD-Hidden), with the latter adding zero output overhead. The method enables halving recurrent loops while matching or exceeding full-depth baselines, reducing forward FLOPs by 22.5% to 48.2%.

Key numbers: Ouro-2.6B-Thinking AIME 2024 pass@1: 61.88% to 73.33%; Huginn HumanEval pass@1: 22.56% to 31.71%; forward FLOPs reduction: 22.5% to 48.2%; recurrent depth halved.

Ideas for us: Implement LoopCD-Hidden on nano_mythos to contrast the final recurrent state against an early loop state (e.g., loop 1 vs loop D) before the coda/head to improve decoding without extra output passes. Test fixed guidance strength omega and early-exit self-distillation by using LoopCD-guided targets from shallow loops to train deeper loop counts or prelude/coda ratios.

### Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts
Di He, Pengxiang Li, Da Chang, Qingyan Meng, Lu Yin, Shiwei Liu · 2026-10-01 · http://arxiv.org/abs/2610.01153v1 · relevance 10 · read: abstract+fulltext(14k chars)

LOOM addresses the failure of looped MoE to scale beyond two loops by identifying two obstacles: the curse of depth (variance growth from residual accumulation) and expert selection collapse (routers selecting the same experts across loops). It stabilizes recurrence using residual scaling (γ = λ/(H√M)) and input re-injection (g_t = λ/(t√M)), and diversifies computation via per-loop routers and a Looping Residual. Experiments from 100M to 1.7B parameters show stable scaling to 9-12 loops, with a 700M model peaking at 5 loops under iso-FLOP and a 1.7B model peaking at 9 loops without FLOP matching.

Key numbers: 100M-1.7B parameters; 9-12 loops; 700M model best at 5 loops (PPL 18.36 to 16.54, accuracy 38.84% to 39.53%); 1.7B model peaks at 9 loops (PPL 9.62 to 7.77, accuracy 42.4% to 47.7%); γ = λ/(H√M); g_t = λ/(t√M); 60B training tokens.

Ideas for us: Test per-loop routers and a Looping Residual (EMA of previous loop outputs) in nano_mythos to prevent expert selection collapse during fixed-loop testing (2-16 loops). Implement residual scaling γ = λ/(H√M) and input re-injection g_t = λ/(t√M) to stabilize deeper recurrent blocks and compare against the current Parcae LTI reinjection.

### Rotate the Loop: Blockwise Conformality for Looped Transformers
Alexey Zemtsov, Albina Klepach, Vadim Lomshakov, Andrei Polubarov, Nikita Lyubaykin, Alexander Derevyagin · 2026-09-30 · http://arxiv.org/abs/2610.14030v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper proposes blockwise conformal injections for looped transformers to preserve state readability across recurrent steps. It argues that the transition operator between steps should be a scaled rotation (conformal) rather than a free diagonal, splitting the state into groups with their own timescales. Experiments show improved last-token accuracy on state-tracking tasks and better language modeling perplexity and LAMBADA scores.

Ideas for us: Replace the current LTI input reinjection with a blockwise conformal transition operator that rotates residual groups at different rates. Test if this improves state tracking on fixed loop counts (2-16 loops) and enables better extrapolation to deeper loops than trained.

### Scaling Laws for Looped Mixture of Experts
Yanbei Chen, Anirudh Goyal, Raghuraman Krishnamoorthi · 2026-09-30 · http://arxiv.org/abs/2609.40316v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Loop Scaling Laws, a unified scaling law that jointly models model size, data, recurrence, and sparsity for looped Mixture-of-Experts (MoE) models. It proposes a bounded, sparsity-conditional recurrence mapping where each recurrent pass adds effective parameters with diminishing returns, approaching a finite asymptote rather than growing unbounded. The law predicts held-out loss more accurately than prior linear or power-law mappings and recovers standard dense and MoE scaling laws as special cases. Empirical results show that sparsity delivers ~3x active-parameter efficiency and recurrence yields ~2x total-parameter efficiency on reasoning benchmarks.

Key numbers: ~3x active-parameter efficiency (sparsity), ~2x total-parameter efficiency (recurrence), 0.3B active/1.3B total looped MoE matches 0.6B active/2.9B total non-looped MoE, RMSE for bounded mapping: 0.0092, RMSE for power law: 0.0313, RMSE for linear: 0.2566

Ideas for us: Test if adding MoE routing to the recurrent block of nano_mythos sustains the effective capacity gain across more loop counts (2-16) as predicted by the bounded mapping. Compare the held-out loss of fixed loop counts against the paper's bounded, power-law, and linear recurrence mappings to determine the optimal loop-depth for the 1B-parameter run.

### Shared Weights, Selected Computations: How Looped Transformers Route What Each Loop Does
Jiaju Wu, Yi Hu, Muhan Zhang · 2026-09-30 · http://arxiv.org/abs/2609.39892v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper investigates whether shared weights in looped Transformers perform the same computation at every loop. It demonstrates that a frozen loop can be steered toward different transitions by modifying the entering hidden state via a learned linear layer J, with attention patterns acting as the causal routing mechanism. It also shows that intermediate supervision during training affects which computations J can induce, suggesting J selects existing backbone computations rather than creating new ones.

Key numbers: d=8; 10-node cycles; D8L8 (8 loops); D8L6 (6 loops); five matched pairs of graph models; 2-layer Transformer block

Ideas for us: Test if a learned linear layer J at the nano_mythos loop boundary can steer the shared recurrent block toward different computational modes (e.g., varying progress rates) without changing shared weights. Use activation patching of attention patterns between different loop-depth runs to verify if attention routing is the causal pathway for computation selection in our MoD/Hyperloop architecture.

### LoopVL: Recurrent Visual Intelligence
Zhe Qian, Ziyang Gong, Zhongxing Xu, Hehan Li, Zhonghua Wang, Hanmeng Liu · 2026-09-29 · http://arxiv.org/abs/2609.38426v2 · relevance 10 · read: abstract+fulltext(14k chars)

LoopVL extends Loop Transformers to vision-language models by combining Module-Loop and Model-Loop computation to iteratively update a unified vision-language state using shared parameters. It employs a nested L/H update scheme where a low-level module is invoked repeatedly under a stable high-level state, followed by a high-level update. The model uses selective backpropagation across recurrent invocations and demonstrates that shared parameters can support deeper multimodal computation over evolving states.

Key numbers: 16 Transformer layers per L/H stack; default H2L3 schedule; 128 Transformer-layer calls per forward execution; 1024-dimensional Penguin Vision Encoder features; 1536-dimensional projector output; LoopVL-1B model size; 2 model loops; 3 L invocations per loop.

Ideas for us: Test a nested L/H loop schedule (e.g., 3 L updates per 1 H update) on nano_mythos to see if hierarchical recurrence improves reasoning over flat looping. Implement selective backpropagation warmup (starting gradients from final invocations and expanding backward) to reduce memory during fixed-loop-count experiments.

### Thinking in Depth, Speaking Directly: Recurrent Latent Reasoning for Paralinguistically Grounded Spoken Dialogue
Shengbo Cai, Yuxiang Wang, Jingran Xie, Zhisheng Zhang, Shun Lei, Di Cao · 2026-09-29 · http://arxiv.org/abs/2609.37818v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopSLM is a speech language model that uses a weight-shared looped Transformer (L9–11) for recurrent latent reasoning in decoder depth, avoiding autoregressive CoT tokens at inference. It employs ReCue for per-pass acoustic re-anchoring and a two-stage CoT-to-response curriculum: Stage 1 trains on speech-grounded CoT, Stage 2 freezes the Loop/ReCue and trains a Coda for direct response generation. Training uses R=4 passes, increasing executed depth from 28 to 37 layer calls without duplicating weights. On EchoMind, LoopSLM-R4 achieves 63.90/63.38 reasoning accuracy with 34.4 mean tokens and half the latency of CoT-SFT, outperforming Qwen2.5-Omni-7B and Qwen3-Omni-Thinking on empathetic reply metrics.

Key numbers: R=4 passes; 28-layer Qwen2.5-Omni-7B base; executed depth increases from 28 to 37 layer calls; LoopSLM-R4 reasoning accuracy 63.90/63.38; mean tokens 34.4; 64.5% fewer tokens; half the latency; 34x lower latency than Qwen3-Omni-Thinking; >20 points reasoning accuracy gain over CoT-SFT; paralinguistic understanding 70.10/64.97; reply quality C1-C4: 4.80/4.75, 4.62/4.76, 4.21/4.54, 4.54; MMSU 64.72, MMAU-Pro 56.09; Loop L9–11 vs L15–17 improves understanding by 6.29 and reasoning by 3.57 points; 16 learned queries; acoustic memory depths {8, 16, 24, 32}.

Ideas for us: Run a fixed-loop-count ablation (Dory-style, 2–16 loops) on nano_mythos using a shared recurrent block with zero-initialized gating and per-pass modulation, measuring whether reasoning quality saturates at a specific depth and how prelude/coda ratios affect stability. Implement a two-stage curriculum on a small text-only subset: Stage 1 SFT on explicit CoT traces with the recurrent block active, Stage 2 freeze the recurrent block and train only a direct-response readout/coda, then compare inference latency and token count against a single-stage CoT-SFT baseline.

### Looped Transformers as Optimizers
Yulong Huang, Chen Jiang, Zhanpeng Zhou, Hongtao Zhang, Tianyu Li, Tianyu He · 2026-09-29 · http://arxiv.org/abs/2609.37379v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper presents a unified optimization framework for looped Transformer transitions, treating the recurrent state as a fast weight updated across depth. It derives loop transitions from a projection, local objective, and optimizer update rule, identifying mismatches in existing models. The authors introduce OperLoop, which combines explicit weight decay, adaptive step size, and a delta objective to improve performance under matched training FLOPs.

Key numbers: 2-16 loops (from user context, not text); 10M and 1B parameters (from user context, not text); 2 DGX Spark GPUs (from user context, not text)

Ideas for us: Implement OperLoop's explicit weight decay and adaptive step size in nano_mythos's recurrent block to stabilize fixed loop counts (2-16). Test ELT-style early-exit self-distillation by comparing hidden states at intermediate loop depths against the final state using the paper's projection-objective framework.

### LoopICL: Looping a single transformer block to solve tabular tasks
Amir Rezaei Balef, Katharina Eggensperger · 2026-09-28 · http://arxiv.org/abs/2609.36108v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopICL is a looped transformer for tabular in-context learning that uses a single shared block to decouple parameter count from computational depth. It employs two coupled streams (cell and row) refined through within-column and cross-column attention, with a learned exit-gate for automatic halting. The model matches or exceeds larger TFMs like TabICLv2 while using nearly 90% fewer parameters at the same FLOPs cost.

Key numbers: 90% fewer parameters; 9-times larger TFMs matched/exceeded; 2 coupled streams (cell and row)

Ideas for us: Test a learned exit-gate for our fixed loop counts (2-16) to enable adaptive halting during inference; implement distributional input encoding (marginal/discriminative histograms and Fourier quantile encodings) to stabilize training for our nano_mythos recurrent block.

### How to Loop MoE: Flatten the Experts, Untie the Attention
Shouren Wang, Chuang Ma, Mohsen Hariri, Debargha Ganguly, Wang Yang, Xiaoqing Tong · 2026-09-28 · http://arxiv.org/abs/2609.35751v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Foil, a method for looping Mixture-of-Experts (MoE) models by flattening expert layers (reducing layers, increasing experts per layer and passes) and untying attention across passes. It holds expert parameters and compute per token fixed while increasing the routing pool and equivalent expert count. Experiments show Foil lowers pretraining loss compared to unflattened looped baselines and improves routing confidence and expert usage.

Key numbers: 20B tokens, 100B tokens, 0.012 nat loss improvement, 553.7M parameters, 490.8-520.2M parameters, k=2 experts, shapes (8,8,2), (16,4,4), (32,2,8), (64,1,16), 255,500 probe tokens

Ideas for us: Test Foil's flattening strategy on nano_mythos by comparing a looped block with fewer layers and more experts per layer against the current configuration at fixed parameter count. Implement untied attention for each loop pass in nano_mythos to measure if it improves routing confidence and expert balance compared to shared attention.

### Improving Test-Time Scaling with Adaptive Looped Transformers
Yichen You, Tianyu Fu, Aosong Feng, Xingtai Lv, Xuefei Ning, Ning Ding · 2026-09-28 · http://arxiv.org/abs/2609.35748v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper studies test-time scaling of looped transformers, finding that existing looped models often have steeper accuracy-compute slopes but underperform non-looped baselines at matched compute. It proposes TaH2, which jointly post-trains a backbone and an iteration decider using lookahead depth supervision to allocate extra iterations only to tokens that benefit from them. TaH2 improves both efficiency and attainable accuracy, with gains growing as maximum iteration depth increases from 2 to 8.

Key numbers: AIME slope: 2.74 vs 1.79 (53% improvement); matched compute accuracy gain: ~3.4 points; gain over baseline at depth 2: +2.8 points; gain at depth 8: +3.9 points; Ouro tokens unchanged: 52.3%; Ouro tokens worse: 21.7%; Huginn tokens unchanged: 33.5%; Huginn tokens worse: 15.9%.

Ideas for us: Implement a token-level iteration decider for nano_mythos using online lookahead depth supervision (labels based on whether further iteration reduces prediction loss) to replace or augment ACT halting. Run a fixed-loop ablation (2-16 loops) comparing uniform depth vs. adaptive token-dependent depth on a small reasoning subset to measure accuracy-compute slope.

### Shallow Queries, Mature Values: Depth-Asynchronous Self-Speculation for Looped Transformers
Guanghao Li, Zihan Su, Hao Yu, Jinyang Jiang, Tao Ren, Zehao Li · 2026-09-28 · http://arxiv.org/abs/2609.34538v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Depth-Asynchronous Self-Speculation (DAS) for looped Transformers, which decouples the recurrent depth of draft computation from the depth of verified-prefix representations it reads. It finds that queries and keys mature earlier than values, and introduces Mature-V, a training-free primitive allowing shallow draft queries to retrieve full-depth prefix values. DAS-Wave combines this with carried parallel refinement and an independent full-depth verifier to achieve 4.00-6.96x throughput speedup.

Key numbers: 4.00-6.96x mean throughput speedup; Q/K/V cosine similarity to final depth on Huginn: 0.79/0.84/0.57; Q/K/V cosine similarity to final depth on Raven-Llama: 0.96/0.97/0.79; Huginn draft-target agreement at r=1: 47.22% to 66.93% with deep V; Raven-Llama agreement: 83.99% to 87.96% with deep V.

Ideas for us: Implement Mature-V in nano_mythos by allowing shallow ACT-halting drafts to read full-depth cached prefix values during MoD routing and Hyperloop residual updates. Test fixed loop counts (2-16) with DAS-Wave's carried parallel refinement to measure if shallow queries with mature values improve early-exit self-distillation accuracy without increasing recurrent passes.

### Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness
Nux Li · 2026-09-25 · http://arxiv.org/abs/2609.30820v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper identifies two failure modes in quantizing looped transformers: feedback exposure, where quantization error enters the recurrent state without an identity path and is amplified over steps, and calibration blindness, where one-step GPTQ uses step-0 activations and misses later recurrence directions. It shows that accumulating the GPTQ Hessian across recurrence steps outperforms RTN and one-step GPTQ across nine checkpoints from seven looped architectures, recovering bf16-level accuracy on Huginn.

Key numbers: Huginn-3.5B: 3.5B parameters, effective depth of 132 layers at 32 recurrences; Ouro-1.4B and Ouro-2.6B: 96 and 192 block applications over four loop steps; GSM8K diagnostic exact match drops from 34% (bf16) to 0% (per-channel INT4); adapter-only INT4 agreement 10.5% vs core-only 87.9%; removing prior state error raises agreement from 10.4% to 69.2%; mean spectral radius 1.064 (bf16) vs 1.145 (INT4 adapter); one-step GPTQ worse than RTN on 5 of 9 checkpoints.

Ideas for us: 1. Accumulate the GPTQ Hessian across all loop steps (rather than just step 0) when quantizing nano_mythos to mitigate calibration blindness. 2. Compare quantization sensitivity of the non-residual loop-entry adapter versus the residual core in nano_mythos, and test if protecting the adapter (e.g., higher precision) prevents error feedback collapse at fixed loop counts of 2-16.

### FlashLoop: Fast and Memory-Efficient Looped Transformers via Lazy Updates
Wanqi Yang, Shiwei Liu · 2026-09-24 · http://arxiv.org/abs/2609.29812v2 · relevance 10 · read: abstract+fulltext(14k chars)

FlashLoop is a training-free inference framework for looped Transformers that reduces cross-loop redundancy through token-sparse updates, sparse attention, and KV-residual quantization. It claims lossless accuracy with up to 1.64x end-to-end speedup and up to 6x KV-cache memory reduction across several looped models. The method exploits the observation that hidden-state changes, attention-column importance, and KV residuals become increasingly sparse and stable as recurrence progresses.

Key numbers: 1.64x end-to-end speedup, 6x KV-cache memory reduction, 5.8x KV cache memory reduction, 1.8x 8K prefill FLOPs reduction, 2-16 loops (project context), 32K-token prompt prefill time 27s for Ouro-2.6B vs 3s for LLaMA-3.1-8B, 48 GiB vs 4 GiB KV cache

Ideas for us: Apply FlashLoop's token-sparse updates and KV-residual quantization to nano_mythos fixed-loop runs (2-16 loops) to measure speedup and memory savings on our 10M-param model. Test whether sparse attention column selection from the previous loop preserves accuracy during ELT-style early-exit self-distillation at deeper recurrent block depths.

### Statistical Gains from Looped Estimation under Parameter Budgets
Xinyu Tian, Xiaotong Shen · 2026-09-22 · http://arxiv.org/abs/2609.25778v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper establishes that looped estimators, which reuse a single update function across iterations, can achieve vanishing worst-case risk at fixed parameter budgets where untied estimators remain bounded away from zero. It proves that looped residual feedforward networks and post-layer-normalized Transformers attain the minimax polynomial rate for Hölder-smooth targets up to logarithmic factors. The authors derive a tradeoff where additional loop iterations reduce approximation error but increase estimation complexity, providing a theoretical basis for parameter-efficient deep computation.

Key numbers: 2-16 loops (project context), 10M parameters (project context), 1B parameters (project context), 2 DGX Spark GPUs (project context), T_n = ceil(n^(1/(a+1)) * (log(en))^(a/(a+1))), a = 2s/d, risk rate (log(en)/n)^(2s/(2s+d)), untied budget requirement n^(d/(4s+2d))

Ideas for us: Run fixed loop-depth sweeps (Dory-style 2-16 loops) on nano_mythos to empirically validate the approximation-estimation tradeoff described in the paper. Compare ACT halting and Poisson loop-depth sampling against fixed loop counts to determine if dynamic halting provides statistical gains over fixed-depth computation under the same parameter budget.

### Trading Depth for Time in Recurrent Transformers
Zeyi Huang, Xuehai He, Yelong Shen, Yong Jae Lee · 2026-09-18 · http://arxiv.org/abs/2609.21605v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper compares physical depth scaling with temporal recurrence using Latent Recurrent Transformers (LRT). It inserts latent thought tokens between vocabulary tokens to refine hidden states through a shared backbone, matching the block count of doubling depth while using shared parameters. Results show that one thought token recovers 89% of the gains from doubling LRT depth with approximately 48% fewer parameters, and cross-token feedback is identified as a key contributor to performance.

Key numbers: 16- and 20-layer NanoChat MoE backbones; 2L blocks per vocabulary token; gap to double-depth narrows from 0.014 to 0.001 bits per byte; ~46% fewer parameters; ~half as many unique training tokens; 89% of gains from doubling depth recovered; ~48% fewer parameters; feedback projections add ~5% to parameter count.

Ideas for us: Test if a single latent thought token (shared recurrent block pass) can replace adding physical layers in the nano_mythos prelude/recurrent block while maintaining loss. Compare fixed loop counts (2-16) against inserting thought tokens between vocabulary tokens to find the most parameter-efficient depth allocation for the 1B-parameter run.

### Storing Is Not Remembering: LSTM-UT and Bounded Gated Memory for Looped Transformers
Aras Kavuncu, Muhammad Burhan Hafez · 2026-09-17 · http://arxiv.org/abs/2609.19521v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper compares recurrent-depth Transformers, showing that CoTFormer's expanding cache is less effective than a Block Universal Transformer (BUT) for depth extrapolation and delayed recall. It introduces LSTM-UT, which adds a bounded, gated cell state to the shared block to preserve information across iterations. LSTM-UT trained to depth 12 maintains 99.7% exact-row accuracy at depth 60 and above 99.95% at depth 1,000, while also outperforming baselines in OpenWebText2 language modeling with faster training per step.

Key numbers: 99.7% exact-row accuracy at depth 60; >99.95% at depth 1,000; 90.4% cell accuracy at horizon 12 (vs 79.5% BUT, 60.0% CoTFormer); CoTFormer needs up to 91% more training time per step; LSTM-UT within 0.6 perplexity of parameter-matched CoTFormer; 6.5% fewer parameters than BUT yet reaches 88.9% recall.

Ideas for us: Add a bounded, gated LSTM cell state to the shared recurrent block of nano_mythos, initialized to zero per input and updated via input/forget/output gates, with an additive output gate read that leaves the Transformer proposal intact. Run fixed loop-count (Dory-style) depth extrapolation tests at 2-16 loops on Rule 30 or synthetic delayed-recall tasks to measure whether the gated cell improves stability and recall over the current Poisson ACT/Parcae reinjection setup.

### How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents
Zixi Chen, Akshay Vegesna, Samip Dahal, Andrew Gordon Wilson · 2026-09-16 · http://arxiv.org/abs/2609.19107v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper demonstrates that architectural interventions, specifically looping (recursive depth) and model growth, can modify scaling exponents in pre-training, leading to power-law improvements in performance as computation increases. It compares vanilla transformers, looped transformers with tied weights, untied looped models, and models using a boundary operator. The authors find that increasing the number of loops during training improves compute efficiency, particularly in data-constrained, multi-epoch settings where looping acts as a regularizer.

Key numbers: 7.4B model growth matches GPT-3 13B with ~20x less compute; 1.55x compute multiplier for untied growth at 10^20 FLOPs; 1.36x compute multiplier for tied looping at 10^20 FLOPs; 1.25x compute multiplier for boundary operator at 10^20 FLOPs; 2.2x less compute for looping vs weight-decay-tuned standard transformer in multi-epoch setting.

Ideas for us: 1. Test if increasing fixed loop counts (Dory-style) during training phases improves the scaling exponent for nano_mythos compared to a static loop count. 2. Implement the boundary operator (normalizing and injecting the prelude block) into the nano_mythos architecture to measure if it provides a local compute efficiency gain over the current LTI reinjection.

### LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers
SangLyul Cho, Langqing Cui, Sehoon Kim, Dongsu Han, Insu Han · 2026-09-15 · http://arxiv.org/abs/2609.17184v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopSpec is a training-free self-speculative decoding framework for Looped Transformers that extracts draft tokens from intermediate recurrent states to accelerate inference. It uses a pipelined approach to overlap draft generation of future tokens with target verification of the current token, including a selective second proposal from deeper recurrent depths with a gating mechanism. The method ensures lossless decoding under both greedy and sampling regimes and derives optimal proposal depths in closed form based on power-law decay of intermediate-to-target TV distance.

Key numbers: 6.83x inference speedup, up to 1.7x outperforming DFlash, seven checkpoints from two Looped Transformer families, recurrent depths d1=1 and d2=2 mentioned in Figure 1

Ideas for us: Implement LoopSpec-style pipelined self-speculative decoding for nano_mythos using early recurrent states (e.g., loop 2-4) as draft tokens for fixed loop counts (8-16), with a second proposal from deeper depths and gating based on disagreement. Measure speedup and acceptance rates on the 10M param model across different fixed loop depths (2, 4, 8, 16) to validate the closed-form optimal depth derivation before the 1B parameter run.

### Right Direction, Wrong Step: Geometric Analysis of Finite-Step Failure in Looped Transformers
Zhihao Guo, Zonghan Wu, Haizhou Du, Huan Huo, Yilei Shao, Athanasios V. Vasilakos · 2026-09-15 · http://arxiv.org/abs/2609.16665v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper analyzes finite-step failures in looped Transformers, where a locally improving update direction becomes harmful when the full displacement is taken. It uses pathwise curvature decomposition and local quadratic models to show that step scale, not just direction, determines task progress. Experiments show a fixed quarter step recovers positive utility in 72.2–83.2% of selected failures across four settings.

Key numbers: 72.2–83.2% recovery rate, 9 model–task conditions, 4 settings, quarter step (α=0.25)

Ideas for us: Test whether applying a fixed contraction factor (e.g., α=0.25) to the recurrent block's displacement in nano_mythos improves reference utility at fixed loop counts (2-16). Compare ACT halting depths against a quadratic-model-selected step scale to see if shorter, more frequent updates reduce finite-step failures in deeper recurrent blocks.

### T-LoopFormer: Token-Level Elastic-Depth Looped Transformers with Dynamic Routing
Mingqian Yu, Wenpeng Zhang, Shaobo Cui, Peilin Zhao · 2026-09-14 · http://arxiv.org/abs/2609.15160v3 · relevance 10 · read: abstract+fulltext(14k chars)

T-LoopFormer introduces token-level elastic-depth looped Transformers using a dynamic token-choice router that lets each token adaptively determine its own number of loop iterations based on its hidden state. It also employs a recursion-wise KV cache that maintains independent key-value caches for each recursion loop of active tokens, restricting attention to locally cached states to accelerate autoregressive decoding. Experiments show robust performance on language modeling and zero-shot reasoning tasks with the lowest decoding latency among baselines.

Key numbers: Average recurrent depth on FineWeb-Edu validation set: 3.69; loop-depth range: 2-16 loops (from project context); model scale: ~10M params (nano_mythos) and planned 1B run.

Ideas for us: Implement a lightweight token-choice router in nano_mythos to replace Poisson loop-depth sampling, allowing each token to exit early or continue recursing based on its hidden state; add recursion-wise KV cache to the shared recurrent block to measure decoding latency improvements during fixed loop-count testing (2-16 loops).

### Looped Transformers under the Jacobian Lens: Does the Global Workspace Survive Recurrence?
Wenlong Wang, Fergal Reid · 2026-09-01 · http://arxiv.org/abs/2609.01924v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper applies a Jacobian lens to looped and depth-recurrent transformers (Ouro-2.6B and Huginn-0125) to determine if a global workspace emerges through recurrence. It finds that a workspace does form in iterated architectures, but access differs: Ouro reconstructs content every loop requiring interventions across all remaining loops, while Huginn carries content forward within a sliding window of roughly two recurrences. The study uses a virtual-unrolling adapter to treat weight-tied graphs as deep feedforward stacks for lens fitting.

Key numbers: Ouro-2.6B: 48 layers looped 4 times (virtual depth 192); Huginn-0125: 4-layer core recurred 16 times (virtual depth 68); Qwen3.6-27B: 64 untied layers; Huginn transport horizon: ~2 recurrences; Huginn verbal report: 98% self-computed, 0/97 externally injected; cosine similarity of matched offset lenses: 0.9991.

Ideas for us: 1. Test if the global workspace in nano_mythos survives across loop boundaries by applying Jacobian lens transport maps between virtual layers at different loop counts (2 vs 8). 2. Compare the effect of interventions (ablations/steering) applied at the start of a loop versus the end of a loop to see if content must be re-injected every iteration (Ouro-style) or persists (Huginn-style).

### SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers
Shaowen Wang, Ge Zhang, Kairong Luo, Yuhao Wu, Shaofan Liu, Jiaheng Liu · 2026-09-01 · http://arxiv.org/abs/2609.01343v3 · relevance 10 · read: abstract+fulltext(14k chars)

SMELT studies looped Mixture-of-Experts Transformers while matching per-token FLOPs, total non-embedding parameters, and KV cache to isolate architectural advantage from extra compute. The recipe loops the middle 50% of layers twice, narrows hidden dimensions, increases expert count to recover parameters, and scales looped residuals by 1/sqrt(2) with adjusted head size and GQA ratio. Scaling up to 54B non-embedding parameters, SMELT saves 6.8-18.0% of training FLOPs on the compute-optimal frontier and outperforms baselines on downstream benchmarks, especially Code and in-context learning. Mechanistically, the second visit amplifies the first residual updates and reduces attention sink, redirecting mass toward content-relevant tokens.

Key numbers: 54B non-embedding parameters max scale; 6.8-18.0% training FLOPs saving on compute-optimal frontier; 14.7% less compute at C=10^21 FLOPs to reach same loss; middle 50% of layers looped twice; residual scaling 1/sqrt(2); sparsity S≈97%; 1.6B scale downstream results; r recurrences ≈ r^0.46 unique-block equivalents (Schwethelm et al. comparison)

Ideas for us: Run fixed loop-count ablations (2-16 loops) on nano_mythos with MoE routing and matched FLOPs/params/KV, comparing middle-half vs full-stack looping at 2 loops to test SMELT's core recipe at small scale. Implement ELT-style early-exit self-distillation across loop depths with Parcae LTI reinjection, measuring whether the second visit amplifies aligned residual updates and reduces attention sink as observed in SMELT's mechanistic analysis.

### Dynamical phase selection controls compute scaling in looped transformers
Gunn Kim · 2026-08-27 · http://arxiv.org/abs/2608.26556v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper demonstrates that looped transformers with identical architecture and accuracy can fall into distinct dynamical phases (fold, Neimark-Sacker, rigid) based on initialization. In the fold phase, critical slowing down follows a parameter-free normal-form relation, producing a workload tail where the probability of exceeding N iterations scales as N^-2. Test-time compute is therefore governed by the dynamical phase of the trained solution rather than architecture alone.

Key numbers: 15 networks; 10 initializations at μ=0.50 (5 folds, 1 Neimark-Sacker, 1 rigid, 3 monostable); 8 of 10 networks in final phase by step 250; tail exponent P(τ>N) ~ N^-2; Π(ε) → π = 3.1416 (measured 3.146 ± 0.017); ρ(Ā) = 0.90; d = 16-20, K = 24-48; 8.4 × 10^4 problems.

Ideas for us: Run multiple nano_mythos seeds with fixed loop counts (2-16) and classify each by tracking the leading Jacobian multiplier along the fixed-point branch as input evidence varies, to see if fold vs Neimark-Sacker phases emerge in our architecture. Measure the empirical tail P(τ>N) across a difficulty-sorted dataset for fold-phase vs non-fold-phase checkpoints to validate the N^-2 workload scaling and its impact on ACT halting thresholds.

### LoopMTP: A looped transformer guided by latent multi-token prediction
Behzad Shomali, Markus Frey, David Berghaus, Joachim Koehler, Mehdi Ali · 2026-08-04 · http://arxiv.org/abs/2608.03624v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopMTP is a looped transformer architecture that uses latent multi-token prediction to guide intermediate representations across loops. It aligns each loop's hidden state with the embedding of a future token and uses a learned gate to aggregate representations, preventing latent overthinking and undifferentiated computation. The method includes Loop-LNS for stability across variable unrollings and remains stable for up to 15 loops.

Key numbers: 8.1% relative accuracy improvement over non-looped baseline; stable for up to 15 loops; T loops correspond to anticipating T future tokens.

Ideas for us: Implement latent MTP alignment in nano_mythos by aligning loop t hidden states with token t+1 embeddings using cosine similarity; test Loop-LNS (1/T scaling) on fixed loop counts (2-16) to improve gradient propagation in deeper recurrent blocks.

### Looped Transformers with Source-Centered State Evolution
Bum Jun Kim, Kohei Hayashi, Shunsuke Kamiya, Masanori Koyama, Yusuke Iwasawa, Yutaka Matsuo · 2026-07-30 · http://arxiv.org/abs/2607.27656v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Source-Centered State Evolution (SCSE) for looped Transformers, addressing the zero-deviation forcing bias caused by additive source injection at every recurrent step. SCSE uses a learned input-conditioned anchor and evolves deviations from that anchor through a zero-preserving recurrent core, making the anchor a one-step fixed point. It reports improvements across WikiText-2, WikiText-103, web corpora, and LAMBADA, with the clearest gains at deep extra-loop depths, and identifies the learned anchor and anchor-coordinate deviation recurrence as primary contributors.

Key numbers: 139.2M parameters (shared-block comparison max), context lengths 128 to 1024, WikiText-2, WikiText-103, LAMBADA

Ideas for us: Implement SCSE in nano_mythos by computing a learned anchor once per sequence and evolving anchor-relative deviations through the shared recurrent block with a zero-preserving mask, then compare fixed loop counts (2-16) against the current additive-injection baseline. Run an anchor-response diagnostic on trained nano_mythos models to measure the zero-deviation forcing bias and verify whether SCSE eliminates it while retaining off-anchor recurrent motion.

### Agreement rather than normalisation keeps depth reusable in looped language models
Ivan Viakhirev, Kirill Borodin, Maxim Abramov, Grach Mkrtchian · 2026-10-22 · http://arxiv.org/abs/2610.31943v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper investigates whether loop-boundary normalisation or topology agreement between training and inference keeps depth reusable in looped language models. It finds that removing boundary normalisation from released Ouro models causes accuracy to collapse to chance, while models trained from scratch without it perform better and extrapolate more effectively. The authors conclude that agreement between training and inference topologies is the critical factor, not the normalisation itself.

Key numbers: Ouro-1.4B, Ouro-2.6B, Huginn-3.5B; 19M to 76M parameters; 16 seeds per configuration; state norm factor 2.021; readout entropy 0.786 to 12.470 bits; Cliff's delta up to 0.867; r=1 identity check.

Ideas for us: Test nano_mythos with and without LTI/Parcae reinjection at fixed loop counts (2-16) to see if agreement with training depth matters more than state normalisation. Run an early-exit self-distillation experiment comparing fixed loop counts against matched overshoot ratios (K/r0) to isolate the looping advantage from compute differences.

### PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking
Pan Wang, Yihao Hu, Guangyuan Dong, Yitong Lin, Hang Wang, Zimi Liu · 2026-10-20 · http://arxiv.org/abs/2610.27628v1 · relevance 10 · read: abstract+fulltext(14k chars)

PAR introduces Progress-Aligned Recurrence for looped language models, training each iteration k to predict the state after k operations and anchoring recurrent state RMS norms to the first iteration to maintain a stationary step map. It includes a learned halting rule using the existing output head to detect completion and prefix self-distillation to derive intermediate targets from final answers alone. On swap-S5, a Qwen2.5-0.5B model trained on at most 32 moves achieves 95.5% accuracy at 512 moves (16x extrapolation), compared to 55.7% for RLT and 0.8% for explicit CoT. Ablations show norm anchoring is critical, dropping extrapolation from 16x to below 2.2x without it.

Key numbers: 95.5% accuracy at 512 moves (16x training length of 32 moves); 55.7% for RLT; 0.8% for explicit CoT; norm anchoring removal reduces extrapolation to below 2.2x; final-answer supervision reduces it to at most 8x; Poisson(lambda) mean 12, cap 48; position stretch s ~ U[1, 8].

Ideas for us: Implement progress-aligned targets for our fixed loop counts (Dory-style 2-16 loops) by training iteration k to predict the state after k operations, combined with RMS norm anchoring after each recurrent block to stabilize state scale. Test prefix self-distillation with our Poisson loop-depth sampling and ACT halting to generate intermediate supervision from final answers alone, then evaluate length generalization on swap-S5-like state tracking tasks.

### Dual-Slot: From Serial Loops to Parallel Slots in Language Models
Yiqin Hao, Zitong Wang, Boyi Zeng, He Li, Shixiang Song, Feichen Song · 2026-10-18 · http://arxiv.org/abs/2610.25086v2 · relevance 10 · read: abstract+fulltext(14k chars)

Dual-Slot reorganizes serial looped computation into parallel latent and prediction slots to reduce autoregressive decoding latency. It shifts the latent dependency across adjacent tokens, allowing two sequential module invocations to be replaced by a single paired invocation. Token-level parallelism during training is preserved using a small number of Jacobi iterations to refine latent states across the sequence.

Key numbers: Pile validation perplexity: 7.25 to 7.00 (1B dense); average five-shot accuracy improvement: 2.26 percentage points; GSM8K improvement: 5.83 (1B dense) and 3.49 (A0.6B MoE) percentage points; decoding throughput retention: 93.1-96.5% (dense) and 98.1-99.6% (MoE).

Ideas for us: Implement Dual-Slot's latent/prediction slot pairing in nano_mythos to test if parallel slot invocation reduces wall-clock time for fixed loop counts (2-16). Compare the convergence and perplexity of Jacobi-iteration training against the current ACT halting and Poisson loop-depth sampling approach on the 10M parameter scale.

### Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design
Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing · 2026-10-16 · http://arxiv.org/abs/2610.23036v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper isolates three architectural axes for looped language models: iteration envelope, input injection, and latent-state organization. It compares Ouro-style (tied full stack) and Huginn-style (prelude-recurrent-coda sandwich) architectures at matched parameter scale and logical depth. Results show that the Huginn-style design consistently outperforms Ouro across ten benchmarks, with input injection aiding knowledge retrieval and context extraction, while random initialization and separate H/L states provide mixed or negative returns.

Key numbers: 730M stored parameters, 2.9B unrolled parameter applications, 28 Transformer blocks, 112 logical block executions, 8-layer prelude, 12-layer recurrent core, 8-layer coda, 8 iterations, 336B tokens, 8BA0.8B scale, 500B tokens, 75% fewer resident parameters than 112-layer feedforward MoE baseline.

Ideas for us: 1. Compare fixed loop counts (2-16 loops) against ACT halting and Poisson loop-depth sampling on the nano_mythos architecture to determine if a fixed Huginn-style sandwich (prelude-loop-coda) outperforms dynamic halting at 10M parameters. 2. Test LTI input reinjection (Parcae) versus the paper's fixed prelude representation injection to see if continuous reinjection improves context utilization and knowledge retrieval in small looped transformers.

### Monitoring latent reasoning in looped language models
Pawel Olszowiec, Paulina Tomaszewska, Maciej Żelaszczyk, Michal Byra, Neo Christopher Chung · 2026-10-13 · http://arxiv.org/abs/2610.18127v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper presents a mechanistic interpretability protocol with five lenses to analyze eight looped language models from four families. It distinguishes between carry loops, where the prompt enters once and is transmitted via the looped state, and re-read loops, where the prompt is re-supplied at every iteration from a random starting state. Key findings include that answers are decided early, more iterations do not always improve accuracy, attention heads maintain their roles, and carry loops are significantly more vulnerable to noise injection than re-read loops.

Key numbers: 8 models, 4 families, 5 lenses; Ouro trained depth 4 (fixed), LoopUS up to 20 (halting), Parcae mean 8, Huginn mean 32, Rec-OLMo mean 16; noise collapse at alpha=4 for carry loops, re-read models unaffected up to alpha=50; 76-89% of tokens settle on final top-1 in iteration 1-2.

Ideas for us: 1. Test if our Parcae-style LTI input reinjection provides noise robustness similar to re-read loops by injecting isotropic noise into the latent state at various loop depths and measuring output degradation. 2. Compare fixed loop counts (2-16) against ACT halting to determine if the 'answer decided early' phenomenon holds for nano_mythos and if deeper loops degrade prediction quality as seen in Ouro-Thinking models.

### One Model to Loop Them All: Elastic Latent Recurrence for Language Models
Faisal Ladhak, Shantanu Acharya, Santiago Akle Serano, Simeng Sun, Boris Ginsburg, Krishna C Puvvada · 2026-10-08 · http://arxiv.org/abs/2610.12582v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Elastic Latent Recurrence (ELR), a training recipe for looped language models that eliminates the one-loop tax. It uses a gated state update with a learned step size, mixes loop counts within each optimizer update, and conditions each pass on its loop index and budget. The method allows a single checkpoint to maintain baseline accuracy at one loop while improving with additional loops, validated on 4B and 30B-A3B models.

Key numbers: 4B dense model, 30B-A3B MoE model, 1T tokens, 300B tokens, 0.52 points below baseline (4B), 1.04 points above baseline (30B-A3B), 7.25 points gain (4B), 4.75 points gain (30B-A3B), 16% more training FLOPs (late mixed depth), 62% more training FLOPs (full mixed depth), 11.7% mixed-depth update time reduction (MoE route replay), Koi-4B scores 18 (1 loop) and 29 (4 loops) on Artificial Analysis Intelligence Index.

Ideas for us: Test the adaptive-delta state update with a learned step size on nano_mythos to see if it preserves one-loop accuracy during fixed loop count training. Implement cross-loop route replay for MoD token routing to measure if it reduces training time without affecting accuracy.

### Safe at One Loop, Risky at Another: Aligning Safety Across Recurrent Depths in Looped Language Models
Yi Wang, Xiuyuan Qi, Dongqi Han, Dongsheng Li, Wenjie Wang · 2026-10-07 · http://arxiv.org/abs/2610.10625v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper evaluates safety across recurrent depths in LoopLMs, finding that attack success can increase at deeper depths and that jailbreaks transfer between depths. It introduces SafeBridge, which uses depth-specific FiLM-style modulation, a selective gated state bridge from an earlier safer depth, and joint safety supervision across depths. Experiments on the Ouro family show SafeBridge reduces attack success while improving utility and maintaining over-refusal behavior.

Key numbers: L=4, D={2,3,4}, scale 1+tanh(a) in (0,2), a=b=0 initialization

Ideas for us: Test whether safety ASR varies across fixed loop counts (2-16) in nano_mythos under jailbreak attacks; add a lightweight depth-specific FiLM modulation and a gated bridge from an early loop to later loops during alignment fine-tuning.

### Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models
Yi Wang, Rui Qian, Yu Li, Haoyang Yao, Wenjie Wang · 2026-10-07 · http://arxiv.org/abs/2610.10623v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces LoopOPD and Dynamic LoopOPD (D-LoopOPD), frameworks for post-training Looped Language Models (LoopLMs). LoopOPD uses a frozen terminal loop policy as a teacher to provide dense on-policy distillation to an intermediate loop student, while D-LoopOPD refreshes this teacher as shared parameters are updated to enable recurrent self-improvement. Experiments on Ouro-Thinking models show improvements in mathematical reasoning, general reasoning, and code generation.

Key numbers: Ouro-2.6B-Thinking D-LoopOPD raises terminal loop AIME 2024 Avg@16 from 62.08 to 71.67

Ideas for us: Implement LoopOPD on nano_mythos by using the output of the final loop (e.g., loop 8) as a frozen teacher to distill dense token-level supervision into intermediate loops (e.g., loops 1-4) during SFT. Test D-LoopOPD by periodically unfreezing and refreshing the terminal loop teacher as the shared recurrent block weights are updated to observe if recurrent self-improvement stabilizes fixed loop counts.

### SanSi: A Looped Typed Decision Model for System 1.5 Thinking
Shuyu Gan, Young-Jun Lee, Dongyeop Kang · 2026-10-06 · http://arxiv.org/abs/2610.07730v2 · relevance 10 · read: abstract+fulltext(14k chars)

SanSi converts a pre-trained looped language model into a typed decision model by applying the same transformer stack recursively and reading option probabilities after every loop. Each loop is trained with a proper scoring rule (cross-entropy and Brier score), allowing a single model to serve any budget from one to eight loops. The backbone remains frozen while LoRA adapters and per-loop readouts are trained, demonstrating that looping improves accuracy on multi-step reasoning and generalizes to depths beyond training without adding parameters.

Key numbers: 72.0% accuracy on 10,027 test decisions; 13.5 points above non-looped same-shape model; 5.3 points above newer non-looped model of its size; 1.8 points below 3x parameter model; 61M total trainable parameters (60.6M LoRA + 0.27M readouts); T=8 loops trained (twice Ouro's 4 pre-training loops); 7.7x computation of a single pass; 58.4% accuracy after one loop when all loops trained vs 35.6% when only last loop trained; second loop brings 62% of gain from loop 1 to 8, third brings to 88%; 4.5 average loops with Ouro's exit gate keeping loop 8 accuracy; RL judge raises generator correct answers from 37.0% to 52.1%.

Ideas for us: Train per-loop readouts with cross-entropy + Brier loss on our nano_mythos model so early exits are accurate, not just the final loop. Test fixed loop counts (2-16) with a frozen Ouro-style backbone and shared LoRA adapters to measure accuracy vs. compute trade-offs and calibration drift after answers settle.

### Towards Looped Models Done Right, Part II: Rethinking at Fixed Points
Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing · 2026-10-05 · http://arxiv.org/abs/2610.06833v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper argues that looped language models approach fixed points as recurrence depth increases, allowing the endpoint to stand in for the full trajectory. It introduces a learned depth prior with an entropy term and orthogonal input injection (OrthoInj) to improve fixed-point shaping. These methods enable truncated backpropagation, terminal KV sharing for decoding, distilled prefill, and RL updates using saved rollout states, reducing costs across training and inference from 100M to 1.6B parameters.

Key numbers: 100M to 1.6B parameters; 1.79x faster prefill; 2x faster RL updates; 3x smaller KV cache; 0.4 points downstream average difference; 1.0-4.5 points trailing teacher downstream average; 0.5-0.9 points above teacher at matched latency; 1.6 points GSM8K pass@1 difference; 0.3-1.4% validation perplexity reduction.

Ideas for us: Implement orthogonal input injection (OrthoInj) in nano_mythos to prevent the state's component along the input from amplifying or canceling the LTI reinjection. Test a learned depth prior with an entropy term for Poisson loop-depth sampling to optimize supervision at target depths while maintaining KV sharing compatibility.

### Loopy: Low-Bit Quantization Framework for Looped Language Models
Zeyu Li, Yipu Zhang, Jintao Cheng, Xin Li, Wei Zhnag · 2026-10-04 · http://arxiv.org/abs/2610.05265v2 · relevance 10 · read: abstract+fulltext(14k chars)

Loopy is a post-training quantization framework for looped language models that selects shared low-bit representations based on final prediction loss at a target recurrent depth. It uses channel scaling and orthogonal rotations to parameterize candidate configurations and employs progressive full-depth evaluation to efficiently allocate calibration windows to promising candidates. The framework was evaluated on Ouro, Huginn, and Recurrent-Llama under W4A8 and W4A4 quantization, showing a 36.5% relative reduction in LAMBADA perplexity on Ouro-1.4B compared to SpinQuant.

Key numbers: 36.5% relative LAMBADA perplexity reduction on Ouro-1.4B under W4A4 vs SpinQuant; recurrent depths T=1 to T=4 tested; W4A8 and W4A4 quantization settings; 20 rotation-scaling configurations evaluated.

Ideas for us: Test if quantization configuration rankings for nano_mythos shift across fixed loop counts (2-16) using Loopy's target-depth objective. Run a small-scale progressive full-depth selection on nano_mythos to find optimal channel scaling and rotation seeds for deployment at 8 loops.

### Closing the Loop: Practical Training Recipes for Looped Language Models
Andrei Marchenko, Viacheslav Bezrukov, Oleg Kashurin, Inessa Fedorova, Dmitry Bocharov, Yuliana Shakhvalieva · 2026-09-30 · http://arxiv.org/abs/2610.00673v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper presents practical training recipes for looped language models, reducing the training budget from 7.7T to 310B tokens. It demonstrates that a 1.4B LoopLM outperforms parameter-matched dense models on 12 benchmarks and introduces a minimal conversion method for pretrained dense models using a single learned input-mixing scalar and a smoothed exit loss. The authors identify gate collapse as a key failure mode and solve it using learning-rate warmup and a higher KL coefficient (β=0.15).

Key numbers: 7.7T tokens (Ouro budget), 310B tokens (this work), 1.4B LoopLM, 3.9B dense model, 36% parameters, +14 GSM8K, +10 MATH, +22 DROP, β=0.15, T=4 fixed depth, Qwen3-1.7B-Base conversion.

Ideas for us: 1. Test the paper's gate stability findings by comparing fixed loop counts (2-16) with and without learning-rate warmup and varying KL coefficients (β=0.05 vs 0.15) on nano_mythos to prevent early exit-gate collapse. 2. Implement the minimal dense-to-loop conversion recipe (single input-mixing scalar + smoothed exit loss) on a small pretrained checkpoint to see if it improves over a continued dense baseline without step-specific adapters.

### What Makes Recurrence Effective in Looped Language Models?
Xinlin Zhuang, Siyuan Wang, Imran Razzak, Weiyang Liu · 2026-09-29 · http://arxiv.org/abs/2609.36636v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper examines when recurrence helps in looped language models, where to place recurrent and non-recurrent layers, and how to condition recurrent computation. It finds that recurrence improves reasoning beyond the training horizon but degrades knowledge performance, and that non-recurrent output layers improve robustness to under-unrolling. It proposes history-state injection combined with timestep conditioning as a more effective alternative to initial-state injection for variable inference budgets.

Ideas for us: Train fixed-loop variants at depths 2, 4, 8, and 16 to compare knowledge versus reasoning performance across under-unrolling, training horizon, and extrapolation. Replace conventional initial-state injection with channel-wise history-state injection plus timestep conditioning in the shared recurrent block and measure robustness across loop counts.

### Loop Dropout: Regularizing Shared Updates in Looped Language Models
Zirui Zhu, Hailun Xu, Xuanlei Zhao, Yong Liu, Yingxuan Ren, Kanchan Sarkar · 2026-09-28 · http://arxiv.org/abs/2609.34218v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper identifies a late-loop bias in standard LoRA fine-tuning for looped language models, where shared updates are less effective at early loop positions. It introduces Loop Dropout, which applies stochastic masking to adapter applications during training with inverse-survival rescaling to preserve expected update strength. This method allows the shared update to adapt effectively across all loops without adding trainable parameters or increasing inference compute, and it enables zero-shot generalization to deeper recurrence depths.

Key numbers: GSM8K loss reduction: Loop 1 (11.9% LoRA vs 30.8% Loop Dropout), Loop 4 (32.4% LoRA vs 35.7% Loop Dropout); Zero-shot generalization to 8 loops outperforms LoRA by 4.75 percentage points on GSM8K; Survival probability p = 1/2 example provided.

Ideas for us: Implement Loop Dropout on the shared recurrent block of nano_mythos with a survival probability q=0.5 to test if it reduces late-loop bias in fixed-loop counts (2-16). Compare fixed-depth training with Loop Dropout against standard LoRA using ELT-style early-exit self-distillation to measure early-loop adaptation improvements.

### LoopCD: Loop-wise Contrastive Decoding for Improving Reasoning in Looped Language Models
Byeongho Yu, Junhyuk So, Eunhyeok Park · 2026-09-21 · http://arxiv.org/abs/2609.24196v1 · relevance 10 · read: abstract+fulltext(14k chars)

LoopCD is a training-free inference-time decoding method for Looped Language Models that contrasts logits from an earlier loop iteration against the final refined iteration to suppress unstable, early-overconfident predictions on hard tokens. It requires no architectural changes and adds negligible overhead by reusing existing hidden states. Experiments show it improves reasoning performance across GSM8K, MATH-500, HumanEval, MBPP, and StrategyQA for models like Huginn-0125.

Key numbers: Hard tokens account for 10%–15% of the reasoning trace; replacing hard tokens causes 52.9% of correct GSM8K trajectories to become incorrect (vs 31.3% for easy tokens) and 71.9% for StrategyQA (vs 25.9%); Huginn-0125 architecture is (2, 4⊗32, 2).

Ideas for us: Apply LoopCD at inference to our fixed-loop nano_mythos checkpoints (2–16 loops) by contrasting logits from an early loop (e.g., loop 1 or 2) against the final loop to improve reasoning on GSM8K-style tasks without retraining. Compare LoopCD against greedy decoding and ACT halting on the same fixed-loop models to measure whether contrastive decoding stabilizes hard-token refinement across different loop depths.

### WaveFront Decoding: Parallelized Self-Speculative Decoding for Looped Language Models
Hyeongju Ha, Jae-Joon Kim · 2026-09-19 · http://arxiv.org/abs/2609.23033v2 · relevance 10 · read: abstract+fulltext(14k chars)

WaveFront Decoding (WFD) is a training-free self-speculative decoding framework for looped language models that batches token states at different positions and recurrence depths into a diagonal wavefront. It concurrently performs drafting at shallow depth and verification at full depth within the same recurrent-block calls, correcting rejected drafts using full-depth predictions. WFD achieves 2.42x speedup on Ouro-2.6B and 3.54x on Huginn-3.5B over autoregressive decoding, with cross-recurrence KV sharing increasing Huginn speedup to 4.81x.

Key numbers: 2.42x speedup (Ouro-2.6B), 3.54x speedup (Huginn-3.5B), 1.27x higher throughput than DtV, 4.81x speedup with KV sharing (Huginn-3.5B)

Ideas for us: Implement WFD for nano_mythos to batch mixed-depth states during fixed-loop inference (2-16 loops) and measure speedup against standard autoregressive decoding. Test cross-recurrence KV sharing with Parcae LTI input reinjection to reduce KV traffic during deeper recurrent block loops.

### Allocating Recurrent Compute in Looped Language Models
Ruhai Lin, Yiyang Guo, Rui-Jie Zhu, Hao Ye, Jason K. Eshraghian · 2026-08-18 · http://arxiv.org/abs/2608.18230v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces MixerLoop, which repeats only the Gated DeltaNet token mixer while applying the dense FFN once per physical layer. It compares MixerLoop against NoLoop and FullLoop at 15M and 110M parameters using the same data and optimization settings. MixerLoop surpasses FullLoop on aggregate CORE at 15M and retains 41.5% of its CORE improvement at 110M while reducing recurrent-backbone projection FLOPs by 45.9%.

Key numbers: 15M and 110M parameters; T=4 loops; 45.9% reduction in recurrent-backbone projection FLOPs; 41.5% of CORE improvement retained at 110M; 1.52x FullLoop prefill throughput; FFNs account for 61.2-61.3% of per-layer projection FLOPs.

Ideas for us: Test MixerLoop on nano_mythos by looping only the Gated DeltaNet mixer while applying the FFN once to compare against full-block recurrence. Run an ablation on fixed loop counts (2-16) to measure the marginal ITR and determine the optimal loop depth for the 1B-parameter run.

### Looped Language Models Improve Compositional Tool Calling
Andrei Cristian Popescu, Haitz Sáez de Ocáriz Borde, Pietro Liò · 2026-08-17 · http://arxiv.org/abs/2608.18171v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper evaluates looped language models on compositional tool-calling benchmarks (API-Bank, BFCL, NESTful), comparing native and retrofitted models under matched SFT recipes. It finds that recurrent computation improves multi-step and dependency-aware tool use, with accuracy generally increasing as recurrent depth increases. Adaptive inference is shown to provide a better compute-performance trade-off by allocating iterations only when needed.

Key numbers: Ouro-1.4B, Ouro-2.6B, OLMo-2-1B, Llama-3.2-1B, 4,096 tokens, 2 epochs, 2e-5 LR, LoRA rank 32, bf16, beta=0.1, NVIDIA A100 80GB

Ideas for us: Test fixed loop counts (2-16 loops) on a synthetic compositional tool-calling dataset to measure if deeper recurrent blocks improve dependency preservation. Compare ACT halting against fixed-depth inference on small-scale tool-use tasks to validate the compute-performance trade-off.

### Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching
Kristian Schwethelm, Daniel Rueckert, Georgios Kaissis · 2026-08-10 · http://arxiv.org/abs/2608.09444v2 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Continuous Depth Batching (CDB), an efficient inference method for looped language models that allows tokens to exit a shared recurrent core at different depths. It addresses scheduling challenges by forming new batches between loop steps, managing looped KV-caching, and using a lookahead gate to overlap batch preparation. Experiments on Ouro 1.4B and Huginn 3.5B show that fully looped architectures are best suited for depth-adaptive inference, with CDB realizing up to 99% of the estimated maximum speedup on Ouro.

Key numbers: Ouro 1.4B, Huginn 3.5B, up to 99% of estimated maximum speedup, 83 to 96% speedup on Huginn, 2-4-2 model example, r_t <= r_max

Ideas for us: Implement a simple CDB-style scheduler for nano_mythos to test if refilling batch slots between loop steps improves throughput on small fixed loop counts (2-16). Compare the inference latency of a fully looped (0-core-0) nano_mythos variant against one with larger prelude/coda boundary stages to measure the impact of boundary-stage cost on depth-adaptive speedup.

### Running Fewer Loops Removes Refusal Behaviour in Recurrent-Depth Language Models
Seyedarmin Azizi, Erfan Baghaei Potraghloo, Arya Fayyazi, Saeid Shokoufa, Massoud Pedram · 2026-10-10 · http://arxiv.org/abs/2610.15218v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper demonstrates that reducing inference loops in recurrent-depth language models causes a disproportionate loss of safety compared to task ability. Models retain the ability to detect harmful requests at shallow depths but lose the refusal mechanism, leading to increased harmful compliance. Training at randomly chosen loop depths is shown to preserve safety separation without sacrificing full-depth accuracy.

Key numbers: Ouro-1.4B-Thinking GSM8K: 0.692 at half depth vs 0.775 full (89% retention); Harmful help rate: 0.038 to 0.205 (5.3x increase); Harmful help rate Ouro-2.6B at 1 loop: 0.035 to 0.344 (9.9x); Linear classifier ROC AUC: 0.967 after one loop vs 0.986 full depth; Refusal rate drop via direction removal: 0.950 to 0.130; Random depth training separation: +0.399 vs +0.118 standard; Huginn-0125 harmful help: <=0.040 at depths 1-32 vs 0.194 standard.

Ideas for us: Test if fixed loop counts (2-16) in nano_mythos create a safety gap where harmful compliance rises faster than task performance drops. Compare standard safety fine-tuning against training with Poisson/randomly sampled loop depths to see if random depth training preserves the refusal gap at low loop counts without harming full-depth accuracy.

### Scaling RNA Foundation Models through Iterative Refinement
Pengan Chen, Liang Hong, Xinyuan Liu, Han Huang, Ziqian Lin, Jiayang Chen · 2026-10-09 · http://arxiv.org/abs/2610.13106v1 · relevance 10 · read: abstract+fulltext(14k chars)

Eco-RNA is an 80M-parameter RNA foundation model that uses a shared, input-conditioned Transformer core to enable adjustable inference depth through iterative refinement. It employs a prelude to encode input once, a recurrent core applied K times, and a coda for readout, with persistent input injection and bounded state retention. Training involves masked reconstruction at sampled terminal depths, state regularization, and a one-sided reconstruction anchor to ensure deeper iterations improve upon shallower references.

Key numbers: 80M parameters; 70-assay RNAGym benchmark; aggregate absolute Spearman correlation 0.242 (1 iteration) to 0.286 (4 iterations); 18% relative improvement at approximately twice forward FLOPs; surpasses Evo 2 40B at 4 iterations with ~500x fewer parameters; iterations K=1 to 8; D(K) = Lp + K*Lr + Lc; 0 < alpha_min < alpha_max < 1.

Ideas for us: 1. Implement the one-sided reconstruction anchor loss to test if anchoring deeper recurrent states to a shallow reference stabilizes training for our fixed loop counts (2-16). 2. Experiment with the channelwise retention coefficient (alpha) and input scale (delta) parameterization to regulate state dynamics in our 1-layer shared recurrent block.

### RAMP: Recurrent Adaptive Mixed Precision for Efficient Prefill in Recurrent-Depth Language Models
Ke Wan, Chen Chen · 2026-10-09 · http://arxiv.org/abs/2610.13041v1 · relevance 10 · read: abstract+fulltext(14k chars)

RAMP is a training-free mixed-precision inference method for recurrent-depth language models. It analyzes quantization error propagation and finds that errors introduced earlier in recurrent depth are attenuated by subsequent steps, while errors near the end have a larger impact. Consequently, RAMP uses low-precision computation for early recurrent refinement and high precision for terminal refinement to minimize final error under a fixed precision budget.

Key numbers: 1.16x end-to-end first-prefill speedup, 13.9% latency reduction, context lengths up to 4K, NVIDIA H100, 24 FP8 and 8 BF16 recurrent steps budget

Ideas for us: Test RAMP on nano_mythos by quantizing the first K loops of the shared recurrent block in FP8/INT8 and keeping the final loops in BF16 during prefill. Measure prefill latency and final hidden-state deviation across fixed loop counts (2-16) to validate the early-quantization placement hypothesis.

### Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth
Ivan Viakhirev, Kirill Borodin, Amirah Almutairi, Serguei Barannikov, Maxim Abramov, Grach Mkrtchian · 2026-08-18 · http://arxiv.org/abs/2608.18222v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper studies recurrent-depth reasoners, showing that a trained operator's finite-time dynamical regime (settling, marginal, or drifting) predicts whether additional test-time iterations improve, preserve, or degrade accuracy. It introduces a sufficient condition for depth-safety based on per-step displacement relative to decoder margin and demonstrates that a terminal fixed-point objective can shift an operator into a settling regime, enabling depth-safe extrapolation and conversion to harder instances. The authors apply these measurements to Huginn-3.5B, classifying it as non-settling, and provide a label-free objective to repair its depth-safety.

Key numbers: 800 unaugmented examples per difficulty tier; Sudoku accuracy 0.19 to 0.34 past training horizon; reachability plateau 0.92; Huginn-3.5B digit copying 0.00 to 1.00; test-time depths {3, 16, 32, 64, 128}; trained at 16 recurrent steps; d=256; batch 64; gradient clipping 0.5.

Ideas for us: 1. Implement a terminal fixed-point objective on the nano_mythos recurrent block and measure if it shifts the dynamical regime from drifting to settling, then test if fixed loop counts (2-16) become depth-safe. 2. Log the per-step displacement and decoder margin during inference on fixed loop counts to validate the sufficient condition for depth-safety and identify the optimal halting point before degradation occurs.

### Behaviour Is an Incomplete Measure of Reasoning Development: Cross-surface pre-arrival accessibility and the limits of developmental inference in a recurrent-depth reasoner
Simon Lam-Muir · 2026-08-17 · http://arxiv.org/abs/2608.16085v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper studies a 30M-parameter recurrent-depth transformer with a 4-layer block applied repeatedly within a single forward pass, decoupling computational depth from parameter count. It compares behavioral thresholds, internal probe accessibility, and final checkpoints across symbolic and verbal training surfaces, finding that these measurements do not identify the same event. The authors conclude that behavior and decoder accessibility are insufficient to identify acquired computation and that causal intervention is necessary.

Key numbers: 30M parameters; 768-dimensional state; 4-layer recurrent block; 40 answer classes; 3-hop cost 70 logical epochs (symbolic) vs 13,055 logical epochs (verbal), 186.5x contrast; verbal 4-hop cleared in 8 logical epochs; 4-hop held-out behavior peaked at 3/40 and ended 0/40 during the grind; probe accuracy 0.056159 vs uniform chance 0.025, untrained control 0.024758, population baseline 0.048309 (p=0.012987, 16/40 classes); symbolic surface probe 0.1020 vs zero-step control 0.0460 (p=0.000999) and 0.0618 at readout comparator (p=0.004, 21/40 classes); k=5 stage accumulated at least 6,400 logical epochs; 1,643 frames analyzed; B=1000 permutation replicates.

Ideas for us: Run fixed loop counts (2-16) on nano_mythos and use preregistered linear probes on hidden states at intermediate loop depths to detect pre-arrival answer accessibility before behavioral competence emerges. Compare probe accessibility and behavioral thresholds across symbolic vs verbal task surfaces to see if the 186.5x acquisition-cost contrast replicates in our architecture with ACT halting and MoD routing.

### Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation
Amr Hegazy, Amr Alanwar, Mostafa Elhoushi · 2026-08-15 · http://arxiv.org/abs/2608.15062v4 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Gated Recurrent Transformer (GRT), a recurrent depth architecture with fixed prelude and coda blocks bracketing a single shared core iterated R times. It uses a lightweight projection and elementwise update gate conditioned on the hidden state, prelude output, and resampled noise to prevent representational collapse. Results show it matches a 12-layer GPT-2 Small baseline under isoFLOPS with 64% fewer parameters, and achieves 2.76 validation loss versus 2.84 for a non-recurrent counterpart under isoPARAMS.

Key numbers: 3-layer GRT matches 12-layer GPT-2 Small; 64% fewer parameters at small scale; 63% fewer parameters and 59% less peak decoding memory at large scale with 10% increased compiled generation latency; 2.76 vs 2.84 validation loss under isoPARAMS; ~92% accuracy retained when running half recurrence steps; nine scale-by-budget cells evaluated.

Ideas for us: Implement the GRT gating mechanism in nano_mythos's shared recurrent block, conditioning the update gate on hidden state, prelude output, and per-step noise, then compare fixed loop counts (2-16) against ACT/Poisson depth sampling. Run an isoPARAMS ablation on the 10M parameter model varying prelude/coda ratios and recurrence depth to validate whether deeper recurrence improves validation loss at matched parameter and data budget.

### The Ignition Is Real, and It Lives at the Readout: Latent composition, difficulty-clocked ignition, and the interface-constituted commit in a recurrent-depth reasoner
Simon Lam-Muir · 2026-08-04 · http://arxiv.org/abs/2608.03263v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper replicates a 30M-parameter recurrent-depth transformer and validates a pre-registered "compositional ignition" signature across the vocabulary readout and hidden state. It finds that arrival time rises lawfully with problem depth, the decision margin jumps 5.8–8.0 logits in one iteration at commitment, and the hidden-state direction snaps and freezes while subsequent displacement is predominantly radial and readout-null. The authors conclude the ignition is real and located at the readout interface, not an artifact or inherited from verbal training data.

Key numbers: 30M parameters; 768-d state; 4-layer recurrent block; 2,000 synthetic facts; 200 entities × 10 relations; curriculum to stage 21; first-hit medians 1 → 4 across k=1–10; sharpness 0.805 ∈ [0.714, 0.867]; margin jump 5.8–8.0 logits; 96% of cases exceed null p90; 274/274 event-eligible items sign change; radial share 0.961 of squared-norm; radial logit effect ≤ 5.7e-6; LN angular steps 52.9° to 1.2° over eight iterations; relay = 0.00.

Ideas for us: Run fixed-loop-count sweeps (2–16 loops) on nano_mythos and measure first-hit arrival time and margin jump at commitment to see if the ignition signature scales with loop depth and ACT halting. Instrument the tied readout and hidden state to test whether intermediate compositions are recoverable (relay metric) and whether post-commit displacement is readout-null radial motion, informing MoD routing and early-exit distillation thresholds.

### Retrofitting Recurrent Depth into a Pretrained Language Model: Installation, Extrapolation, Transfer, and Retention at Two Parameter Budgets
Mark Shapiro · 2026-07-31 · http://arxiv.org/abs/2608.11233v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper describes retrofitting a pretrained Qwen2.5-0.5B model into a recurrent-depth system by splitting it into Prelude, Recurrent Block, and Coda. It compares two training budgets: a 6.01M parameter adapter over frozen base weights and a 180.6M parameter full-block adaptation. Key findings include the ability to extrapolate to roughly 1.5x supervised depth, superior performance over scratchpad-trained models beyond depth 10, and a catastrophic-interference boundary when attempting to learn inverse tasks.

Key numbers: 6.01M adapter parameters; 180.6M full-block parameters; 1.5x supervised depth extrapolation; 70% accuracy through depth 18; 84% vs 72% overall win rate; 53% vs 2.5% retention beyond depth 10; 7.6x faster inference.

Ideas for us: 1. Test if a 6M parameter adapter over a frozen base can install a stable iterative latent transition in nano_mythos using intermediate-step supervision. 2. Evaluate forced-depth extrapolation in nano_mythos to determine if accuracy holds at 1.5x the supervised loop depth.

### Enabling Dynamic Computation in Looped LMs
Aayush Mishra, Arnau Padrés Masdemont, Victor Conchello Vendrell, Jordi Ros Giralt, Arash Behboodi, Fabio Valerio Massoli · 2026-10-06 · http://arxiv.org/abs/2610.09013v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper addresses the failure of looped LMs to realize dynamic computation in practice due to rigid KV caches and uniform early-exit priors. It proposes Ragged KV, a caching strategy that aliases deeper blocks to the best-available cache, enabling true early-exit without additional training. It also introduces global prior enforcement to allow tokens to exit at heterogeneous depths based on input difficulty. These methods were validated on Ouro models and smaller looped LMs trained from scratch.

Key numbers: 30% reduction in FLOPs and KV memory; Ouro models U=4; Nanbeige U=2; Ouro-1.4B, 2.6B, 1.4B-thinking, 2.6B-thinking; arXiv:2610.09013v1

Ideas for us: Implement Ragged KV in nano_mythos to allow tokens to exit at different depths while maintaining a shared physical KV pool via page tables. Replace the uniform entropy regularizer with a global prior enforcement objective to encourage heterogeneous exit depths based on token difficulty during fixed-loop testing.

### Looped Uniform Diffusion Language Models
Saiteja Utpala, Chieh-Hsin Lai, Yuki Mitsufuji, Praneeth Vepakomma · 2026-10-21 · http://arxiv.org/abs/2610.30066v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper investigates selective looping in Uniform Diffusion Language Models (UDLMs), where a small block of layers is recursively applied within each denoising step. It finds that looping significantly improves training efficiency, reducing required FLOPs by up to 2.8x on OpenWebText and 3.3x on FineWeb-Edu to reach the same validation NLL. It also shows that sampling smaller loop counts more frequently during training improves inference efficiency by using 12% fewer loops without degrading accuracy.

Key numbers: 2.8x fewer training FLOPs (OpenWebText), 3.3x fewer training FLOPs (FineWeb-Edu), 9.8% lower generation perplexity, 11.5% lower zero-shot perplexity on average, 12% fewer loops with FLOP-proportional sampling, 26.0% fewer loops with adaptive looping (uniform sampling), 34.8% fewer loops with adaptive looping (FLOP-proportional sampling).

Ideas for us: 1. Test FLOP-proportional loop-count sampling on nano_mythos to see if smaller loop counts reduce inference cost while maintaining accuracy. 2. Implement adaptive looping (halting based on relative hidden state change) for the recurrent block to compare against fixed loop counts (2-16) in terms of compute efficiency and NLL.

### Looped GPT-BERT: Trading Parameters for Computation in Small Language Modeling
Tingshuo Fan, Hongtao Mu, Tianyu Zhou, Hansen Liu, Tao Ji · 2026-09-09 · http://arxiv.org/abs/2609.09691v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces Looped GPT-BERT, which combines GPT-BERT's hybrid masked next-token and causal language-modeling objectives with depth-wise parameter sharing. It compares a 12-layer non-looped baseline against a model with four physical layers applied 1, 3, 6, or 12 times, finding that additional recurrent computation can improve training and preserve performance on linguistic tasks, though few physical layers may restrict representational space. The final 4x12 model contains 12.18M parameters and achieves a BabyLM 2026 Overall Average of 35.42 and NLP Average of 48.48.

Key numbers: 4x12 model, 12.18M parameters, 7,482,189 words, 8k tokenizer, hidden size 384, 3.15M embedding parameters, 1:3 BERT:GPT ratio, BabyLM Overall Average 35.42, NLP Average 48.48, loop counts 1/3/6/12

Ideas for us: Run a fixed-loop-count ablation (2, 4, 8, 16 loops) on nano_mythos using the same 1:3 masked-to-causal objective ratio to isolate loop-depth effects from ACT halting. Compare a shallow 2-layer prelude with a deeper recurrent block against the current 2-layer prelude and 1-layer block at matched parameter counts to test whether representational space limits looped performance.

### RecurTrace: Adaptive Latent Reasoning with Loop-Time Memory
Yuxiang Wang, Kunyu Feng, Yingda Shen, Haoning Xu, Junyu Wang, Zhizheng Wu · 2026-09-03 · http://arxiv.org/abs/2609.03379v3 · relevance 10 · read: abstract+fulltext(14k chars)

RecurTrace addresses limitations of looped transformers by introducing Loop Memory Attention, which allows each looped layer to attend to its own states from previous iterations along the loop-time axis. It also implements an oracle-distilled halting head that predicts whether additional depth will reduce loss, enabling per-input loop budgeting. The method was tested on Qwen3 models from 0.6B to 8B, showing accuracy gains over fixed-depth and adaptive-compute baselines on MathQA and other tasks.

Key numbers: MathQA accuracy: 56.9% at ~2.0 loops (RecurTrace) vs 54.7% best fixed depth; TaH-Mismatch: 55.7% at 2.1 loops; LoopUS-Conf: 55.3% at 3.2 loops; CALM: 54.1% at 5.6 loops; Added modules: at most 2.2% of parameters; Generation accuracy gains: 0.6 to 3.4 points across 0.6B, 1.7B, 4B, and 8B models; Loop blocks: layers 12-14 for 0.6B/1.7B, layers 15-17 for 4B/8B.

Ideas for us: Implement Loop Memory Attention in the nano_mythos recurrent block to allow the model to attend to previous loop states instead of relying solely on the latest output. Replace Poisson loop-depth sampling with an oracle-distilled halting head to dynamically determine loop counts per input during the 1B-parameter run.

### Beyond Depth Truncation: Controlled Evaluation of Depth Utilization in Recursive Language Models
Ha Van Dau, Thanh Tung Khuat, Nguyen Thanh Dung · 2026-09-17 · http://arxiv.org/abs/2609.19934v1 · relevance 10 · read: abstract+fulltext(14k chars)

The paper introduces the Depth Control Protocol (DCP) to evaluate depth-recurrent language models, arguing that standard depth truncation conflates block application count, distinct computation, and readout miscalibration. It identifies that models trained at a fixed depth develop readout specialization, causing naive truncation to overestimate the contribution of depth. The authors show that sampling recurrence depth during training eliminates this confounder and that distinct depth often contributes approximately zero to performance gains.

Key numbers: 1.4251 nats naive depth effect; 47.9% block applications; -2.8% distinct depth; 25.6% readout miscalibration; 54.8% inter-block composition; 1,000 steps of continued training reduced calibration share from 29.9% to 5.1%; Huginn-0125 temperature flat within 0.045 across 32x depth range vs 0.57 across 8x range in fixed-depth model.

Ideas for us: 1. Implement the DCP linchpin control on nano_mythos by running the full loop budget (e.g., 8 loops) while executing only one distinct recurrent iteration to isolate the effect of block application count from distinct depth. 2. Conduct a controlled training intervention where a small branch of nano_mythos is trained with Poisson loop-depth sampling to see if it eliminates readout miscalibration compared to a fixed-loop-count branch.

### IFM/LoopedLM-P2-huginn-s-parcae
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-s-parcae · relevance 10 · read: abstract

The item is a model repository for a looped language model named LoopedLM-P2-huginn-s-parcae. It is tagged with looped-language-model and paper-part2, and references arxiv:2610.06833. The model uses safetensors and is licensed under Apache-2.0.

Ideas for us: Compare fixed loop counts (2-16) against ACT halting on the nano_mythos architecture. Test ELT-style early-exit self-distillation using the Huginn-S-Parcae model as a teacher.

### IFM/LoopedLM-P2-huginn-s-orthoinj
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-s-orthoinj · relevance 10 · read: abstract

The repository is a paper-part-2 model for a looped language model (LoopedLM) named Huginn-S. It is tagged as a looped-language-model and uses safetensors for text generation under an Apache-2.0 license.

Ideas for us: Compare fixed loop counts (2-16) against Poisson loop-depth sampling to measure stability and perplexity. Test ELT-style early-exit self-distillation using the looped block's intermediate states as supervision targets.

### IFM/LoopedLM-P2-huginn-s-learned-entropy0
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-s-learned-entropy0 · relevance 10 · read: abstract

The item is a Hugging Face model repository for a looped language model named LoopedLM-P2-huginn-s-learned-entropy0. It is tagged as a looped-language-model and is associated with paper-part2 and arxiv:2610.06833. The model is licensed under Apache-2.0 and is based in the US region.

Key numbers: 2026-09-28

Ideas for us: Implement learned entropy-based halting criteria for the ACT mechanism to dynamically determine loop depth. Compare the performance of fixed loop counts (2-16) against the learned entropy halting approach on the nano_mythos architecture.

### IFM/LoopedLM-P2-huginn-s-huginn-linear
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-s-huginn-linear · relevance 10 · read: abstract

The item is a Hugging Face model repository for a looped language model named LoopedLM-P2-huginn-s-huginn-linear, released by IFM on 2026-09-28. It is tagged as a looped-language-model and paper-part2, associated with arxiv:2610.06833, and uses Apache-2.0 licensing.

Ideas for us: Benchmark the fixed loop counts (2-16) against the Huginn-s linear variant to compare depth scaling. Test ELT-style early-exit self-distillation using the Huginn architecture as a teacher for the nano_mythos recurrent block.

### IFM/LoopedLM-P2-huginn-s-fixed-r5
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-s-fixed-r5 · relevance 10 · read: abstract

The item describes a looped language model (LoopedLM-P2-huginn-s-fixed-r5) that uses fixed loop counts ranging from 2 to 16. It is a Hugging Face model repository associated with a paper on looped language models and is tagged for text generation.

Key numbers: 2-16 loops

Ideas for us: Test fixed loop counts of 2, 8, and 16 on the nano_mythos architecture to compare convergence and perplexity against the current ACT halting and Poisson sampling approach. Implement ELT-style early-exit self-distillation using the fixed-loop model as a teacher for the recurrent block.

### IFM/LoopedLM-P2-huginn-m-parcae
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-m-parcae · relevance 10 · read: abstract

The repository is a model checkpoint for a looped language model, specifically tagged as 'LoopedLM-P2-huginn-m-parcae'. It is associated with 'paper-part2' and includes tags for xllm, safetensors, and text-generation. The model is licensed under Apache-2.0 and linked to arxiv:2610.06833.

Key numbers: 2026-09-28

Ideas for us: Compare the fixed loop counts (2-16) of this Huginn-style model against our ACT halting and Poisson loop-depth sampling to evaluate stability and convergence. Test the Parcae LTI input reinjection mechanism from this repo against our current implementation to see if it improves gradient flow in deeper recurrent blocks.

### IFM/LoopedLM-P2-huginn-s-fixed-pln5
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-s-fixed-pln5 · relevance 10 · read: abstract

The model IFM/LoopedLM-P2-huginn-s-fixed-pln5 is a looped language model using fixed loop counts (2-16 loops), a Huginn-style architecture, and is part of a paper on looped LMs. It is available as safetensors under an Apache-2.0 license.

Key numbers: 2-16 loops

Ideas for us: Test fixed loop counts of 2, 5, and 10 on our nano_mythos architecture to compare convergence against ACT halting; implement ELT-style early-exit self-distillation using the fixed-loop model as a teacher for the recurrent block.

### IFM/LoopedLM-P2-huginn-m-orthoinj
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-m-orthoinj · relevance 10 · read: abstract

The item is a model repository for a looped language model (LoopedLM-P2-huginn-m-orthoinj) associated with a paper part 2. It is tagged as a looped-language-model and references arxiv:2610.06833, indicating it focuses on looped or recurrent architectures for text generation.

Ideas for us: Test the orthoinj (orthogonal injection) mechanism from this model to stabilize deep recurrent blocks during our fixed loop count experiments (2-16 loops). Compare the halting behavior of this Huginn-style model against our ACT halting and Poisson loop-depth sampling to evaluate stability in deeper recurrent blocks.

### IFM/LoopedLM-P2-huginn-m-learned-entropy0p01
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-m-learned-entropy0p01 · relevance 10 · read: abstract

The model is a looped language model (LoopedLM) based on the Huginn architecture, featuring a learned entropy parameter of 0.01. It is part of the IFM paper-part-2 series and is available as safetensors under an Apache-2.0 license.

Key numbers: 0.01

Ideas for us: Test learned entropy scaling for Poisson loop-depth sampling in the nano_mythos project; compare fixed loop counts (2-16) against learned entropy halting behavior.

### IFM/LoopedLM-P2-huginn-m-fixed-r5
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-m-fixed-r5 · relevance 10 · read: abstract

The item is a model repository for a looped language model named LoopedLM-P2-huginn-m-fixed-r5, released by IFM on 2026-09-28. It uses a fixed loop count of 5 and is associated with the paper arxiv:2610.06833. The model is tagged as a looped-language-model and is available in safetensors format under an Apache-2.0 license.

Key numbers: 5 loops, 2026-09-28, arxiv:2610.06833

Ideas for us: Run a fixed-loop ablation study comparing loop counts of 2, 5, and 8 to evaluate the impact of depth on perplexity. Implement early-exit self-distillation on the fixed-r5 model to test if shallower exits can match the performance of the full 5-loop depth.

### IFM/LoopedLM-P2-huginn-m-huginn-linear
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-m-huginn-linear · relevance 10 · read: abstract

The repository hosts a looped language model checkpoint named LoopedLM-P2-huginn-m-huginn-linear, associated with paper arxiv:2610.06833. It is tagged as a looped-language-model and text-generation model under Apache-2.0 license.

Ideas for us: Compare fixed loop counts (2-16) against ACT halting on the nano_mythos prelude/recurrent block setup. Test ELT-style early-exit self-distillation using this model's linear Huginn variant as a teacher for the small recurrent block.

### IFM/LoopedLM-P2-huginn-m-fixed-pln5
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-m-fixed-pln5 · relevance 10 · read: abstract

The item is a Hugging Face model repository for a looped language model named LoopedLM-P2-huginn-m-fixed-pln5. It uses fixed loop counts (pln5) and is associated with a paper on looped language models. The model is tagged as a looped-language-model and is available under Apache-2.0 license.

Key numbers: pln5, 2026-09-28, arxiv:2610.06833

Ideas for us: Test fixed loop counts of 5 on our nano_mythos model to compare against ACT halting and Poisson sampling. Implement early-exit self-distillation using the fixed-loop model as a teacher for shallower student configurations.

### IFM/LoopedLM-P2-huginn-l-learned-entropy0
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-l-learned-entropy0 · relevance 10 · read: abstract

The repository is a looped language model based on the Huginn architecture, tagged as paper-part2 and focused on learned entropy. It is a safetensors model released under Apache-2.0, associated with arxiv:2610.06833.

Ideas for us: Test learned entropy gating to determine loop halting thresholds for fixed loop counts. Compare the effect of learned entropy-based early exits against ELT-style self-distillation on the nano_mythos prelude/coda ratios.

### IFM/LoopedLM-P2-huginn-l-learned-entropy0p01
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-l-learned-entropy0p01 · relevance 10 · read: abstract

The item is a Hugging Face model repository for a looped language model named LoopedLM-P2-huginn-l-learned-entropy0p01. It is tagged as a looped-language-model and associated with paper-part2 and arxiv:2610.06833. The model uses safetensors and is licensed under Apache-2.0.

Key numbers: 2026-09-28, 0.01

Ideas for us: Test learned entropy regularization with a target of 0.01 on the nano_mythos recurrent block to encourage stable halting. Compare fixed loop counts between 2 and 16 using the Huginn-style architecture to validate the Dory-style depth scaling.

### IFM/LoopedLM-P2-huginn-l-fixed-pln5
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-l-fixed-pln5 · relevance 10 · read: abstract

The repository IFM/LoopedLM-P2-huginn-l-fixed-pln5 is a looped language model implementing fixed loop counts (Dory-style) and Huginn-style architecture. It is tagged as a looped-language-model and is associated with arxiv:2610.06833.

Key numbers: 2-16 loops (from project context), 10M parameters (project scale), 1B parameters (planned scale), 2 DGX Spark GPUs (planned hardware), 8 times (previous ACT max loops), 2 layers (prelude).

Ideas for us: Run a fixed-loop ablation comparing 2, 5, and 16 loops on the nano_mythos 10M parameter model to measure perplexity vs. compute cost. Implement ELT-style early-exit self-distillation using the fixed-pln5 model as a teacher for the recurrent block.

### IFM/LoopedLM-P2-huginn-l-fixed-r5
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-l-fixed-r5 · relevance 10 · read: abstract

The item is a model repository for a looped language model based on Huginn, featuring fixed loop counts. It is tagged as a looped-language-model and is associated with paper-part2 and arxiv:2610.06833.

Ideas for us: Compare fixed loop counts (2-16 loops) against ACT halting to evaluate stability and performance trade-offs. Test ELT-style early-exit self-distillation using the fixed-loop model as a teacher for the recurrent block.

### IFM/LoopedLM-P1-moe-huginn-336b
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-336b · relevance 10 · read: abstract

The repository is a Mixture-of-Experts Looped Language Model named LoopedLM-P1-moe-huginn-336b. It is tagged as looped-language-model and paper-part1, indicating it is part of a research series on looped architectures for text generation.

Key numbers: 336b

Ideas for us: Test MoE routing within the shared recurrent block to see if expert specialization changes with loop depth. Compare fixed loop counts (2-16) against ACT halting to measure the impact on convergence and perplexity for a 10M parameter model.

### IFM/LoopedLM-P1-moe-huginn-500b
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-500b · relevance 10 · read: abstract

The repository IFM/LoopedLM-P1-moe-huginn-500b is a 500-billion parameter mixture-of-experts looped language model. It is tagged as a looped-language-model and paper-part1, released under Apache-2.0, and focuses on text generation.

Key numbers: 500b

Ideas for us: Test MoE routing within the shared recurrent block to scale parameter count without increasing compute per loop; compare fixed loop counts (2-16) against ACT halting on the nano_mythos prelude/coda architecture.

### Recursive CLIP: Vision-Language Distillation with Looped Transformer Blocks
Szymon Mazurek, Alberto Presta, Michał Byra · 2026-10-25 · http://arxiv.org/abs/2610.34563v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper investigates recursive CLIP models where a small set of N transformer blocks is applied M times to maintain a fixed computational depth of L=12 blocks. It compares rCLIP (vision-only recursion) and r²CLIP (vision and text recursion) using knowledge distillation from a DFN-2B ViT-B/16 teacher on CC3M and CC12M datasets. Results show CLIP can tolerate substantial block reuse with parameter efficiency, though performance varies by encoder and downstream task.

Key numbers: L=12 blocks; (N,M) ∈ {(1,12), (2,6), (3,4), (4,3), (6,2)}; Ref. CLIP 149.62M total params; rCLIP 1b12i 71.65M total; r²CLIP 1b12i 36.97M total; 13,874,493 image-caption pairs (CC3M + CC12M).

Ideas for us: Test fixed loop counts (2-16 loops) on nano_mythos using a small set of shared blocks with N*M fixed to match computational depth, similar to the rCLIP configurations. Implement early-exit self-distillation from a deeper non-recursive teacher to a looped student to see if it mitigates the performance drop seen in high-recurrence configurations like 1b12i.

### Test-Time Forecasting: Recurrent Depth Enables In-Context Dynamics Prediction
Jeffrey B. Lai, John Quinn, Anthony Bao, William Gilpin · 2026-10-14 · http://arxiv.org/abs/2610.19552v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper establishes a theoretical equivalence between looped linear transformers and Koopman operator theory for one-step forecasting of linear dynamical systems. It introduces Test-Time Forecasting (TTF), where a one-step forecaster uses iterations of gradient descent as recurrent depth to generalize to out-of-distribution dynamics during autoregressive rollout. The framework demonstrates that lightweight looped models can perform in-context forecasting of unseen ODEs and PDEs without massive parameter scale.

Key numbers: 10M-100M parameters; 2-16 loops (from project context, not paper); 8 loops (project context)

Ideas for us: Implement an iTTF block using linear attention with shifted keys/values and input injection to test if nano_mythos can learn ODE propagators in-context. Compare fixed loop counts (2, 4, 8, 16) on a small synthetic ODE task to measure the relationship between recurrent depth and spectral forecasting error.

### Looped Actor: Depth-Recurrent Reasoning Models for Reinforcement Learning
T. Konstantin Rusch, Tim Seyde, Jared Boyer, Zach J. Patterson, Daniela Rus · 2026-09-29 · http://arxiv.org/abs/2609.37432v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper introduces Looped Actor, a transformer-based RL policy that repeatedly applies a shared block to refine a latent representation and halts when the action distribution stabilizes. It provides a complexity-theoretic proof that state-adaptive computation can achieve optimal returns with asymptotically less expected computation than fixed-runtime policies. Experiments across 22 tasks in six environments show it matches or exceeds an untied baseline with 16x more parameters, particularly in tasks requiring multistep planning.

Key numbers: 22 tasks, 6 environments, 16x more parameters, Boxoban success rate roughly twice the baseline, arXiv:2609.37432v1

Ideas for us: Implement a state-dependent halting criterion for nano_mythos based on latent action-distribution stability (similar to Looped Actor) to replace or supplement ACT halting. Test if loop count correlates with task difficulty (e.g., sequence length or puzzle complexity) in a small-scale language reasoning benchmark to validate adaptive depth allocation.

### Beyond the Training Horizon: Mechanisms and Limits of Length Generalization in Looped Transformers
Jia Liang, Xi Jin, Liangming Pan · 2026-09-27 · http://arxiv.org/abs/2609.33144v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper compares Matched-Recurrence and Decoupled-Recurrence Looped Transformers on length generalization across polynomial iteration, finite-state composition, and knowledge-graph traversal. It finds that both models use distinct recurrent mechanisms but share a representational principle where states encode content and computational status via live/consumed and fresh/aged residual directions. Length generalization fails at extreme horizons due to state degradation and limited self-correction, and it does not necessarily imply faithful step-by-step reasoning.

Key numbers: 3 shared Transformer blocks, 2 attention heads per block, width 256, loop counts 2-16 (from project context), 1B-parameter run (from project context)

Ideas for us: Test fixed loop counts (2-16) on polynomial iteration and finite-state composition to identify the horizon where readout state degradation or progress cue failure occurs. Probe for live/consumed and fresh/aged residual directions in the nano_mythos recurrent block to see if they causally control information participation beyond the training horizon.

### Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models
Ke Wan, Chen Chen · 2026-09-23 · http://arxiv.org/abs/2609.27373v2 · relevance 9 · read: abstract+fulltext(14k chars)

The paper studies recurrent-depth language models and finds that attention routing (support and distribution) stabilizes significantly earlier than hidden states and outputs. It introduces WISE, a training-free method that uses unrestricted attention in early recurrent steps to discover a sparse block-structured working set, then reuses that routing support in later steps while keeping within-support attention and representation refinement dynamic. This reduces late-step attention routing cost from O(n^2) to O(nm) and achieves up to 1.76x late-step attention speedup over FlashAttention at 4K context while largely preserving performance on multi-hop QA benchmarks.

Key numbers: 1.76x late-step attention speedup over native FlashAttention at 4K context; O(n^2) to O(nm) routing cost reduction; 2K context quality preservation; 4K context speedup evaluation

Ideas for us: Implement WISE-style block-structured working-set discovery in nano_mythos: use full attention for the first 1-2 recurrent loops to identify a sparse block support, then restrict subsequent loops to that support while keeping queries/keys/values and attention weights dynamic. Compare fixed loop counts (2-16) with and without WISE support reuse on a small multi-hop QA subset, measuring attention speedup and perplexity/accuracy tradeoffs.

### Mimir: A Sub-2B Hierarchical Reasoning Model Trained on Permissible Data Punches Above Its Weight Class
Peter Schneider-Kamp, Jacob Nielsen, Gianluca Barmina, Danial Namazifard, Annemette Brok Pirchert, Kenneth Enevoldsen · 2026-08-13 · http://arxiv.org/abs/2608.13517v3 · relevance 9 · read: abstract+fulltext(14k chars)

Mimir is a 1.78B-parameter Hierarchical Reasoning Model (HRM-Text) trained from scratch on permissible data. It uses a recurrent-depth architecture with interleaved low-level (L) and high-level (H) stacks. The paper reports that varying L-stack applications leads to graceful degradation, while changing H-stack applications breaks the model.

Key numbers: 1.78B parameters; 70.5B tokens per epoch (Mimir v1); 103B tokens per epoch (Mimir v1.5); 161 datasets; outperforms HRM-Text 1B, Ouro 1.4B, and Ouro 2.6B; competes with Qwen 3.5 9B and Gemma 4 E4B.

Ideas for us: Test if splitting nano_mythos's recurrent block into separate low-level and high-level stacks with an interleaved schedule improves reasoning; run an ablation on fixed loop counts to determine if high-level recurrence is more sensitive to depth changes than low-level recurrence in our architecture.

### Cross-Loop Query Handoff: How Later Loops Redirect Two-Hop Retrieval in the Ouro Looped Language Models
Ziyu Mu, Zehua Wang · 2026-10-25 · http://arxiv.org/abs/2610.34639v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper analyzes Ouro looped language models (1.4B and 2.6B) by transplanting late-loop attention queries into earlier loops at the same layer, head, and position. It finds that injecting loop-3 queries into loop 1 rescues most two-hop retrieval failures that loop 3 corrects, while shuffled donors and late keys/values provide far less rescue. The results suggest a transferable collection of late-loop queries supports bridge-conditioned retrieval, though no single-vector bottleneck is established.

Key numbers: Ouro-1.4B: 24 layers, 16 heads, head dim 128, T=4 loops; Ouro-2.6B: 48 layers, 16 heads, head dim 128, T=4 loops. Rescue rates: 90% (1.4B) and 97% (2.6B) for late query vs 24% and 21% for shuffled donor. Loop-1 two-hop failure: 42% (1.4B) and 35% (2.6B). Chance: 25%.

Ideas for us: Test whether injecting queries from a later ACT-halting loop into an earlier loop of nano_mythos rescues two-hop retrieval failures at the same layer/head coordinates. Compare rescue rates using late queries from the same example versus shuffled donors and late keys/values to determine if query handoff improves composition in our small looped transformer.

### Hybrid Latent Attention for Looped Language Models
Yuhan Chen, Siyuan Zhang, Nan Wang, Feiyang Kang, Ruoxi Jia · 2026-10-06 · http://arxiv.org/abs/2610.07940v1 · relevance 9 · read: abstract+fulltext(14k chars)

Hybrid Latent Attention (HLA) reduces the KV cache of looped language models by storing exact keys and values for a sliding window of W recent tokens and compact latents for older tokens, which each loop reads directly without reconstruction. It is uptrained on frozen Ouro looped models (T=4) with 1.4B and 2.6B parameters, training only added maps to reproduce original attention. HLA shrinks the cache by 10.7x per token, increases concurrent sequences per GPU by 4.0-8.8x, and improves decoding throughput by 2.5x at 1K contexts and up to 7.4x at 16K, while retaining over 97% accuracy on math, knowledge, and reasoning benchmarks.

Key numbers: T=4; 1.4B and 2.6B parameters; 10.7x cache shrink per token; 768 to 72 KiB on smaller model; 4.0-8.8x more concurrent sequences; 2.5x throughput at 1K tokens; 7.4x throughput at 16K tokens; over 97% accuracy on MATH500, MMLU-Pro, BBH; 96-100% on RULER up to 16K tokens; 1,200 uptraining steps; 80 GB GPU holds only five 16K-token sequences of Ouro-1.4B; 13.6 ms at 1K context to 49.3 ms at 16K for single sequence.

Ideas for us: Implement HLA-style latent caching for nano_mythos fixed-loop counts (2-16) to test if cache compression scales with loop depth and whether ACT-halting models benefit from loop-specific latents; run a small ablation comparing exact KV cache vs HLA latent cache on fixed loop counts D=4 and D=8, measuring perplexity, cache size, and decoding throughput on a 10M-parameter model.

### Exploring Recurrent-Depth Language Models for Multitask Clinical Text Processing
Xi Yang, Guodong Liu, Cheng Peng, Kaleb Smith, Yingying Zhang, Xing He · 2026-10-20 · http://arxiv.org/abs/2610.27559v1 · relevance 9 · read: abstract

The paper evaluates small recurrent-depth language models for multitask clinical NLP, comparing models like Ouro1.4B-Thinking and HRM-Text-1B against standard transformers across 13 datasets. It finds that recurrent-depth models, particularly Ouro variants, achieve higher performance on inference and question answering tasks compared to similarly sized standard transformers. The authors suggest further controlled experiments to isolate architectural effects of recurrent computation on reasoning and generalization.

Key numbers: 1.4B, 2.6B, 1.7B, 8B, 1B, 9 training datasets, 4 evaluation datasets, 13 total datasets, 11/12 datasets where Ouro-1.4B-Thinking outperformed Qwen3-1.7B

Ideas for us: 1. Fine-tune nano_mythos on a subset of clinical datasets (e.g., one extraction and one QA task) using fixed loop counts (2, 4, 8) to measure if deeper recurrent blocks improve clinical reasoning over shallow baselines. 2. Run an early-exit self-distillation experiment where nano_mythos exits at varying loop depths on easy clinical examples and distills the final-depth representation to earlier exits, measuring accuracy vs. compute trade-off.

### One Block, Multiple Depths: Recurrent Vision Transformers with Depth-Programmed Experts
Adrian Bulat, Yassine Ouali, Georgios Tzimiropoulos · 2026-10-08 · http://arxiv.org/abs/2610.12448v1 · relevance 9 · read: abstract+fulltext(14k chars)

reViT replaces a ViT's depth-wise stack with a single recurrent Transformer block. It uses a normalized depth coordinate to softly mix a small bank of shared FFN experts into one dense FFN per recurrent step, allowing elastic-depth inference by resampling the same coordinate grid. The paper compares weight-space merging against token-dispatch and output-mixture MoE alternatives under a one-FFN-per-depth compute budget, reporting 73% fewer stored parameters for reViT-B/16 while matching DeiT III accuracy on ImageNet-1k.

Key numbers: 73% fewer stored parameters (reViT-B/16 vs DeiT III); E=4 experts for from-scratch training; E=8 experts for DINOv2 distillation; normalized depth coordinate s_t = t/(L-1); tau=1; one dense FFN evaluated per depth; L recurrent depths; k=4 distinct blocks in Raptor comparison.

Ideas for us: Implement a depth-programmed FFN expert bank in nano_mythos's recurrent block, gated by normalized loop depth t/(D-1), and compare weight-space merging against MoD token routing at fixed loop counts (2-16) on a small language task. Train nano_mythos with Poisson loop-depth sampling using a continuous normalized-depth coordinate so the same checkpoint can be deployed at multiple fixed loop counts without retraining, then materialize the merged FFNs per depth for fixed-depth inference.

### Invariant-Measure Reasoners: Stable Representations for Latent Reasoning
Yuto Inui, Takuya Konishi, Yoshinobu Kawahara · 2026-10-07 · http://arxiv.org/abs/2610.10996v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper introduces Invariant-Measure Reasoners (ImR), a framework for latent reasoning models that repeatedly apply a recurrent block to a latent state. It addresses prediction instability across recurrent depths by predicting from the expectation of the prediction head's output under an invariant measure (the long-run distribution of latent states) rather than a single final state. The approach applies to both fine-tuning existing models and training from scratch, and is validated on maze and Sudoku tasks where it improves accuracy and allows high performance even with non-fixed-point dynamics.

Ideas for us: 1. Replace final-state prediction in nano_mythos with an empirical invariant measure by averaging the prediction head's logits over the recurrent trajectory (e.g., across ACT halting steps or fixed loop counts 2-16) to stabilize predictions across depths. 2. Add an early-exit self-distillation loss that matches the per-loop prediction head output to the trajectory-averaged (invariant-measure) prediction, encouraging stable representations at all recurrent depths.

### FlexLoop: Depth-Elastic Looped Policies for Adaptive Test-Time Computation in Deep RL
Xun Wang, Ruishuo Chen, Yu Chen, Zhuoran Li, Longbo Huang · 2026-09-28 · http://arxiv.org/abs/2609.34488v1 · relevance 9 · read: abstract+fulltext(14k chars)

FlexLoop is a post-training framework that converts pretrained fixed-depth looped RL policies into depth-elastic policies. It preserves full-depth capability via the original RL objective while using adjacent-depth policy distillation to make shallower recurrent depths reliable. The framework also enables state-wise adaptive inference using recurrent-depth consistency as a stopping signal.

Key numbers: 30 environments, 43% average recurrent depth reduction, 1.34x wall-clock speedup

Ideas for us: Test if fixed-depth trained nano_mythos models exhibit recurrent-depth specialization by evaluating performance at loop counts 2-16. Implement adjacent-depth distillation between consecutive loop counts to enable elastic inference and adaptive early exiting based on hidden-state consistency.

### Think Deep, Speak Once: Relit, A Recursive Latent Implicit Transformer Framework
Abhishek Panwar, Maheep Singh, Saksham Bansal · 2026-08-08 · http://arxiv.org/abs/2608.08113v1 · relevance 9 · read: abstract+fulltext(14k chars)

ReLIT is a hybrid framework that augments a frozen TinyLlama-1.1B backbone with a lightweight trainable recursive block. It iteratively refines latent thinking states (z) and answer states (y) over N_latent iterations and T deep recursions using a residual state refinement strategy. The framework avoids BPTT via a Recall-Then-Learn protocol to maintain a minimal memory footprint during deep recurrence.

Key numbers: TinyLlama-1.1B; N_latent iterations; T deep recursions

Ideas for us: Test a frozen TinyLlama backbone with a small trainable recursive block using fixed loop counts (2-16) to compare against nano_mythos's ACT halting; implement a residual state refinement update (z_{n+1} = z_n + delta) instead of direct state replacement in the recurrent block.

### Block Recursive Transformers for Structured Parameter Sharing
Josep Marques, Hyunyoung A. Cho, Youngeun Kim, Youjia Zhang, Aecheon Jung, Kyuhong Shim · 2026-10-18 · http://arxiv.org/abs/2610.24998v1 · relevance 9 · read: abstract+fulltext(14k chars)

Block Recursive Transformers (BRT) propose partitioning shared layers into contiguous blocks that are recursively reused before moving to the next block, serving as an intermediate topology between layer-wise repetition and global cyclic reuse. The method preserves depth-wise transitions while enabling local refinement within blocks without adding routing modules, auxiliary states, or extra training objectives. Evaluations across decoder-only models up to 1.7B parameters show BRT improves validation negative log-likelihood and downstream accuracy over static recursive sharing baselines.

Key numbers: 1.7B parameters, 1 <= S <= P, 2-16 loops (from user context)

Ideas for us: Test BRT block partitioning on nano_mythos by grouping the 1-layer shared recurrent block into contiguous blocks of varying sizes (e.g., S=1, 2, 4) while keeping fixed loop counts of 2-16. Compare validation NLL and CKA similarity patterns between BRT block granularity and the current global cyclic reuse to measure local refinement vs. depth-wise differentiation.

### Tiny Recursive Message Passing
Tommaso Salvatori, Matteo Forasassi, Alexis Toumi, Giovanni de Felice · 2026-10-13 · http://arxiv.org/abs/2610.18649v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper applies tiny recursive model training procedures—detached recursion, deep supervision, and adaptive halting—to message passing neural networks. It demonstrates that these methods enable parameter-efficient models, achieving 99% accuracy on Sudoku-Extreme with only 16k parameters. The authors also show that these models can generalize zero-shot to larger instances of N-Queens and SAT.

Key numbers: 16k parameters, 99% test accuracy on Sudoku-Extreme, 10 minutes training on H100, 9 hours training on CPU, 27M parameters (baseline), 800k parameters (baseline)

Ideas for us: Test detached recursion with deep supervision on the nano_mythos shared recurrent block to stabilize deeper fixed loop counts (2-16). Implement per-vertex halting logits with a smooth minimum as an alternative to the current ACT halting mechanism.

### Squeezing More from Limited Data with Recursive Transformers
Serdar Gülbahar, Lukas Edman, Alexander Fraser · 2026-08-27 · http://arxiv.org/abs/2608.26973v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper studies recursive Transformers that reuse a shared block across depth to decouple per-token computation from representational capacity under limited data budgets. It trains models across 10M to 100M word budgets and finds that recursive models outperform standard Transformers at both 10M and 100M words while remaining competitive with BabyLM Challenge 2025 winners. The authors also use factorized embeddings to reduce vocabulary-map parameters, which dominate small model budgets.

Key numbers: 10M-100M word budgets; 13M-1.2B parameter models; 12 layers; hidden sizes 256 to 3072; E values 96, 192, 352 for H=256, 384, 512; 10 epochs; 10M words ≈ 13M BPE tokens.

Ideas for us: Run fixed loop-count sweeps (2-16 loops) on 10M/100M word subsets to find the optimal compute-to-parameter ratio for our nano_mythos architecture. Compare factorized embeddings (E << H) against standard embeddings in our prelude block to reduce vocabulary parameter overhead at the 10M parameter scale.

### bioMoR: Biology-Guided Mixture-of-Recursions for Effective Genomic Learning
Koushik Howlader, Tirtho Roy, Md Tauhidul Islam, Wei Le · 2026-08-07 · http://arxiv.org/abs/2608.06727v2 · relevance 9 · read: abstract+fulltext(14k chars)

bioMoR applies Mixture-of-Recursions (MoR) to genomic learning, using a shared recursive Transformer block where a router determines token-specific recursion depths. It integrates biological knowledge at three sites: embedding smoothing, attention bias toward related tokens, and a graph-aware router for recursion depth. The framework reports improved macro-F1 and balanced accuracy with fewer parameters and FLOPs than non-recursive Transformers across eight omics benchmarks.

Key numbers: 8.2 percentage points macro-F1 improvement, 7.1 percentage points balanced accuracy improvement, 75% fewer parameters, up to 58% fewer FLOPs, 8 benchmarks, 5-fold cross-validation, max depth K, 0.92 linear-probe accuracy

Ideas for us: Test a graph-aware router using gene co-expression or pathway adjacency to set per-token loop counts in nano_mythos; compare fixed loop counts (2-16) against biology-guided adaptive depths on a small genomic or synthetic token-interaction task.

### Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It
Zehao Jin, Ruixuan Deng, Junran Wang · 2026-09-29 · http://arxiv.org/abs/2609.36585v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper shows that pretrained transformers follow only 1.4-3.6 reference lines by default, and extra loops add little. A rank-8 LoRA at one early layer, with all other weights frozen, extends this reach to 50 lines in one pass and at least 160 lines with loops in Ouro-1.4B. The LoRA initiates a relay where program lines carry chain identity through middle layers, and frozen heads read progressively further up the chain.

Key numbers: 13 base models; reach 1.4-3.6 lines (median 2.2); rank-8 LoRA; 65,537 added parameters in Qwen3-8B; Qwen3-8B 24-line accuracy 15.5% to 99%; Ouro-1.4B 60 lines after 4 loops, at least 160 after 8 loops; Ouro-1.4B reach 0, 1.6, 2.3, 2.2, 2.2 lines after 1-5 loops; Huginn 1.4-1.6 lines after 8, 16, or 32 recurrences.

Ideas for us: Test whether a rank-8 LoRA at a specific early layer in nano_mythos extends reference-chain reach under fixed loop counts (2-16). Measure if the LoRA's benefit persists when applied only in the first loop versus every loop, and locate the last useful intervention layer for our recurrent block.

### Prediction Dynamics in Depth-Recurrent Language Models
Xinyue Luo, Fei Yu · 2026-09-18 · http://arxiv.org/abs/2609.21383v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper analyzes depth-recurrent language models, deriving a margin characterization that decomposes score updates into common translation, direction relative to the winner, and competitor pairing. It shows that intermediate answers can agree with the endpoint despite changing scores, and that accounting for update direction and pairing reduces the earliest qualifying depth by 22.5-34.4% across Huginn-3.5B and Ouro-1.4B. It also separates common and contrast motion in predictive distributions, showing how preference-change share can grow while absolute updates shrink.

Key numbers: 22.5-34.4% reduction in mean earliest qualifying depth; Huginn-3.5B; Ouro-1.4B

Ideas for us: 1. Apply the paper's translation/direction/pairing decomposition to nano_mythos intermediate readouts at fixed loop counts (2-16) to measure how much score movement is common vs contrastive and correlate with early-exit triggers. 2. Test ELT-style early-exit self-distillation using the sharp margin criterion (winner reserve R > 0) as a halting signal instead of ACT, comparing fixed loop depths against adaptive halting on validation loss and answer stability.

### Training-Free Hidden-State Refinement for Flow-Matching Image Generators
Yuanyi Yan, Xinzhe Rao, Canyu Shen, Yang Chen, Yunlu Chen, Meng Tang · 2026-08-29 · http://arxiv.org/abs/2608.29160v1 · relevance 9 · read: abstract+fulltext(14k chars)

The paper presents a training-free inference-time looping framework for frozen flow-matching image generators. It repeatedly applies selected transformer layers within each denoising call using Dense or Sparse Token Loops, controlled by Sampling-Progress Gating and loop layer ranges. Loop count and strength manage repeated updates, while Loop Guidance combines ordinary and looped vector-field predictions to improve quality metrics like GenEval and DPG-Bench without changing model weights or the outer sampler.

Key numbers: GenEval: 0.4471 to 0.5691; DPG-Bench: 0.7656 to 0.8053; Scale-RAE DiT2.4B; lambda_loop = 1; eta = 1/K; K = 1 recovers ordinary layer output.

Ideas for us: Test fixed loop counts (2-16) on the nano_mythos recurrent block to see if internal hidden-state refinement improves LM performance without weight updates. Implement a 'Loop Guidance' style mechanism for nano_mythos by combining vector predictions from a standard pass and a looped pass of the shared recurrent block.

### saitejautpala/looped_udlm
saitejautpala · 2026-10-21 · https://huggingface.co/saitejautpala/looped_udlm · relevance 9 · read: abstract

The repository saitejautpala/looped_udlm is a PyTorch model implementing a looped transformer for discrete-diffusion language modeling. It utilizes uniform diffusion and adaptive computation for text generation, trained on the openwebtext-gpt2-packed-1024 dataset under an Apache-2.0 license.

Ideas for us: Test a looped transformer architecture on a discrete-diffusion objective instead of standard next-token prediction. Compare fixed loop counts (2-16) against ACT halting to evaluate stability and convergence in a diffusion-language setting.

### summerAIO/Longbook-Looped-8-4-pre
summerAIO · 2026-10-20 · https://huggingface.co/summerAIO/Longbook-Looped-8-4-pre · relevance 9 · read: abstract

The repository is a quantized (4-bit and 8-bit) model tagged as 'looped' and 'indexed-loop-execution'. It is based on Qwen3_5 and supports image-text-to-text generation. It includes custom code for its architecture.

Key numbers: 4-bit, 8-bit

Ideas for us: Test 4-bit and 8-bit quantization on the nano_mythos recurrent block to evaluate the impact on ACT halting stability. Implement indexed-loop-execution logging to compare fixed loop counts (2-16) against Poisson sampling.

### j-llm/LoopedQwen3.8-27B-Sparkle
j-llm · 2026-10-18 · https://huggingface.co/j-llm/LoopedQwen3.8-27B-Sparkle · relevance 9 · read: abstract

The repository j-llm/LoopedQwen3.8-27B-Sparkle is a 27B parameter looped language model based on Qwen3, released under Apache 2.0. It is a model card entry from October 2026 associated with the US region.

Key numbers: 27B parameters

Ideas for us: Distill a 27B looped teacher into our 10M nano_mythos student using fixed loop counts between 2 and 16; compare early-exit self-distillation across different prelude and coda layer ratios.

### quandao10/looptrans-looped_5x4_v2
quandao10 · 2026-10-13 · https://huggingface.co/quandao10/looptrans-looped_5x4_v2 · relevance 9 · read: abstract

The item is a Hugging Face model repository named looptrans-looped_5x4_v2 by quandao10, dated 2026-10-13. It is tagged with diffusers and safetensors and is hosted in the US region. The repository name indicates a looped transformer configuration with 5 loops and 4 layers.

Key numbers: 5 loops, 4 layers, 2026-10-13

Ideas for us: Test a fixed loop count of 5 with 4 recurrent layers to compare against the current 1-layer shared block looped up to 8 times. Run an early-exit self-distillation experiment using the 5x4 configuration as the teacher for a smaller fixed-depth student model.

### eac123/ptr-recurrent-depth
eac123 · 2026-10-12 · https://huggingface.co/eac123/ptr-recurrent-depth · relevance 9 · read: abstract

The repository is a model implementation focused on recurrent depth for length generalization and pointer-chasing tasks. It is built in vanilla PyTorch and references several arXiv papers related to recurrent architectures and depth. The project is tagged as research and ml-intern, indicating it serves as an experimental baseline for recurrent-depth mechanisms.

Ideas for us: 1. Compare fixed loop counts (2-16) against Poisson loop-depth sampling on pointer-chasing tasks to evaluate stability. 2. Test ELT-style early-exit self-distillation using the recurrent block to see if early halting improves convergence on small-scale LMs.

### StarpowerTechnology/WVY-Liquid-Recurrent-Depth
StarpowerTechnology · 2026-09-08 · https://huggingface.co/StarpowerTechnology/WVY-Liquid-Recurrent-Depth · relevance 9 · read: abstract

The repository describes a model using recurrent depth with a gated DeltaNet architecture, trained from scratch for text generation. It is tagged as experimental and references arXiv papers 2412.06464 and 2502.05171, focusing on liquid recurrent depth for language modeling.

Ideas for us: Compare fixed loop counts (2-16) against the gated DeltaNet recurrent-depth baseline on a small validation set. Test ELT-style early-exit self-distillation using the recurrent block's intermediate states as supervision targets.

### xquantize/nemotron-labs-dory1.0-4b-mlx-6bit
xquantize · 2026-10-26 · https://huggingface.co/xquantize/nemotron-labs-dory1.0-4b-mlx-6bit · relevance 9 · read: abstract

The item is a 6-bit MLX quantized version of NVIDIA Nemotron-Labs Dory1.0-4B, a looped transformer base model. It is distributed as Safetensors for Apple Silicon and tagged for text generation and conversational use. The repository references the BF16 base model and its quantized variant under an 'other' license.

Key numbers: 4B parameters, 6-bit quantization, 2026-10-26 release date

Ideas for us: Benchmark the fixed-loop-count behavior of the 4B Dory model across 2-16 loops to validate depth scaling trends for nano_mythos. Use the MLX weights on Apple Silicon to run early-exit self-distillation experiments comparing shallow versus deep recurrent block exits.

### ProCreations/Dory-Beam-4B-Accurate
ProCreations · 2026-10-26 · https://huggingface.co/ProCreations/Dory-Beam-4B-Accurate · relevance 9 · read: abstract

The repository hosts Dory-Beam-4B-Accurate, a 4B-parameter looped transformer trained for latent reasoning and beam search. It uses looped-transformer architecture, latent reasoning, distillation, and is trained on Beam-Reasoning-Traces for math, text generation, and conversational tasks in English.

Key numbers: 4B parameters

Ideas for us: 1. Replicate the beam-search decoding strategy on our nano_mythos model to evaluate if fixed loop counts improve reasoning accuracy on small-scale math tasks. 2. Apply ELT-style early-exit self-distillation using a teacher model trained on synthetic reasoning traces to accelerate convergence of our recurrent block.

### ProCreations/Dory-Beam-4B-GGUF
ProCreations · 2026-10-24 · https://huggingface.co/ProCreations/Dory-Beam-4B-GGUF · relevance 9 · read: abstract

The item is a GGUF-quantized release of Dory-Beam-4B, a looped transformer for text generation. It is tagged with looped-transformer and beam, and lists ProCreations/Dory-Beam-4B as its base model. The excerpt provides no architectural details, loop counts, or performance metrics.

Key numbers: 4B

Ideas for us: Replicate Dory-style fixed loop counts (2-16) in nano_mythos and compare against ACT halting and Poisson sampling on perplexity and compute. Distill a deeper recurrent block from the 4B Dory-Beam teacher into the small looped block using ELT-style early-exit self-distillation.

### ProCreations/Dory-Beam-4B
ProCreations · 2026-10-24 · https://huggingface.co/ProCreations/Dory-Beam-4B · relevance 9 · read: abstract

ProCreations released Dory-Beam-4B, a 4-billion parameter looped transformer for latent reasoning. It is trained on the Beam-Reasoning-Traces dataset and supports beam search for text generation and conversational tasks. The model is distributed as safetensors and is tagged with dory, looped-transformer, and distillation.

Key numbers: 4B parameters

Ideas for us: Test fixed loop counts (2-16) on nano_mythos using a synthetic reasoning trace dataset similar to Beam-Reasoning-Traces. Implement ELT-style early-exit self-distillation from a deeper teacher loop count to a shallower student loop count.

### litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B
litert-community · 2026-10-22 · https://huggingface.co/litert-community/NVIDIA-Nemotron-Labs-Dory1.0-4B · relevance 9 · read: abstract

The repository is for NVIDIA-Nemotron-Labs-Dory1.0-4B, a 4-billion parameter model tagged as looped, ngpt, and reasoning. It is described as an on-device/edge model for text generation, with a BF16 base model and a quantized variant available.

Key numbers: 4B

Ideas for us: Test fixed loop counts of 2-16 loops in the nano_mythos recurrent block to mirror Dory-style depth scaling. Compare the impact of deeper recurrent blocks against varying prelude/coda layer ratios on a small-scale subset.

### cloudnathan5/Dory1.0-4B-GGUF
cloudnathan5 · 2026-10-19 · https://huggingface.co/cloudnathan5/Dory1.0-4B-GGUF · relevance 9 · read: abstract

The repository is a GGUF quantized version of NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, a 4-billion parameter base model. It is tagged as a looped-transformer and supports text generation in English, utilizing llama.cpp and imatrix quantization.

Key numbers: 4B, 2026-10-19

Ideas for us: Compare fixed loop counts (2-16) on nano_mythos against the Dory architecture's depth scaling behavior. Implement ELT-style early-exit self-distillation using the 4B Dory model as a teacher for the small recurrent block.

### netprojects/nemotron-dory1.0-4b-mlx-bf16
netprojects · 2026-10-17 · https://huggingface.co/netprojects/nemotron-dory1.0-4b-mlx-bf16 · relevance 9 · read: abstract

The repository contains the MLX-quantized bf16 version of NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, a 4-billion parameter looped language model tagged for reasoning and text generation. It is explicitly categorized as a looped-language-model and is based on the Dory architecture.

Key numbers: 4B parameters, bf16 precision

Ideas for us: Compare fixed loop counts (2-16) against ACT halting on the 10M parameter nano_mythos to measure perplexity and compute trade-offs; implement ELT-style early-exit self-distillation using the 4B Dory model as a teacher for the small recurrent block.

### netprojects/nemotron-dory1.0-4b-mlx-8bit
netprojects · 2026-10-17 · https://huggingface.co/netprojects/nemotron-dory1.0-4b-mlx-8bit · relevance 9 · read: abstract

The item is a Hugging Face repository for an 8-bit MLX quantized version of the NVIDIA Nemotron-Labs Dory1.0-4B model. It is tagged as a looped language model and is based on the Nemotron Dory architecture, which focuses on reasoning and text generation.

Key numbers: 4B parameters, 8-bit quantization

Ideas for us: 1. Compare the fixed loop counts (2-16) of the Nemotron Dory architecture against our ACT halting and Poisson loop-depth sampling to evaluate stability and performance.
2. Analyze the Dory model's implementation of looped depth to inform our scaling strategy for the planned 1B-parameter run on DGX Spark GPUs.

### KristianS7/Ouro-2.6B-Thinking
KristianS7 · 2026-10-17 · https://huggingface.co/KristianS7/Ouro-2.6B-Thinking · relevance 9 · read: abstract

Ouro-2.6B-Thinking is a looped language model designed for reasoning and chain-of-thought generation. It utilizes recurrent-depth architecture to enable deeper processing within a single forward pass, focusing on thinking and conversational capabilities.

Key numbers: 2.6B parameters

Ideas for us: Test if increasing fixed loop counts (2-16) improves performance on complex reasoning tasks compared to ACT halting. Implement ELT-style early-exit self-distillation to see if shallow exits can mimic the reasoning depth of deeper loops.

### KristianS7/huginn-0125
KristianS7 · 2026-10-04 · https://huggingface.co/KristianS7/huginn-0125 · relevance 9 · read: abstract

KristianS7/huginn-0125 is a Huginn Raven model available as safetensors on Hugging Face. It is tagged for text generation, code, math, reasoning, and conversational tasks, and is associated with the tomg-group-umd/huginn-dataset. The repository includes custom code, indicating a specialized architecture implementation.

Ideas for us: Implement the Huginn Raven architecture from the custom code to compare its fixed loop-depth behavior against our ACT halting and Poisson sampling. Run a small-scale ablation on the 10M parameter nano_mythos using the Huginn dataset to evaluate if fixed loop counts (2-16) improve reasoning and math performance compared to dynamic halting.

### IFM/LoopedLM-P2-huginn-m-learned-entropy0
IFM · 2026-09-28 · https://huggingface.co/IFM/LoopedLM-P2-huginn-m-learned-entropy0 · relevance 9 · read: abstract

The repository is a looped language model (LoopedLM) based on the Huginn architecture, identified as part of a paper series (paper-part2). It is tagged with looped-language-model and xllm, indicating a focus on recurrent depth and looped structures for language modeling.

Ideas for us: Compare the learned entropy halting mechanism of this model against our Poisson loop-depth sampling to evaluate stability and convergence. Test if the Huginn-style looped block can be integrated into our nano_mythos prelude/coda ratio experiments for the 1B-parameter run.

### TAFFY: A Task-Adaptive Tabular Foundation Model with In-Context Diversity
Zijian Li, Xiangchen Song, Gongxu Luo, Jie Qiao, Ruichu Cai, Zhenhao Chen · 2026-10-06 · http://arxiv.org/abs/2610.07559v1 · relevance 8 · read: abstract+fulltext(14k chars)

TAFFY introduces a Task-Conditioned Looped Transformer that iteratively applies a shared group of Transformer blocks to refine contextual representations for tabular in-context learning. It uses a task-conditioning module that maps support-dataset statistics to a scalar gate modulating the final hidden-state update. The model also employs an In-Context Diversity Prior that samples from multiple related environments derived via controlled interventions and distribution shifts on a shared causal process. Across six classification and five regression benchmark datasets, TAFFY attains the lowest average rank.

Key numbers: 6 classification datasets, 5 regression datasets, 999 predicted quantiles

Ideas for us: Test a task-conditioned scalar gate on the final hidden-state update of nano_mythos's recurrent block, modulated by support-set statistics. Experiment with synthetic pretraining contexts that mix observations from multiple related environments created via distribution shifts and interventions to improve in-context learning.

### Universal interpolation for deep residual self-attention networks
Sibylle Marcotte, Joan Bruna · 2026-10-01 · http://arxiv.org/abs/2610.01981v1 · relevance 8 · read: abstract+fulltext(14k chars)

The paper proves universal interpolation for deep residual self-attention networks using a fixed finite set of frozen attention blocks, where expressivity comes from depth and composition order rather than layer-specific parameters. It specifically motivates this work through looped transformers and recurrent-depth architectures. It establishes that two frozen random single-head noncausal attention blocks suffice for universal interpolation on a connected domain of full measure, and characterizes causal masking restrictions.

Key numbers: two frozen single-head blocks; number of blocks grows linearly with batch size; causal interpolation ruled out when batch size exceeds token dimension; loop counts 2-16 (project context, not paper)

Ideas for us: Test whether nano_mythos can interpolate a fixed synthetic dataset using only its frozen shared recurrent block by varying loop count, sign, and duration per sequence, per the paper's fixed-generator framework. Run a small ablation comparing ACT halting versus fixed loop depths (2-16) on interpolation tasks to measure how depth-only expressivity scales with parameter sharing.

### Nanbeige4.2-3B on Apple Silicon: Fixing Deployment Bugs and Decreasing Looped Transformer Memory Overhead
John T. Halloran · 2026-08-14 · http://arxiv.org/abs/2608.13987v1 · relevance 8 · read: abstract+fulltext(14k chars)

The paper analyzes a 3B-parameter Looped Transformer (LT) that reuses a single stack of layers for a second forward pass to add effective depth without increasing parameters. It identifies five deployment bugs on Apple Silicon and demonstrates that LTs double peak attention memory during prefill compared to non-looped models. To mitigate this, the authors introduce a chunked-prefill strategy that bounds peak attention memory and extends allowable context width by 2.7x on 32 GiB of shared memory.

Key numbers: 3B parameters, 2 effective passes, 2.7x context width extension, 32 GiB shared memory, 256 token chunk size, 30% task completion on MCPMark, 1024 to 12,244 token prompt lengths

Ideas for us: 1. Implement chunked prefill for nano_mythos to reduce peak attention memory during the prefill phase of looped blocks. 2. Benchmark the memory overhead of fixed loop counts (2-16) versus ACT halving to determine the optimal depth-to-memory trade-off for the 1B-parameter run.

### DeepThink: A Conceptual Model for Knowledge-Addressable Recurrent Language Modeling
Weiguo Chen · 2026-10-12 · http://arxiv.org/abs/2610.16171v1 · relevance 8 · read: abstract+fulltext(14k chars)

DeepThink is a knowledge-addressable recurrent language modeling framework that separates a Knowledge Bank, state-conditioned Knowledge Activator, Knowledge Assembly, and a Shared Thinking Operator. It treats knowledge capacity (N), activation width (K), and recurrent depth (R) as independent design variables and provides an exact Fixed-Schedule Realization of a fixed-depth model, plus shared-base reparameterization and pretrained output-tail reuse for initialization. The paper formalizes state-dependent selection, parameter-space assembly, and thinking followed by generation as a route from pretrained models to dynamically assembled recurrent computation.

Key numbers: N=12, K=3, R=3 (illustrative example in Figure 2); 1 ≤ K ≤ N; R ≥ 1; D ≥ 1 backbone depth; ℓ ∈ {0, …, D}; arXiv:2610.16171v1; 12 October 2026

Ideas for us: Implement a fixed-loop ablation (D=2,4,8,16) on nano_mythos using DeepThink’s exact Fixed-Schedule Realization to compare against ACT/Poisson loop-depth sampling and measure whether shared recurrent blocks with typed Knowledge Units match fixed-depth Transformer outputs. Initialize the shared recurrent block via exact shared-base reparameterization from a pretrained 2-layer prelude and reuse the pretrained output tail, then test state-dependent top-K Knowledge Unit selection (K=1,2,4) within the MoD routing to see if activation width trades off against recurrent depth at constant compute.

### Bypass Observation: A Conceptual Design of a Non-Intrusive Layer-Wise Semantic Extraction Architecture
Haibin Tong, Jiang Yu · 2026-09-12 · http://arxiv.org/abs/2609.13807v1 · relevance 8 · read: abstract+fulltext(14k chars)

The paper proposes Bypass Observation, a non-intrusive architecture that attaches read-only observation heads to selected Transformer layers to expose intermediate predictive distributions without altering the backbone. It derives a closed-form overhead approximation of V/(12d) for full-vocabulary readout and discusses cost-reduction strategies like sparse observation and top-k readout. It specifically analyzes the value of this architecture for looped and recurrent-depth Transformers, suggesting iteration-wise readout can expose convergence, oscillation, and halting signals.

Key numbers: V/(12d); 30% to 240% overhead; 12d^2 backbone cost per layer; dV observation head cost; 2.4 x 10^4 tokens context length for d=4096

Ideas for us: Attach shared or layer-specific bypass readout heads to the shared recurrent block of nano_mythos to monitor semantic drift across fixed loop counts (2-16) and ACT halting steps. Use the iteration-wise readout distributions as an external signal for ELT-style early-exit self-distillation or as a non-intrusive halting criterion for the recurrent block.

### Universal Transformers for Circuit Computations: Perfect Length Generalization in Tiny Transformers
Takuya Ito, Ruchir Puri, Murray Campbell, Parikshit Ram · 2026-08-31 · http://arxiv.org/abs/2608.31067v1 · relevance 8 · read: abstract+fulltext(14k chars)

The paper presents a provably correct transformer parameterization for circuit computations that uses iterative depth reduction to evaluate expressions of arbitrary length. It employs a positional encoding to track gate depth and masked hard attention to identify evaluable subexpressions, terminating after d iterations for depth d problems. The model achieves perfect length generalization on Boolean algebra, modular arithmetic, and ListOps, with only 280 learnable parameters for Boolean tasks.

Key numbers: 280 learnable parameters (Boolean algebra), depth 1 and 2 training, 100% accuracy, O(n) per-iteration complexity, O(n*d) total complexity, embedding dimension of 7, vocabulary size 7

Ideas for us: Implement a fixed-loop-count baseline for nano_mythos using the paper's depth-tracking positional encoding to evaluate loop depth; test if hard-attention masking over the deepest recurrent layer improves length generalization on symbolic tasks.

### Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs
Aur Shalev Merin · 2026-10-20 · http://arxiv.org/abs/2610.28498v2 · relevance 8 · read: abstract+fulltext(14k chars)

The paper demonstrates that a copied digit is present in the latent states of both a looped model (Ouro-1.4B) and a standard transformer (Qwen 2.5-1.5B), but is difficult for standard single-prompt probes to read. Minimal-pair probes, which compare variants of the same prompt differing by one token, reveal the value with high accuracy (0.92-0.97), whereas standard probes achieve only 0.15-0.23 in Ouro and 0.12-0.14 in Qwen. The study finds that while intermediate values like a+b may be readable, they are not necessarily used by the model to generate the output.

Key numbers: Ouro-1.4B minimal-pair probe: 0.92-0.97; Ouro-1.4B standard probe: 0.15-0.23; Qwen 2.5-1.5B minimal-pair probe: 0.75-0.98 (mean 0.88); Qwen 2.5-1.5B standard probe: 0.12-0.14; 5,249 training questions; GSM8K accuracy rises from 6.5% after one loop to 23% after four loops; clamping the answer moves the written digit toward the other problem's in 91% of pairs.

Ideas for us: Implement a minimal-pair probing protocol for nano_mythos to test if specific computed values are present in the recurrent block's hidden states across different loop counts (2-16). Conduct clamping experiments on nano_mythos where a suspected intermediate value is replaced with a value from another problem to determine if the value is causally used or merely readable.

### Log-Depth Recurrent Language Modeling
Yiqin Wang, Nuri Cingillioglu, Charles Pert · 2026-09-23 · http://arxiv.org/abs/2609.28212v1 · relevance 8 · read: abstract+fulltext(14k chars)

The paper introduces AR-GRC, a log-depth recurrent language model using a balanced-tree parallel scan (up-sweep and down-sweep) to compute all prefix representations in O(log N) depth and O(N) work. It is trained autoregressively with next-token cross-entropy and uses a Gated Recursive Cell as the composition operator. Experiments on PTB, WikiText-2, and OpenWebText2 show it is competitive with ALiBi Transformers on smaller corpora, maintains stable perplexity when extrapolating to sequences nearly 6x longer than training length, and scales linearly in evaluation time versus quadratic Transformer scaling.

Key numbers: PTB: AR-GRC 14.34 vs ALiBi 13.34 vs Sinusoidal 12.68; WikiText-2: AR-GRC 90.74 vs ALiBi 85.49 vs Sinusoidal 94.30; OpenWebText2: AR-GRC 46.08 ± 0.06 vs ALiBi 34.43 ± 0.03 vs Sinusoidal 37.32 ± 0.10; max training context 512 tokens; extrapolation tested up to 2,976 tokens (~5.8x); AR-GRC ~5x faster than Transformers at max evaluated length on WikiText-2; parameter targets 3.5M (PTB), 20M (WikiText-2), 100M (OpenWebText2).

Ideas for us: Replace the shared recurrent block in nano_mythos with a balanced-tree parallel scan (up-sweep/down-sweep) so each loop computes all prefix representations in log depth, then compare fixed loop counts (2-16) against the scan-based depth on a small PTB/WikiText-2 subset. Run an early-exit self-distillation ablation where shallower tree levels produce exit targets for deeper levels, measuring whether log-depth prefix computation improves length extrapolation beyond 512 tokens compared to the current linear recurrent loop.

### quandao10/looptrans-looped_5x4_infiloop_mixdistill
quandao10 · 2026-10-28 · https://huggingface.co/quandao10/looptrans-looped_5x4_infiloop_mixdistill · relevance 8 · read: abstract

The item is a Hugging Face model repository named 'looptrans-looped_5x4_infiloop_mixdistill' by user quandao10, dated 2026-10-28. It is tagged with diffusers and safetensors and located in the us region. The abstract indicates it is a model repo but provides no further technical details.

Ideas for us: Train a small looped transformer with a fixed loop count of 5 and 4 layers per recurrent block, then compare it against the same architecture using ACT halting to measure the impact of fixed depth versus adaptive halting on perplexity. Apply mixed early-exit self-distillation across the 5 loop iterations and evaluate whether intermediate loop outputs improve final token prediction accuracy.

### quandao10/looptrans-looped_5x4_infiloop_mix
quandao10 · 2026-10-26 · https://huggingface.co/quandao10/looptrans-looped_5x4_infiloop_mix · relevance 8 · read: abstract

The repository quandao10/looptrans-looped_5x4_infiloop_mix is a model hosted on Hugging Face. It is tagged with diffusers and safetensors and is located in the US region. The naming convention suggests a looped transformer architecture with a 5x4 configuration.

Key numbers: 5x4

Ideas for us: Test a 5x4 looped block configuration (5 loops of 4 layers or 4 layers looped 5 times) to compare against the current 1-layer shared recurrent block. Experiment with 'infiloop' mixing strategies for residual connections or token routing within the loop iterations.

### summerAIO/LoopedQwen-GraphEngram-ja-en
summerAIO · 2026-10-20 · https://huggingface.co/summerAIO/LoopedQwen-GraphEngram-ja-en · relevance 8 · read: abstract

The repository is a looped Qwen model for Japanese-English text generation. It is tagged as experimental and includes custom code for graph neural networks and graph memory. It is described as a looped transformer with graph memory.

Ideas for us: Implement a graph-based memory reinjection mechanism similar to Parcae LTI to store and retrieve loop states. Test early-exit self-distillation by comparing hidden states at different fixed loop counts (2-16).

### muscgab/mangaocr-nar-looped-e4-exp
muscgab · 2026-10-13 · https://huggingface.co/muscgab/mangaocr-nar-looped-e4-exp · relevance 8 · read: abstract

The repository hosts a non-autoregressive looped transformer for Japanese manga OCR, trained on the animetext-ocr dataset. It uses safetensors and is licensed under CC-BY-NC-SA-4.0. The model is tagged as a looped transformer for image-to-text tasks.

Key numbers: e4

Ideas for us: 1. Test fixed loop counts (2-16) on a small image-to-text task to compare convergence against ACT halting. 2. Implement ELT-style early-exit self-distillation using the looped OCR architecture to evaluate depth efficiency.

### vladimir94/dory-1.0-4b-ane-coreml
vladimir94 · 2026-10-27 · https://huggingface.co/vladimir94/dory-1.0-4b-ane-coreml · relevance 8 · read: abstract

The repository hosts a 4-billion parameter looped transformer (NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) converted for Apple Neural Engine via CoreML. It uses 4-bit quantization and palettization for efficient deployment on Apple hardware and supports text generation and conversational tasks in English.

Key numbers: 4 billion parameters, 4-bit, 2026-10-27

Ideas for us: Benchmark fixed loop counts (2-16) on nano_mythos to compare convergence and perplexity against the Dory-style architecture; test ELT-style early-exit self-distillation using a smaller loop-depth student to mimic the 4B teacher's outputs.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-depth2-cmp32k-cfg3-156947-20261028
hi-todayis-jh · 2026-10-28 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-depth2-cmp32k-cfg3-156947-20261028 · relevance 8 · read: abstract

The repository is for a 1.4B parameter model named Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset. It is tagged as a looped-language-model and utilizes the verl framework for reinforcement learning via GRPO. The model name explicitly references a native depth of 2.

Key numbers: 1.4B parameters, depth 2

Ideas for us: Replicate the GRPO fine-tuning pipeline on nano_mythos using a math dataset to test if looped recurrent blocks benefit from RL-based reasoning. Compare the performance of a fixed loop count of 2 against ACT halting on a small-scale math reasoning task.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-cfg3-b128g16-r2-156947-20261026
hi-todayis-jh · 2026-10-26 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-cfg3-b128g16-r2-156947-20261026 · relevance 8 · read: abstract

The repository contains a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset. It is tagged as a looped-language-model and is configured for reinforcement learning using the verl framework. The configuration includes ragged batching and per-context settings.

Key numbers: 1.4b, 128, 16, 156947

Ideas for us: Test fixed loop counts between 2 and 16 on a small-scale version of the nano_mythos architecture to compare performance against ACT halting. Implement ELT-style early-exit self-distillation using a teacher model with a higher fixed loop count to guide a student model with fewer loops.

### hi-todayis-jh/ouro-1.4b-thinking-rltt-math-154711-20261008-rltt-fixed
hi-todayis-jh · 2026-10-21 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-rltt-math-154711-20261008-rltt-fixed · relevance 8 · read: abstract

The repository is a 1.4B parameter language model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned for math using RLTT (Reinforcement Learning with Test-Time Training). It is tagged as a looped-language-model and is specifically labeled as 'rltt-fixed', suggesting a fixed loop-depth configuration.

Key numbers: 1.4B parameters, 154711 training steps, date 20261008

Ideas for us: Compare the performance of the fixed-loop 1.4B model against our ACT-halting nano_mythos at similar parameter scales to evaluate if fixed depth (Dory-style) outperforms adaptive halting. Implement a fixed-loop version of our recurrent block (2-16 loops) and use the Ouro-1.4B-Thinking weights as a warm-start for the prelude and recurrent block.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-c0-an-cmp32k-154712-20261020
hi-todayis-jh · 2026-10-20 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-c0-an-cmp32k-154712-20261020 · relevance 8 · read: abstract

The repository is for a 1.4B parameter model fine-tuned from ByteDance/Ouro-1.4B-Thinking using reinforcement learning (verl) on the MATH-lighteval dataset. It is tagged as a looped language model and includes safetensors weights.

Key numbers: 1.4B parameters

Ideas for us: Compare fixed loop counts (2-16) on the Ouro-1.4B-Thinking architecture to evaluate depth scaling; implement ELT-style early-exit self-distillation using the 1.4B model as a teacher for the nano_mythos student.

### amiralifirouzi/Shahmaran-Ouro-1.4B-persian-lora
amiralifirouzi · 2026-10-17 · https://huggingface.co/amiralifirouzi/Shahmaran-Ouro-1.4B-persian-lora · relevance 8 · read: abstract

The repository is a 1.4B parameter Persian/Farsi model using LoRA and SFT for text generation and conversation. It is tagged with 'looped-transformer' and 'ouro', indicating an architectural link to looped or recurrent transformer designs. The model is distributed as PEFT safetensors and includes custom code.

Key numbers: 1.4B

Ideas for us: Compare the Shahmaran-Ouro looped architecture against our nano_mythos fixed loop counts (2-16) on a small Persian subset to evaluate depth scaling. Use the model's custom code as a reference for implementing ELT-style early-exit self-distillation in our recurrent block.

## Medium (5-7)

### TANGO: Treating Tokens as Operators
Joshua Nunley · 2026-08-22 · http://arxiv.org/abs/2608.22117v2 · relevance 7 · read: abstract+fulltext(14k chars)

TANGO introduces a token-mixing update where nonlinear source gates form the diagonal core of a linear operator applied to destination features. It reuses the same block parameters at every application, similar to a looped Transformer, and evaluates full-prefix and windowed variants across multiple datasets and depths (4, 8, and 16). Full-prefix TANGO achieves the lowest mean test NLL in all 16 matched settings, and its development loss does not increase as depth rises from 4 to 8 to 16, unlike the looped Transformer which increases in four settings.

Key numbers: 70M and 160M parameters; depths 4, 8, and 16; 16 settings; 288 training runs; 3.26 to 3.45x throughput of TANGO for the narrow-gate variant; 75% to 96% throughput of looped Transformer; 0.023 NLL difference in one setting; W=64 window; context 8,192 tokens.

Ideas for us: 1. Replace the shared recurrent block in nano_mythos with a TANGO-style gated operator to test if source-conditioned linear operators improve NLL at fixed loop counts (2-16). 2. Implement the three narrower gated-projection sets (first, middle, last) to reduce the O(T^2 F) cost of full-prefix TANGO and compare throughput against the current looped block on the DGX Spark GPUs.

### Structured Sparse Memory for Recurrent Reasoning
Zixuan Zhao, Samuel Wheeler, Neil Getty, Xiaotian Duan, Rick Stevens, Fangfang Xia · 2026-09-27 · http://arxiv.org/abs/2609.33270v1 · relevance 7 · read: abstract+fulltext(14k chars)

The paper introduces CHARM, a hybrid ARC model combining a recurrent backbone with structured task memory and synthetic data. It identifies that task-conditioned memory tables in XRM systems can exceed 30x the backbone size and introduces CoSE, a compositional sparse embedding that reduces these parameters by over 90% while improving accuracy. It also finds that recurrent depth only helps when balanced with the learning horizon and that synthetic data is a major driver of performance gains.

Key numbers: 84% pass@2 on ARC-AGI-1; 46.7% pass@2 on ARC-AGI-2; task table ~448M params (ARC-AGI-1) and ~610M params (ARC-AGI-2); CoSE uses 1/15 of learned task-memory parameters; continual puzzle adaptation with 512 parameters per task; 1.6M steps (ARC-AGI-1) and 2.38M steps (ARC-AGI-2) training budgets.

Ideas for us: Implement a compositional sparse embedding (CoSE) for task-conditioning in nano_mythos to reduce memory overhead while preserving instance-specific capacity. Test if balancing recurrent loop depth with a truncated learning horizon improves reasoning stability on small-scale synthetic puzzles.

### Dynamic System Inspired Continuous Large Language Model for AI4Science Tasks
Zhaoxin Fan, Zheng Zhu, Yuqing Ma, Faguo Wu, Jian Yang, Wenjun Wu · 2026-10-18 · http://arxiv.org/abs/2610.26959v1 · relevance 7 · read: abstract+fulltext(14k chars)

Continuum-LM converts pretrained residual LLM blocks into continuous-depth ODEs evaluated by numerical integration (RK2/RK4), reusing backbone parameters without expanding them. It is trained via continued pretraining and specialized with frozen-backbone LoRA across eight AI4Science tasks. At 0.6B and 1.7B scales, it improves mean exact-match accuracy by 7.77 and 10.96 percentage points over discrete counterparts and outperforms an Ouro-style recurrent control under matched evaluation budgets.

Key numbers: 0.6B, 1.7B, 7.77 percentage points, 10.96 percentage points, 20,000-step solver ablations, 100,000-step continued-pretraining budget, RK2, RK4, c=1

Ideas for us: Replace the fixed discrete loop count in nano_mythos with a continuous-depth ODE integration over the shared recurrent block, using RK2/RK4 stages as the loop-depth equivalent while keeping parameters shared. Compare a fixed-loop-count baseline against an adaptive integration-step/halting variant under matched residual-function evaluation budgets on the same small-scale pretraining corpus.

### quandao10/looptrans-looped_5x4_ldit_dir
quandao10 · 2026-10-21 · https://huggingface.co/quandao10/looptrans-looped_5x4_ldit_dir · relevance 7 · read: abstract

The item is a Hugging Face model repository by quandao10 named looptrans-looped_5x4_ldit_dir, tagged with diffusers and safetensors, hosted in the us region. It is a model repo rather than a paper or detailed technical write-up, and the provided excerpt contains no architecture, training, or loop-depth specifics.

Key numbers: 5x4

Ideas for us: Replicate a 5x4 looped configuration (5 loops, 4 layers per recurrent block) in nano_mythos and compare fixed-loop ACT variants; test LDIT-style directory/conditional routing with MoD token routing on a small language subset.

### nile-academy/nemotron-dory1.0-4b-gguf
nile-academy · 2026-10-19 · https://huggingface.co/nile-academy/nemotron-dory1.0-4b-gguf · relevance 7 · read: abstract

The item is a GGUF quantized release of NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16, a 4-billion parameter base model tagged for text generation and conversational use. It is associated with the Dory architecture, which relates to fixed loop counts, and is hosted by nile-academy with an 'other' license and US region endpoints.

Key numbers: 4B parameters, 2026-10-19 release date

Ideas for us: Use the 4B Dory model as a teacher for ELT-style early-exit self-distillation in the 10M parameter nano_mythos; compare fixed loop counts of 2-16 loops in nano_mythos against the Dory-style architecture to validate depth scaling behavior.

### Cenedril/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF
Cenedril · 2026-10-17 · https://huggingface.co/Cenedril/NVIDIA-Nemotron-Labs-Dory1.0-4B-GGUF · relevance 7 · read: abstract

The text identifies a GGUF repository for NVIDIA-Nemotron-Labs-Dory1.0-4B, a 4-billion parameter model. It lists tags including transformers, GGUF, PyTorch, and Nemotron Labs, along with associated datasets and a non-standard license. The model is associated with the 'Dory' naming convention referenced in the project's fixed loop count testing.

Key numbers: 4B, 2026-10-17

Ideas for us: Compare fixed loop counts (2-16) in nano_mythos against the Dory architecture's depth behavior; test if a 4B-scale fixed-depth configuration matches the performance of the dynamic ACT halting approach on a small subset.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-rltt-native-sd-cmp32k-cfg3-156947-20261029
hi-todayis-jh · 2026-10-29 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-rltt-native-sd-cmp32k-cfg3-156947-20261029 · relevance 7 · read: abstract

The repository is a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset. It is tagged as a looped-language-model and utilizes the verl framework for reinforcement learning. It is configured for native RLTT with supervised distillation.

Key numbers: 1.4B parameters, 32k context, 156947 steps

Ideas for us: Test the Ouro-1.4B-Thinking base model's loop-depth behavior under fixed loop counts (2-16) to compare against nano_mythos. Implement ELT-style early-exit self-distillation using the Ouro model's thinking traces as a teacher for the small recurrent block.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-ragged-sampled-cmp32k-cfg3-156947-20261028
hi-todayis-jh · 2026-10-28 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-ragged-sampled-cmp32k-cfg3-156947-20261028 · relevance 7 · read: abstract

The repository is a 1.4B parameter model fine-tuned from ByteDance/Ouro-1.4B-Thinking using VERL and GRPO with ragged sampled comparison. It is tagged as a looped language model and trained on the MATH-lighteval dataset. The model is available in safetensors format under an Apache-2.0 license.

Key numbers: 1.4B, 32k, 3, 156947

Ideas for us: Use the Ouro-1.4B-Thinking base as a reference for implementing a 1B-parameter looped block on DGX Spark GPUs; test fixed loop counts (2-16) on math reasoning tasks using GRPO to compare against the early-exit self-distillation approach.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-l1exact-cmp32k-cfg3-156947-20261024
hi-todayis-jh · 2026-10-24 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-l1exact-cmp32k-cfg3-156947-20261024 · relevance 7 · read: abstract

The repository contains a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset using verl and GRPO. It is tagged as a looped-language-model and uses native RL training with a 32k context window.

Key numbers: 1.4B, 32k, 156947, 20261024

Ideas for us: Replicate the 1.4B looped architecture on two DGX Spark GPUs to test scaling stability; apply fixed loop counts (2-16) to the math reasoning model to evaluate the impact of depth on GRPO convergence.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-rltt-native-cmp32k-cfg3-156947-20261022
hi-todayis-jh · 2026-10-22 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-rltt-native-cmp32k-cfg3-156947-20261022 · relevance 7 · read: abstract

The repository is for a 1.4B parameter model named Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset. It is tagged as a looped-language-model and is configured for reinforcement learning using the verl framework. The model is based on ByteDance's Ouro-1.4B-Thinking and uses Apache-2.0 licensing.

Key numbers: 1.4B, 32k, 156947, 20261022

Ideas for us: Compare the fixed loop counts (2-16) of this 1.4B model against our nano_mythos ACT halting and Poisson sampling to evaluate depth efficiency. Implement ELT-style early-exit self-distillation using the Ouro model's reasoning traces as a teacher for our smaller recurrent blocks.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-c0-an-cmp32k-cfg3-156947-20261022
hi-todayis-jh · 2026-10-22 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-c0-an-cmp32k-cfg3-156947-20261022 · relevance 7 · read: abstract

The repository is a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset using the verl RL framework. It is tagged as a looped-language-model and implements ragged batching for looped architectures. The model is configured for reinforcement learning and thinking/math tasks.

Key numbers: 1.4B parameters, 32k context (cmp32k), 156947 (config ID), 20261022 (date)

Ideas for us: 1. Implement ragged batching for variable loop depths across a batch to improve GPU utilization during fixed-loop testing. 2. Adapt the Ouro-1.4B-Thinking base model's looped structure to test ELT-style early-exit self-distillation on a small-scale nano_mythos variant.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-c1024-unb-ds4k-154711-20261020
hi-todayis-jh · 2026-10-20 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-c1024-unb-ds4k-154711-20261020 · relevance 7 · read: abstract

The repository is for a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset using the verl RL framework. It is tagged as a looped-language-model and includes safetensors weights. The model name indicates it uses ragged context handling and a specific context window of 1024.

Key numbers: 1.4B, 1024, 4k

Ideas for us: Compare the convergence speed and final math accuracy of this 1.4B looped model against our 10M parameter nano_mythos when scaled to similar depths. Implement a ragged-batch loop-count sampler for our ACT halting mechanism to handle variable loop depths per sequence more efficiently.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-rltt-native-ds4k-154712-20261018
hi-todayis-jh · 2026-10-18 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-rltt-native-ds4k-154712-20261018 · relevance 7 · read: abstract

The repository is for a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset. It is tagged as a looped-language-model and is configured for reinforcement learning via verl. The model is tagged with safetensors and Apache 2.0 license.

Key numbers: 1.4B, 4k, 154712, 20261018

Ideas for us: Compare the convergence and stability of the Ouro-1.4B-Thinking looped architecture against our nano_mythos fixed loop counts (2-16) on a small math subset. Implement ELT-style early-exit self-distillation using the Ouro model's intermediate loop outputs as teacher targets for our smaller recurrent block.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-ours-ds4k-154711-20261018
hi-todayis-jh · 2026-10-18 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-per-context-rb-ragged-ours-ds4k-154711-20261018 · relevance 7 · read: abstract

The repository contains a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the DigitalLearningGmbH/MATH-lighteval dataset. It is tagged as a looped-language-model and is configured for reinforcement learning using the verl framework. The model is provided in safetensors format under an Apache-2.0 license.

Key numbers: 1.4B, 20261018, 154711

Ideas for us: 1. Compare the convergence speed and final performance of the Ouro-1.4B-Thinking base against the nano_mythos 10M parameter model on the same MATH-lighteval dataset. 2. Implement ELT-style early-exit self-distillation on the Ouro-1.4B model to see if loop-depth can be reduced without significant loss in math reasoning accuracy.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-l1exact-ds4k-154712-20261016
hi-todayis-jh · 2026-10-16 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-l1exact-ds4k-154712-20261016 · relevance 7 · read: abstract

The repository is a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset using verl and GRPO. It is tagged as a looped-language-model and includes an exact l1 implementation, suggesting a focus on looped or recurrent depth for language models.

Key numbers: 1.4B parameters, 4k sequence length (ds4k), 154712 steps

Ideas for us: 1. Compare the convergence stability of the Ouro-1.4B looped architecture against our nano_mythos fixed loop counts (2-16) on a small math subset.
2. Implement ELT-style early-exit self-distillation using the Ouro-1.4B model as a teacher for our smaller recurrent blocks.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-ds4k-154711-20261016
hi-todayis-jh · 2026-10-16 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-ds4k-154711-20261016 · relevance 7 · read: abstract

The repository is a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset using verl and GRPO. It is tagged as a looped-language-model and uses native DeepSpeed 4k configuration.

Key numbers: 1.4B parameters, 4k context, 154711 steps

Ideas for us: 1. Compare the convergence stability of GRPO training on a 10M parameter looped transformer versus a standard transformer of equal size. 2. Implement early-exit self-distillation on the Ouro-1.4B base to determine if intermediate loop states can serve as effective supervision targets for smaller looped models.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-ragged-ds4k-154711-20261016
hi-todayis-jh · 2026-10-16 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-ragged-ds4k-154711-20261016 · relevance 7 · read: abstract

The repository is for a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset using VERL and GRPO with ragged DeepSpeed 4k. It is tagged as a looped-language-model and is associated with reinforcement learning.

Key numbers: 1.4B, 4k, 154711, 20261016

Ideas for us: Replicate the GRPO/VERL training pipeline on the nano_mythos model using a small math dataset to test if looped architectures benefit from reinforcement learning. Compare fixed loop counts (2-16) against ACT halting during the GRPO phase to determine if deeper recurrence improves mathematical reasoning.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-ragged-l1exact-ds4k-154712-20261016
hi-todayis-jh · 2026-10-16 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-ragged-l1exact-ds4k-154712-20261016 · relevance 7 · read: abstract

The repository is a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset using VERL and GRPO with ragged L1 exact loss. It is tagged as a looped-language-model and is associated with reinforcement learning and thinking math tasks.

Key numbers: 1.4B, 154712, 20261016

Ideas for us: Apply GRPO to reward loop-depth efficiency in our nano_mythos model; test if early-exit self-distillation improves math reasoning accuracy on small-scale MATH subsets.

### Jinyan0924/flux3-agibot-looped
Jinyan0924 · 2026-10-14 · https://huggingface.co/Jinyan0924/flux3-agibot-looped · relevance 6 · read: abstract

The repository hosts a looped DiT model based on Flux 3 Action, fine-tuned for robotics and video prediction tasks. It applies a looped architecture to an image-to-video diffusion transformer within the Agibot World ecosystem. The model is non-commercial and associated with arxiv:2609.40305.

Ideas for us: Test a fixed loop count (e.g., 4 or 8) on a small language modeling task to compare convergence against the shared recurrent block. Implement early-exit self-distillation using intermediate loop outputs as supervision targets for shallower passes.

### somenetworktrouble/NVIDIA-Nemotron-Labs-Dory1.0-4B-heretic-GGUF
somenetworktrouble · 2026-10-27 · https://huggingface.co/somenetworktrouble/NVIDIA-Nemotron-Labs-Dory1.0-4B-heretic-GGUF · relevance 6 · read: abstract

The item is a Hugging Face repository for a GGUF-quantized version of NVIDIA Nemotron Labs Dory 1.0 4B, described as a heretic variant. It lists tags for transformers, GGUF, PyTorch, NVIDIA, Nemotron Labs, Dory, and text generation, and references Nemotron pre-training and post-training datasets. No details about loop counts, recurrent depth, halting, or architecture are provided in the excerpt.

Key numbers: 4B

Ideas for us: Compare fixed loop counts 2-16 on nano_mythos against Dory-style fixed-depth baselines at small scale; distill a fixed-loop nano_mythos student from the 4B Dory model using early-exit self-distillation targets.

### snsnc/NVIDIA-Nemotron-Labs-Dory1.0-4B-mlx-4bit
snsnc · 2026-10-17 · https://huggingface.co/snsnc/NVIDIA-Nemotron-Labs-Dory1.0-4B-mlx-4bit · relevance 6 · read: abstract

The text identifies a quantized MLX version of NVIDIA Nemotron Labs Dory 1.0 4B, a 4-bit conversational model tagged with 'dory'. It references the base BF16 model and its 4-bit quantized variant, but provides no architectural details about fixed loop counts or loop-depth control.

Key numbers: 4B parameters, 4-bit quantization, loop counts 2-16 (from project context, not the excerpt)

Ideas for us: Test whether the Dory 4B model exhibits fixed loop-depth behavior by probing its internal recurrent structure; compare fixed loop counts of 2, 8, and 16 on nano_mythos using the same 10M-parameter scale to isolate depth effects from ACT halting.

### nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4
nvidia · 2026-10-16 · https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 · relevance 6 · read: abstract

The text identifies NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 as a 4-billion parameter model hosted on Hugging Face. It is tagged with 'dory', 'transformers', and 'text-generation', and is associated with Nemotron post-training and pre-training datasets. The model is available in safetensors and PyTorch formats under an 'other' license.

Key numbers: 4B parameters

Ideas for us: Compare fixed loop counts (2-16) in nano_mythos against the Dory-style architecture to determine optimal depth for small parameter counts. Test if the Dory model's fixed-loop approach can be distilled into the 10M parameter nano_mythos via ELT-style early-exit self-distillation.

### hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-r4-klreward-154712-20261016
hi-todayis-jh · 2026-10-16 · https://huggingface.co/hi-todayis-jh/ouro-1.4b-thinking-math-verl-grpo-native-r4-klreward-154712-20261016 · relevance 6 · read: abstract

The repository contains a 1.4B parameter model based on ByteDance/Ouro-1.4B-Thinking, fine-tuned on the MATH-lighteval dataset using verl and GRPO with a KL reward. It is tagged as a looped-language-model and uses safetensors.

Key numbers: 1.4B, 154712, 20261016

Ideas for us: Compare the convergence stability of GRPO training on a small looped transformer versus a standard transformer of similar size; test if the Ouro-1.4B base architecture's looped properties benefit from RL fine-tuning on math reasoning tasks.

## Low (0-4)

- [MergeOver: Post-Training Token Merging for Recursive Vision Transformers](http://arxiv.org/abs/2608.13141v1) (2026-08-13) rel 4
- [quandao10/looptrans-looped_5x4_infiloop](https://huggingface.co/quandao10/looptrans-looped_5x4_infiloop) (2026-10-22) rel 4
- [yujiepan/dory-1.0-tiny-random](https://huggingface.co/yujiepan/dory-1.0-tiny-random) (2026-10-25) rel 4
- [smdesai/NVIDIA-Nemotron-Labs-Dory1.0-4B-MLX-NVFP4](https://huggingface.co/smdesai/NVIDIA-Nemotron-Labs-Dory1.0-4B-MLX-NVFP4) (2026-10-18) rel 4
- [GitSwarm: Decentralized Compounding Inference](http://arxiv.org/abs/2610.04862v1) (2026-10-04) rel 3
- [PonderPounce: A Pretrained MLLM as an Episode Context Engine for Robot Control](http://arxiv.org/abs/2608.24115v2) (2026-08-25) rel 3
- [quandao10/looptrans-looped_5x4_infiloop_mixanchor](https://huggingface.co/quandao10/looptrans-looped_5x4_infiloop_mixanchor) (2026-10-27) rel 3
- [tiny-random/dory-1.0](https://huggingface.co/tiny-random/dory-1.0) (2026-10-25) rel 3
- [rdaum/decisive-dory](https://huggingface.co/rdaum/decisive-dory) (2026-10-21) rel 3
- [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-MXFP8) (2026-10-16) rel 3
- [Solvent: Long-Horizon Reasoning for Multi-Step Retrosynthesis](http://arxiv.org/abs/2610.32308v1) (2026-10-23) rel 2
- [quandao10/looptrans-looped_5x4_infiloop_mixmotion](https://huggingface.co/quandao10/looptrans-looped_5x4_infiloop_mixmotion) (2026-10-29) rel 2
- [quandao10/looptrans-looped_5x4_v2gt](https://huggingface.co/quandao10/looptrans-looped_5x4_v2gt) (2026-10-19) rel 2
- [JesseHufstetler/Dory1.0-4B-q4f16_1-MLC](https://huggingface.co/JesseHufstetler/Dory1.0-4B-q4f16_1-MLC) (2026-10-18) rel 2
- [quandao10/looptrans-looped_5x4_infiloop_wansyn](https://huggingface.co/quandao10/looptrans-looped_5x4_infiloop_wansyn) (2026-10-25) rel 1
- [quandao10/looptrans-looped_5x4_parcae](https://huggingface.co/quandao10/looptrans-looped_5x4_parcae) (2026-10-21) rel 1
- [quandao10/looptrans-looped_5x4_ilsd](https://huggingface.co/quandao10/looptrans-looped_5x4_ilsd) (2026-10-15) rel 1
- [Intent2Tc: Automated Intent-to-Traffic Control Translation with Language Models](http://arxiv.org/abs/2609.31397v1) (2026-09-25) rel 0
- [quandao10/looptrans-looped_5x4_ldit_infiloop](https://huggingface.co/quandao10/looptrans-looped_5x4_ldit_infiloop) (2026-10-23) rel 0
- [yichen-zhou927/Looped_Transformer](https://huggingface.co/yichen-zhou927/Looped_Transformer) (2026-10-21) rel 0
- [quandao10/looptrans-looped_5x4_ldit](https://huggingface.co/quandao10/looptrans-looped_5x4_ldit) (2026-10-20) rel 0
- [orangejuicesmith/13B-Ouroboros](https://huggingface.co/orangejuicesmith/13B-Ouroboros) (2026-10-24) rel 0
- [Learning to Compute Is Not Learning to Retain or Reuse in Looped Transformers](http://arxiv.org/abs/2610.19768v1) (2026-10-14) rel -1
- [Scheduling Recursive Reasoning in Looped Transformers](http://arxiv.org/abs/2609.36653v1) (2026-09-29) rel -1
- [Do Halting Heads Really Save Compute in Looped Language Models?](http://arxiv.org/abs/2610.32246v1) (2026-10-23) rel -1
