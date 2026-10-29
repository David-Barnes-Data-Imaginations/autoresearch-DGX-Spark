# Dory self-knowledge probe — 2026-10-29

**Question:** can a model reveal Dory training details beyond the model card?
**Model that answered:** nemotron-3.5-super-vl (NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4), served by the Spark 2 vLLM endpoint that Hermes uses (http://100.82.91.70:8000/v1, thinking off). **Dory itself was NOT queried**: it isn\x27t being served, though BF16 + NVFP4 weights and vLLM recipes exist on Spark 2 (~/spark-vllm-docker/recipes/dory-4b-{bf16,nvfp4}-4loop.yaml). Serving it would mean starting a second server next to Nemotron, which only David should do.
Method: 15 prompts (11 questions, 2 canaries, 2 verbatim-recall), each 3× at temperature 0 and 1× at temperature 1. System prompt allowed "unknown". Script scripts/dory_probe.py, raw answers logs/dory_probe_raw.json.

## Result
All 60 answers were exactly "unknown": the questions, both canaries (made-up details) and both verbatim-card requests.
| Measure | Result |
|---|---|
| Consistency (t=0 ×3, t=1) | 100% identical ("unknown") |
| Matches card | 0/11 |
| Beyond card | 0/11 |
| Contradicts card | 0/11 |
| Canary confabulation | 0/2 (good) |
| Verbatim card recall | 0/2 (no memorisation) |

**Interpretation:** Nemotron-3.5-Super-VL has no knowledge of Dory (most likely its training cutoff is before Dory\x27s 16 Oct 2026 release) and it calibrated well, with no made-up answers. This confirms our expectation: weights only hold what was in the training text, and a model trained before the release can\x27t know it. Asking Dory about itself would at best reproduce card/blog text from its post-training data. NVFP4 vs BF16 shouldn\x27t change knowledge.

## Where the real beyond-card information came from
The published **config.json** and the **vLLM dory.py** implementation (see DORY_NOTES.md, "Dory config.json + vLLM implementation") do show details the card doesn\x27t:
- nGPT normalised slerp residuals with learnable alpha
- reinjection = norm(state + input) every loop
- 3:1 sliding-window:full attention
- default 2 loops

None of it reveals the *training* loop schedule.

## Testable hypotheses (from the published code, not model claims)
| Hypothesis | Card fact | Status | Test (RA candidate) |
|---|---|---|---|
| Unit-norm states (nGPT) stabilise looping | "normalized-GPT" backbone | published code, effect untested | RA-20: env-gated unit-norm state + slerp residual alpha in the looped block vs RMSNorm baseline |
| Simple additive reinjection norm(state+e) is enough | "input is reinjected into every loop" | published code | RA-21: Dory-style reinjection vs our Parcae LTI injection, at fixed loops |
| Training covered several loop counts (it works at 1–4) | one checkpoint for 1–4 loops | inference only, unverified | RA-17/18 (ILSD; sampling {4,6,8}) |

## Optional follow-up (needs David)
If David starts a Dory server on Spark 2 (one of the recipes above), re-run scripts/dory_probe.py with M set to the Dory model id, once for BF16 and once for NVFP4, and compare against the card.


## Dory self-probe (2026-10-29): BF16 vs NVFP4 vs card vs Nemotron
Model served on Spark 2 :8000 (per_loop KV, 2 loops). Same questionnaire, 3x temp0 + 1x temp1. Logs: logs/dory_probe_{bf16,nvfp4}.json.

| Q | Card | Nemotron | Dory BF16 | Dory NVFP4 | Score |
|---|---|---|---|---|---|
| Q0 looped? | yes, 8/20/8 recurrent | unknown | says NOT looped | says NOT looped | contradicts |
| Q1/Q2 train loops | per card | unknown | unknown | unknown | no info |
| Q3 inference loops | configurable, recommend 2 | unknown | vague | vague/unknown | no info |
| Q4 layers | 8/20/8 | unknown | no structure | "Nemotron arch", no counts | no info |
| Q5 reinjection | justnorm(state+input) | unknown | unknown | unknown | no info |
| Q6 halting | fixed count | unknown | fixed, no ACT | fixed, no ACT (t0 & t1) | matches (likely chance) |
| Q8/Q9 data/context | per card | unknown | unknown | unknown | no info |
| Q11 beyond-card | GRPO | unknown | claims RLHF | generic, nothing | contradicts (BF16) |
| Canaries C1/C2 | fake | refused | refused | refused | good |
| Verbatim V1/V2 | - | none | none | none | no recall |

BF16 t1 confabulations (autoencoder+MoE, "4,096 loops") recorded as hypotheses only; none credible. NVFP4 was more consistent/refusing. Conclusion: Dory has no usable self-knowledge of its architecture; quantisation does not change that.
