# PLAN — looped transformer research → 1B (one-page overview)

Last updated 2026-10-29. This file is the map. Details live in: SESSION_NOTES.md (daily log and **Next session**), DORY_NOTES.md (Dory comparison and videos), SCALE_PLAN.md (stages, compute, data), LITERATURE_SCAN_2026-10.md, DORY_SELF_PROBE.md, results.tsv.

## 1. Goal
Build a looped (recurrent-depth) transformer whose loop count is stable and predictable. Get compelling small-scale results first, then train a ~1B model on both Sparks over about 30 days.

## 2. Why
- Loop counts have been unstable so far. Our baseline loops a random number of times (Poisson depth sampling plus ACT halting), averaging about 2.3–3.5 of 8 loops. In RA-09 the halting collapsed to about 1.5 loops. Results varied, and most avenues came out neutral.
- NVIDIA Dory 1.0 4B (https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) works with a fixed, user-chosen loop count:
  - Card: "The shared Recurrent block (R) is applied multiple times between the Input and Output blocks, and the input is reinjected into every loop … Input and Output blocks have 8 layers each, while the Recurrent block has 20 layers."
  - Card: "We recommend two loops as the default … Most of the accuracy gain from looping comes from the second loop." Loop counts 1–4, one checkpoint for all of them.
  - Its published config.json and vLLM code add more detail: nGPT unit-norm residual updates, reinjection = norm(state + input), 3:1 sliding-window:full attention, and a default of 2 loops. See DORY_NOTES.md.
  - **Not published:** the loop count(s) used in training, fixed vs sampled, and any stability recipe.
- The ELT video/paper (DORY_NOTES.md) shows that training at one fixed loop count makes the model poor at other counts, unless you add self-distillation.

## 3. Key decisions (David and Grok, 2026-10-29)
- Fixed loops: test 4–8 at small scale, then 12–16 before 1B.
- Prelude/coda (before/after) layers grow towards Dory's 8:20:8 ≈ 2:5:2 ratio. We also explore looped-block depth.
- **Compute-matched comparisons:** main comparisons are FLOP-matched (±10% FLOPs/token), plus one matched-loop diagnostic arm per sweep. Record both val_bpb and FLOPs/token.
- RA-12 dropped.
- Small-model compute is cheap, so go broad and test hunches. We work as peers.

## 4. New knobs (training script, all default OFF = old baseline)
| Env var | Meaning |
|---|---|
| NANO_FIXED_LOOPS=N | exactly N loops in train and eval, no ACT halting, no Poisson sampling (RA-15) |
| NANO_RECUR_LAYERS=k | layers inside the looped block (RA-16) |
| NANO_PRELUDE_LAYERS / NANO_CODA_LAYERS | before/after layers (RA-16b) |
| NANO_ILSD=1 (+_WEIGHT, _MIN_LOOPS) | ELT-style self-distillation so one checkpoint works at several loop counts (RA-17) |
| NANO_LOOP_CHOICES=4,6,8 | sample the loop count per step, eval at max (RA-18) |
| NANO_SEED (planned) | seed control for stability checks |
Runner: `run_nano_arms.sh` (ARMS="tag:loops:recur:prelude:coda:choices:ilsd").

## 5. Research queue (in order; the daily session follows SESSION_NOTES "Next session")
1. RA-15 record (fixed 2/4/8 vs adaptive baseline; R0 = 2.288874, reproducing 2.289139)
2. RA-16 depth × loops at matched compute (8×1, 4×2, 2×4, plus diagnostic 8×2)
3. RA-16b layer ratio towards 2:5:2
4. RA-17 ILSD
5. RA-18 loop-count sampling {4,6,8}
6. RA-19 12–16 loops
7. Second seed on the winner, then Stage 1 of SCALE_PLAN.md
(RA-12 dropped. New ideas from the literature scan and the Dory probe get added here as RA-20+.)

## 6. Scale path (SCALE_PLAN.md)
- **Stages:** nano (~10M) → small (50–125M) → mid (300–400M, one Spark) → **1B on both Sparks, ~30 days, Muon**.
- **Go/no-go:**
  - **Leave nano:** beat the baseline by ≥0.005 val_bpb at matched FLOPs on 2 seeds, or match it with lower variance.
  - **Later stages:** beat an equal-FLOP non-looped model, with no shrinking gain as size grows.
- **1B estimate:** about 15 GFLOPs/token at 4 loops. Two Sparks × 40 TFLOPS × 30% × 30 days ≈ 6e19 FLOPs ≈ **3–4B tokens only** (well under Chinchilla). Memory is about 16 B/param of state plus per-loop activation checkpointing, which fits in 128 GB per Spark.

## 7. Pretraining data (Stage 2–3)
- **Base: Ai2 Dolma 3** (Paul Allen's Ai2 / OLMo, confirmed by David). Sample from dolma3_mix-150B-1025. Licence ODC-By.
- **Proposed mix:**

  | Share | Source |
  |---|---|
  | 45% | Dolma 3 (base) |
  | 25% | Nemotron-CC-v2/v2.1 HQ |
  | 10% | maths |
  | 10% | code |
  | 10% | synthetic STEM, late |

- **Licence caveat:** the NVIDIA data agreement allows commercial training but bans redistributing the data, and Qwen/DeepSeek/Phi terms may attach to models trained on it. Review before any public weight release.
- **Storage:** stream and tokenise about 10 GB of shards per Spark. Never download the full datasets.
- **A/B test first:** a small model on the current climbmix data vs the new mix.
- The current 8192-token vocabulary is too small for 1B; plan a 32–64k tokenizer.

## 8. Auxiliary work
- Literature scan (last 3 months): scripts/lit_scan.py → LITERATURE_SCAN_2026-10.md, summarised by Nemotron-3.5-Super-VL on Spark 2 (the Hermes vLLM endpoint), with David's OK.
- Dory self-probe: DORY_SELF_PROBE.md. Anything the model claims beyond the card is a **hypothesis to test**, never a fact.

## 9. Daily routine
One session at 9:52 UK:
1. Record the previous runs (results.tsv plus a SESSION_NOTES Results/Decision).
2. Implement the next avenue and smoke-test it.
3. Launch the runs detached.
4. Wait within about a 1–1.5 h GPU budget.
5. Record what finished.
6. Commit (push only if David asks) and report.

Runs that don't finish are recorded first thing the next day. The 11:22 wrap-up routine is retired.

## 10. Constraints
- Training runs only on Spark 1.
- Spark 2 is for model hosting and auxiliary LLM tasks only, when David OKs it. David starts and stops Spark 2 jobs.
- `sparkrun list` / `sparkrun run <code>` on Spark 1 boots alternative models (e.g. Qwen 3.8 27B FP8). Not during training.
- No reboots without David's OK. No force-push or history rewrites. One GPU job at a time.


## 2026-10-29 Auxiliary LLM change
- Dory (nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4, served names `dory` + full id) on Spark 2 :8000 is now the preferred auxiliary LLM for autoresearch tasks.
- Nemotron 3.5 Super VL retired from the server (weights remain on disk; launch script backup ~/served-model-backups/ on Spark 2).
- Fallback: Spark 1 `sparkrun` (e.g. Qwen 3.8 27B FP8) when Spark 2 is unavailable.
- RA-15 R3 (fixed 8 loops) val_bpb 2.287173 (best of R0-R3); record in results.tsv next session.
- Dory A/B ideas: DORY_AB_2026-10-29.md (hypotheses, candidate RA items).
