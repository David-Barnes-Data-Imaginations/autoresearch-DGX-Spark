# Archived one-off runners (reference only — DO NOT EXECUTE)

These scripts were single-avenue or early-iteration runners from the research
program, superseded by the canonical runners at the repo root. They are kept
for provenance (exact flags used in logged experiments) — see `results.tsv`
and `SESSION_NOTES.md` for the runs they produced.

- `run_ra06_*` — RA-06 Parcae experiments (completed, adopted)
- `run_ra08_*` — RA-08 RoPE loop-index experiments (completed, adopted)
- `run_ra01_*`, `ra01_monitor.sh` — RA-01 Mixture-of-Recursions (completed, neutral)
- `run_batch*` — early batch-experiment attempts, superseded by `run_exp_spark.sh`
- `run_experiment.sh` — DO NOT USE via SSH (escaped-quote `DOCKER_RUN` breaks; see skill docs)
- `run_exp.sh`, `run_spark_exp.sh`, `run_inline_patch.sh`, `run_nano_mythos_inline.sh`
  — early variants, superseded
- `train_baseline.py` — frozen baseline snapshot (reproducible from git history)

## Canonical runners (repo root)

- `run_nano_mythos_spark.sh` — standard Nano-Mythos training run (Docker, Spark 1)
- `run_exp_spark.sh` — hyperparameter-sweep runner (`sed -f` patch pattern)
- `run-dgx.sh` / `monitor-dgx.sh` — upstream DGX tooling
