#!/usr/bin/env python3
"""RA smoke test: run nano_mythos_train.py with a tiny step budget.

Sets the environment the nano harness reads, then execs the training script as
a subprocess (the train script runs its whole pipeline at import, so we must
pass config via env, not CLI args). Used to validate a new avenue compiles,
loads data, and trains a few steps before committing to a full 1200-step run.

Usage:
    python training/smoke_test_train.py --steps 50 --batch_size 4 [--extra NANO_X=1 ...]
"""
import argparse
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument(
        "--extra", nargs="*", default=[],
        help="Extra NANO_* env assignments, e.g. --extra NANO_RAD_LO=1 NANO_MOD=1",
    )
    ap.add_argument("--tag", default="SMOKE")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_script = os.path.join(here, "training", "nano_mythos_train.py")

    env = dict(os.environ)
    env.update(
        {
            "NCCL_P2P_DISABLE": "1",
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "NANO_SEQ_LEN": "512",
            "NANO_MICRO_BATCH": str(args.batch_size),
            "NANO_DEVICE_BATCH": str(args.batch_size),
            "NANO_GRAD_ACCUM": str(args.batch_size),
            "NANO_MAX_STEPS": str(args.steps),
            "NANO_EVAL_EVERY": str(max(10, args.steps // 5)),
            "NANO_LOG_EVERY": "5",
            "NANO_LEARNING_RATE": "0.0003",
            "NANO_WARMUP_STEPS": "10",
            "NANO_GRAD_CLIP": "1.0",
            "NANO_MAX_LOOP_ITERS": "8",
            "NANO_PARCAE_E_NORM": "1",
            "NANO_PARCAE_DEPTH_SAMPLE": "1",
            "NANO_PARCAE_INIT": "1",
            "NANO_ROPE_LOOP": "1",
            "NANO_MOR": "0",
            "NANO_MOD": "1",
            "NANO_MOD_CAPACITY": "0.5",
            "NANO_MOD_NOISE": "0.1",
            "NANO_CKPT_TAG": args.tag,
        }
    )
    for k in args.extra:
        if "=" in k:
            key, val = k.split("=", 1)
            env[key] = val
        else:
            env[k] = "1"

    print(f"[smoke] steps={args.steps} batch={args.batch_size} extra={args.extra}")
    print(f"[smoke] env: " + " ".join(f"{k}={env[k]}" for k in (
        "NANO_RAD_LO", "NANO_MOD", "NANO_LT2") if k in env))
    rc = subprocess.run([sys.executable, train_script], env=env, cwd=here).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
