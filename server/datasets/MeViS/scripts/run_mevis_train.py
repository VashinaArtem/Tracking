#!/usr/bin/env python3
"""
Launcher for MeViS fine-tuning with explicit GPU selection.

Example:
  python datasets/MeViS/scripts/run_mevis_train.py --gpu 0
  python datasets/MeViS/scripts/run_mevis_train.py --gpu 1 --config configs/minicpmv4_5_mevis.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
LLAMA_FACTORY_DIR = REPO_ROOT / "LLaMA-Factory"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLaMA-Factory training on selected GPU.")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU id(s) for CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/minicpmv4_5_mevis.yaml",
        help="Config path relative to LLaMA-Factory dir, or absolute path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = LLAMA_FACTORY_DIR / config_path
    config_path = config_path.resolve()

    if not LLAMA_FACTORY_DIR.exists():
        raise FileNotFoundError(f"LLaMA-Factory directory not found: {LLAMA_FACTORY_DIR}")
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = ["llamafactory-cli", "train", str(config_path)]
    print(f"[run_mevis_train] CUDA_VISIBLE_DEVICES={args.gpu}")
    print(f"[run_mevis_train] Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(LLAMA_FACTORY_DIR), env=env, check=True)


if __name__ == "__main__":
    main()

