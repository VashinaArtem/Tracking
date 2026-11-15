"""
bbox_monitor.py

Daemon that watches the lora_v2 checkpoint directory and periodically
evaluates new checkpoints via bbox_eval_checkpoint.py (GIoU on 500 valid_u frames).

The evaluator is launched as a subprocess so that vLLM GPU memory is fully
released after each evaluation run.

Usage:
    python bbox_monitor.py --gpu 1
    python bbox_monitor.py --gpu 1 --eval_interval 18000 --poll_interval 300

Defaults:
    eval_interval  = 18000 steps  (evaluate every 18 saved checkpoints)
    poll_interval  = 300 seconds  (check for new checkpoints every 5 minutes)
    batch_size     = 16
    gpu            = 1
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR      = Path(__file__).resolve().parent
EVAL_SCRIPT     = SCRIPT_DIR / "bbox_eval_checkpoint.py"

LLAMA_SAVES_DIR = Path(__file__).resolve().parents[3] / "LLaMA-Factory" / "saves"
CHECKPOINT_DIR  = LLAMA_SAVES_DIR / "minicpmv4_5" / "mevis" / "lora_v2"

DEFAULT_CSV     = SCRIPT_DIR / "bbox_eval_log.csv"

DEFAULT_EVAL_INTERVAL = 18000   # steps between evaluations
DEFAULT_POLL_INTERVAL = 300     # seconds between directory scans
SUBPROCESS_TIMEOUT    = 1800    # 30 min max per evaluation run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def latest_checkpoint_step(checkpoint_dir: Path) -> int:
    """Returns the highest step number found in checkpoint_dir, or -1."""
    if not checkpoint_dir.exists():
        return -1
    steps = []
    for d in checkpoint_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                steps.append(int(d.name.split("-")[-1]))
            except ValueError:
                pass
    return max(steps) if steps else -1


def checkpoint_path_for_step(checkpoint_dir: Path, step: int) -> Path:
    return checkpoint_dir / f"checkpoint-{step}"


def read_last_evaluated_step(csv_path: Path) -> int:
    """Reads the CSV and returns the maximum step already evaluated (or -1)."""
    if not csv_path.exists():
        return -1
    max_step = -1
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s = int(row["step"])
                    if s > max_step:
                        max_step = s
                except (KeyError, ValueError):
                    pass
    except Exception:
        pass
    return max_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monitor lora_v2 checkpoints and evaluate with GIoU via vLLM"
    )
    parser.add_argument("--gpu",           type=str, required=True,
                        help="CUDA_VISIBLE_DEVICES value for the evaluator subprocess, e.g. '1'")
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL,
                        help="Minimum step gap between evaluations (default: 18000)")
    parser.add_argument("--poll_interval", type=int, default=DEFAULT_POLL_INTERVAL,
                        help="Seconds between checkpoint directory scans (default: 300)")
    parser.add_argument("--csv_path",      type=str, default=str(DEFAULT_CSV),
                        help="Path to output CSV log file")
    parser.add_argument("--batch_size",    type=int, default=16,
                        help="Batch size passed to bbox_eval_checkpoint.py")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINT_DIR),
                        help="Directory containing checkpoint-NNNN subdirs")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    csv_path       = Path(args.csv_path)

    # On restart: resume from where we left off
    last_evaluated = read_last_evaluated_step(csv_path)
    print(
        f"[monitor] Starting  |  checkpoint_dir={checkpoint_dir}  |  gpu={args.gpu}  |  "
        f"eval_interval={args.eval_interval}  |  last_evaluated_step={last_evaluated}",
        flush=True,
    )
    print(f"[monitor] CSV log → {csv_path}", flush=True)
    print(f"[monitor] Polling every {args.poll_interval}s ...", flush=True)

    while True:
        latest_step = latest_checkpoint_step(checkpoint_dir)

        if latest_step < 0:
            print("[monitor] No checkpoints found yet, waiting ...", flush=True)
        else:
            due = last_evaluated + args.eval_interval
            print(
                f"[monitor] latest_step={latest_step}  last_evaluated={last_evaluated}  "
                f"next_eval_at={due}",
                flush=True,
            )

            if latest_step >= due:
                # Find the highest step that is >= due (prefer the exact target or closest above)
                all_steps = sorted([
                    int(d.name.split("-")[-1])
                    for d in checkpoint_dir.iterdir()
                    if d.is_dir()
                    and d.name.startswith("checkpoint-")
                    and _safe_int(d.name.split("-")[-1]) >= due
                ], reverse=True)

                if all_steps:
                    # Evaluate the highest available step that crosses the threshold
                    target_step = all_steps[0]
                    ckpt_path   = checkpoint_path_for_step(checkpoint_dir, target_step)

                    print(
                        f"[monitor] Launching evaluation  checkpoint={ckpt_path.name} ...",
                        flush=True,
                    )

                    cmd = [
                        sys.executable,
                        str(EVAL_SCRIPT),
                        "--checkpoint_path", str(ckpt_path),
                        "--gpu",             args.gpu,
                        "--csv_path",        str(csv_path),
                        "--batch_size",      str(args.batch_size),
                    ]

                    t0 = time.time()
                    try:
                        result = subprocess.run(
                            cmd,
                            timeout=SUBPROCESS_TIMEOUT,
                            check=False,
                        )
                        elapsed = time.time() - t0
                        if result.returncode == 0:
                            last_evaluated = target_step
                            print(
                                f"[monitor] Evaluation DONE  step={target_step}  "
                                f"elapsed={elapsed:.0f}s",
                                flush=True,
                            )
                        else:
                            print(
                                f"[monitor] Evaluation FAILED  returncode={result.returncode}  "
                                f"step={target_step}  elapsed={elapsed:.0f}s  "
                                "Will retry next poll.",
                                flush=True,
                            )
                    except subprocess.TimeoutExpired:
                        print(
                            f"[monitor] Evaluation TIMEOUT after {SUBPROCESS_TIMEOUT}s  "
                            f"step={target_step}  Will retry next poll.",
                            flush=True,
                        )

        time.sleep(args.poll_interval)


def _safe_int(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return -1


if __name__ == "__main__":
    main()
