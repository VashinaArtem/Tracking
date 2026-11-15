#!/usr/bin/env python3
"""
Строит один график: training loss (из trainer_state.json) + eval loss (из eval_loss_per_checkpoint.csv).

  cd /path/to/vashina
  python datasets/MeViS/scripts/plot_train_eval_loss.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_train_loss(state_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(state_path, encoding="utf-8") as f:
        st = json.load(f)
    steps, losses = [], []
    for h in st.get("log_history", []):
        if "loss" not in h or "step" not in h:
            continue
        v = h.get("loss")
        if not isinstance(v, (int, float)):
            continue
        steps.append(int(h["step"]))
        losses.append(float(v))
    return np.array(steps), np.array(losses)


def load_eval_loss(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, losses = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["step"]))
                losses.append(float(row["eval_loss"]))
            except (KeyError, ValueError):
                continue
    return np.array(steps), np.array(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    # scripts/ → MeViS/ → datasets/ → vashina/ (4 levels, same as eval_checkpoint_losses.py)
    script_dir = Path(__file__).resolve().parent
    repo = script_dir.parent.parent.parent
    root = repo / "LLaMA-Factory" / "saves" / "minicpmv4_5" / "mevis" / "lora_v2"
    parser.add_argument("--trainer_state", type=Path, default=root / "trainer_state.json")
    parser.add_argument("--eval_csv", type=Path, default=root / "eval_loss_per_checkpoint.csv")
    parser.add_argument("--output", type=Path, default=root / "train_eval_loss.png")
    args = parser.parse_args()

    train_x, train_y = load_train_loss(args.trainer_state.resolve())
    eval_x, eval_y = load_eval_loss(args.eval_csv.resolve())

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    ax.plot(train_x, train_y, color="#2563eb", alpha=0.7, linewidth=0.6, label="train loss (logged)")
    ax.plot(
        eval_x,
        eval_y,
        color="#dc2626",
        marker="o",
        markersize=2.5,
        linewidth=0.8,
        linestyle="-",
        label="eval loss (mevis_val, пересчёт по чекпойнтам)",
    )
    ax.set_xlabel("optimization step")
    ax.set_ylabel("loss")
    ax.set_title("MeViS LoRA v2 — training vs eval loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
