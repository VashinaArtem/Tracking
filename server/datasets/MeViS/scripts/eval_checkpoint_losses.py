#!/usr/bin/env python3
"""
eval_checkpoint_losses.py

Пересчитывает eval_loss на `mevis_val` для каждого сохранённого LoRA-чекпойнта,
используя тот же пайплайн, что и `llamafactory-cli train` (LLaMA-Factory + Trainer.evaluate).

Зачем:
  Во время обучения промежуточный eval_loss мог не попасть в `trainer_state.json`;
  этот скрипт заново грузит базу + адаптер и прогоняет валидацию.

Использование:
  cd /home/jovyan/shares/SR006.nfs2/vashina
  python datasets/MeViS/scripts/eval_checkpoint_losses.py \\
      --checkpoint_dir LLaMA-Factory/saves/minicpmv4_5/mevis/lora_v2 \\
      --gpu 0,1,2,3 \\
      --output_csv LLaMA-Factory/saves/minicpmv4_5/mevis/lora_v2/eval_loss_per_checkpoint.csv

  # Без пробелов после запятых; иначе shell разобьёт аргументы. Допустимо: --gpu "0, 1, 2, 3"

Опции:
  --step_from / --step_to   — фильтр по номеру шага в имени checkpoint-NNNN
  --max_samples N           — ускорить за счёт подвыборки mevis_val (метрика не сравнима с полным eval)
  --dry_run                 — только список чекпойнтов, без запуска LLaMA-Factory
  --resume                  — пропускать шаги, уже есть в CSV

Время: полный mevis_val (~30k сэмплов) на одном GPU — порядка десятков минут на чекпойнт
       (как при eval во время тренировки). На все ~58 чекпойнтов — очень долго; используйте фильтры.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
LLAMA_FACTORY_DIR = REPO_ROOT / "LLaMA-Factory"

CSV_COLUMNS = ["step", "checkpoint_path", "eval_loss", "eval_runtime", "timestamp"]


def discover_checkpoints(root: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    if not root.is_dir():
        return out
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if not m:
            continue
        adapter = p / "adapter_model.safetensors"
        if not adapter.exists():
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def load_done_steps(csv_path: Path) -> set[int]:
    done: set[int] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["step"]))
            except (KeyError, ValueError):
                pass
    return done


def build_eval_yaml(
    *,
    adapter_path: Path,
    run_output_dir: Path,
    max_samples: int | None,
) -> str:
    """YAML aligned with minicpmv4_5_mevis.yaml but eval-only."""
    max_samples_line = f"max_samples: {max_samples}\n" if max_samples is not None else ""
    return f"""### model
model_name_or_path: openbmb/MiniCPM-V-4_5
adapter_name_or_path: {adapter_path.as_posix()}
trust_remote_code: true
freeze_vision_tower: false

### method
stage: sft
do_train: false
do_eval: true
finetuning_type: lora
lora_target: q_proj,v_proj, k_proj

### dataset
eval_dataset: mevis_val
template: minicpm_v
cutoff_len: 4096
overwrite_cache: false
preprocessing_num_workers: 2
{max_samples_line}
### output
output_dir: {run_output_dir.as_posix()}
overwrite_output_dir: true
logging_steps: 50
plot_loss: false

### train (ignored when do_train=false; required by parser / Trainer)
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.0
bf16: true
fp16: false
ddp_timeout: 180000000

### eval
per_device_eval_batch_size: 1
"""


def run_one_eval(
    *,
    yaml_path: Path,
    gpu: str,
) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    cmd = ["llamafactory-cli", "train", str(yaml_path)]
    print(f"[eval_ckpt] CUDA_VISIBLE_DEVICES={gpu}  {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(LLAMA_FACTORY_DIR), env=env, check=True)


def read_eval_results(run_output_dir: Path) -> tuple[float, float]:
    path = run_output_dir / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    loss = float(data["eval_loss"])
    runtime = float(data.get("eval_runtime", 0.0))
    return loss, runtime


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval mevis_val loss per LoRA checkpoint via LLaMA-Factory")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(LLAMA_FACTORY_DIR / "saves/minicpmv4_5/mevis/lora_v2"),
        help="Directory containing checkpoint-NNNN subfolders",
    )
    p.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES for each eval run")
    p.add_argument(
        "--output_csv",
        type=str,
        default=str(LLAMA_FACTORY_DIR / "saves/minicpmv4_5/mevis/lora_v2/eval_loss_per_checkpoint.csv"),
        help="Append results here",
    )
    p.add_argument("--step_from", type=int, default=None, help="Minimum checkpoint step (inclusive)")
    p.add_argument("--step_to", type=int, default=None, help="Maximum checkpoint step (inclusive)")
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If set, truncate each loaded dataset to N rows (faster, not comparable to full eval)",
    )
    p.add_argument("--dry_run", action="store_true", help="List checkpoints and exit")
    p.add_argument("--resume", action="store_true", help="Skip steps already present in output_csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Quoted "--gpu \"0, 1, 2, 3\"" → normalize to 0,1,2,3. Unquoted spaces break argv before Python sees them.
    args.gpu = re.sub(r"\s+", "", str(args.gpu))
    ckpt_root = Path(args.checkpoint_dir).resolve()
    out_csv = Path(args.output_csv).resolve()

    if not LLAMA_FACTORY_DIR.is_dir():
        print(f"ERROR: LLaMA-Factory not found at {LLAMA_FACTORY_DIR}", file=sys.stderr)
        sys.exit(1)

    pairs = discover_checkpoints(ckpt_root)
    if args.step_from is not None:
        pairs = [(s, p) for s, p in pairs if s >= args.step_from]
    if args.step_to is not None:
        pairs = [(s, p) for s, p in pairs if s <= args.step_to]

    print(f"[eval_ckpt] Found {len(pairs)} checkpoints under {ckpt_root}", flush=True)
    for s, p in pairs[:5]:
        print(f"    step {s:6d}  {p.name}", flush=True)
    if len(pairs) > 5:
        print(f"    ... and {len(pairs) - 5} more", flush=True)

    if args.dry_run:
        return

    done = load_done_steps(out_csv) if args.resume else set()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_csv.exists() or out_csv.stat().st_size == 0

    with open(out_csv, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()

        for step, ckpt_path in pairs:
            if step in done:
                print(f"[eval_ckpt] skip step={step} (already in {out_csv.name})", flush=True)
                continue

            with tempfile.TemporaryDirectory(prefix=f"lf_eval_{step}_") as tmp:
                tmp_root = Path(tmp)
                yaml_path = tmp_root / "eval.yaml"
                run_out = tmp_root / "run_out"
                run_out.mkdir(parents=True, exist_ok=True)

                yaml_path.write_text(
                    build_eval_yaml(
                        adapter_path=ckpt_path,
                        run_output_dir=run_out,
                        max_samples=args.max_samples,
                    ),
                    encoding="utf-8",
                )

                t0 = time.time()
                try:
                    run_one_eval(yaml_path=yaml_path, gpu=args.gpu)
                    eval_loss, eval_rt = read_eval_results(run_out)
                except subprocess.CalledProcessError as e:
                    print(f"[eval_ckpt] FAILED step={step}  exit={e.returncode}", flush=True)
                    raise
                except Exception as e:
                    print(f"[eval_ckpt] FAILED step={step}  {e!r}", flush=True)
                    raise

                wall = time.time() - t0
                writer.writerow(
                    {
                        "step": step,
                        "checkpoint_path": str(ckpt_path),
                        "eval_loss": f"{eval_loss:.8f}",
                        "eval_runtime": f"{eval_rt:.3f}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                csv_file.flush()
                print(
                    f"[eval_ckpt] OK step={step}  eval_loss={eval_loss:.6f}  "
                    f"eval_runtime(json)={eval_rt:.1f}s  wall={wall:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
