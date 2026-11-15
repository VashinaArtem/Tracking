#!/usr/bin/env python3
"""
eval_mevis_val_bbox_metrics_vllm.py

Для каждого LoRA-чекпойнта (`checkpoint-NNNN`) прогоняет vLLM на сэмплах из
`mevis_val.json` (тот же eval-пул, что и `eval_loss` в LLaMA-Factory) и пишет
в CSV сводные IoU / GIoU / acc@0.5 / acc@0.75 и долю успешного парса бокса.

Промпт собирается как в ``run_bbox_eval_vllm.py`` (``format_grounding_question`` +
``build_prompt``, ``skip_special_tokens=False``), а не из сырого ``messages[0]``
в JSON — иначе распределение входов не совпадёт с FMD/valid_u bbox-eval.

Ограничения и риски (читать перед интерпретацией чисел):
  - Метрики на ``mevis_val`` удобны для **ранжирования чекпойнтов** рядом с
    eval-кривой, но это **не** замена финальной оценки на **test FMD**
    (отложенные ~90 видео).
  - При ``--max_samples`` значения **несравнимы** с полным прогоном ~30k строк.
  - Это **не** то же самое, что ``eval_loss`` из Trainer (другая величина).

**Полные метрики (``--full_metrics``)** — агрегация как в ``metrics_finetune.process_csv``:
``per_expression.csv`` (в т.ч. ``iou_tg``), ``per_video.csv``, ``global_summary.txt``.
``per_frame.csv`` совпадает по смыслу с сырым bbox-eval CSV (как
``datasets/MeViS/results_finetune/*_raw.csv``): ``frame_idx`` (= ``n_frame`` внутри кода),
``raw_response``, ``clean_response``, ``pred_bbox_raw``, ``inference_time`` (доля времени
батча ``generate`` на строку), плюс пиксельные предсказания и ``iou``/``giou``/``gt_is_valid``.
Компактный ``summary.csv`` по-прежнему пишется для ранжирования чекпойнтов.

  - **Синтетический ``expr_id``**: в ``mevis_val`` нет id выражения из
    ``meta_expressions_v2.json``. Используется стабильный 16-символьный префикс
    ``sha256(f"{video_name}\\0{ref}")`` по паре (видео, текст ``<ref>``), чтобы
    одна и та же фраза в разных видео не коллаидировала в ``groupby``.
  - **``iou_tg``**: считается по **подмножеству кадров**, присутствующим в
    ``mevis_val``, отсортированным по ``n_frame`` внутри пары
    ``(video_name, expr_id)``. Формула та же, что в legacy FMD, но числа не
    обязаны совпадать с прогоном по полному видео.
  - **Объём диска**: полный ``mevis_val`` даёт порядка **~30k** строк в
    ``per_frame.csv`` на чекпойнт/бейзлайн.
  - **Resume с ``--full_metrics``**: шаг считается завершённым, если в
    ``run_dir`` есть ``global_summary.txt``. Старые прогоны только с
    ``summary.csv`` при первом запуске с ``--full_metrics`` будут **пересчитаны**
    (нет маркера пост-агрегации).

Результаты пишутся **отдельно от папок весов** ``checkpoint-*`` LoRA.

По умолчанию корень артефактов: ``{checkpoint_dir}/vllm_bbox_eval_mevis_val/``.
Для **каждого** чекпойнта создаётся своя подпапка ``checkpoint-{step}/`` с файлами:

  - ``summary.csv`` — одна строка со сводными метриками (step, mIoU, …);
  - при ``--full_metrics`` (или устаревшем ``--save_per_frame``): см. выше;
  - ``baseline/summary.csv`` — при ``--eval_baseline`` (та же выборка и промпт,
    **без LoRA**, только ``--model_path``).

Корень можно переопределить через ``--output_dir``.

Использование (из корня репозитория ``vashina``)::

  python datasets/MeViS/scripts/eval_mevis_val_bbox_metrics_vllm.py \\
      --checkpoint_dir LLaMA-Factory/saves/minicpmv4_5/mevis/lora_v2 \\
      --gpu 0

  # На общей GPU: при необходимости снизить долю VRAM под vLLM (по умолчанию 0.75)
  python datasets/MeViS/scripts/eval_mevis_val_bbox_metrics_vllm.py ... \\
      --gpu_memory_utilization 0.7

  # Надёжнее по VRAM на многих чекпойнтах: отдельный процесс на чекпойнт
  python datasets/MeViS/scripts/eval_mevis_val_bbox_metrics_vllm.py ... \\
      --subprocess_per_checkpoint

  # С бейзлайном (без LoRA) для сравнения с чекпойнтами
  python datasets/MeViS/scripts/eval_mevis_val_bbox_metrics_vllm.py ... \\
      --eval_baseline

  # Полные CSV + global_summary (как metrics_finetune)
  python datasets/MeViS/scripts/eval_mevis_val_bbox_metrics_vllm.py ... \\
      --full_metrics

  # Быстрая подвыборка (метрики не сравнимы с полным eval)
  python datasets/MeViS/scripts/eval_mevis_val_bbox_metrics_vllm.py ... \\
      --max_samples 2000 --seed 42

**Прогресс и ETA (оценочно):** строки с префиксом ``[mevis_val_bbox][progress]`` —
батчи внутри одного чекпоинта/бейзлайна и счётчик чекпоинтов в ``main``.
``--progress_every_batches 0`` — одна **полоска прогресса по батчам** (``tqdm``, иначе ASCII
с ``\\r``); ``N>=1`` — текстовые строки каждые ``N`` батчей. ``--vllm_tqdm`` — отдельные бары vLLM
на каждый ``generate`` (по умолчанию выкл.; обычно не нужны). Первые батчи после загрузки модели
часто медленнее — ETA до прогрева грубый.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
import gc
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

# Same directory imports when run as ``python datasets/MeViS/scripts/...``
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
LLAMA_FACTORY_DIR = REPO_ROOT / "LLaMA-Factory"
DEFAULT_MEVIS_VAL = LLAMA_FACTORY_DIR / "data" / "mevis_val.json"
MODEL_ID = "openbmb/MiniCPM-V-4_5"
METRICS_SUBDIR_NAME = "vllm_bbox_eval_mevis_val"
SUMMARY_CSV_NAME = "summary.csv"
GLOBAL_SUMMARY_NAME = "global_summary.txt"
BASELINE_DIR_NAME = "baseline"
# vLLM targets ``fraction * total_vram``; must stay below free VRAM at ``LLM()`` init (shared GPUs).
DEFAULT_GPU_MEMORY_UTILIZATION = 0.75
PROGRESS_PREFIX = "[mevis_val_bbox][progress]"
BATCH_TIME_WINDOW = 8
_VLLM_USE_TQDM_UNSUPPORTED = False


def format_eta(seconds: float) -> str:
    """Human-readable ETA; handles non-finite values."""
    if seconds != seconds or seconds < 0 or seconds == float("inf"):  # nan or invalid
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s}s"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m}m{s}s"


class _TerminalAsciiBatchBar:
    """Single-line batch progress when ``tqdm`` is not installed."""

    def __init__(self, label: str, total: int, width: int = 28) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.done = 0
        self.width = width

    def update(self, n: int = 1) -> None:
        self.done = min(self.done + n, self.total)
        filled = int(self.width * self.done / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        pct = 100.0 * self.done / self.total
        msg = f"{PROGRESS_PREFIX} {self.label} |{bar}| {self.done}/{self.total} ({pct:.1f}%)"
        print(f"\r{msg}", end="", flush=True)

    def close(self) -> None:
        if self.done > 0:
            print("", flush=True)


def llm_generate_with_tqdm_option(
    llm: Any,
    prompts: list[Any],
    gen_kw: dict[str, Any],
    *,
    use_vllm_tqdm: bool,
) -> Any:
    """Call ``llm.generate``; disable vLLM tqdm unless ``use_vllm_tqdm`` (with TypeError fallback)."""
    global _VLLM_USE_TQDM_UNSUPPORTED
    if use_vllm_tqdm:
        return llm.generate(prompts, **gen_kw)
    if _VLLM_USE_TQDM_UNSUPPORTED:
        return llm.generate(prompts, **gen_kw)
    try:
        return llm.generate(prompts, use_tqdm=False, **gen_kw)
    except TypeError:
        _VLLM_USE_TQDM_UNSUPPORTED = True
        print(
            f"{PROGRESS_PREFIX} vLLM generate() has no use_tqdm=; vLLM progress bars stay enabled.",
            flush=True,
        )
        return llm.generate(prompts, **gen_kw)


from bbox_eval_checkpoint import compute_giou  # noqa: E402
from run_bbox_eval_vllm import (  # noqa: E402
    BBoxParser,
    build_prompt,
    denormalize,
    format_grounding_question,
)

CSV_COLUMNS = [
    "step",
    "checkpoint_path",
    "n_eval",
    "mIoU",
    "mean_GIoU",
    "acc_50",
    "acc_75",
    "parse_success_rate",
    "wall_seconds",
    "timestamp",
]


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


def checkpoint_run_dir(output_dir: Path, step: int) -> Path:
    return output_dir / f"checkpoint-{step}"


def synthetic_expr_id(video_name: str, ref: str) -> str:
    """Stable synthetic id for (video, expression); not MeViS meta ``expr_id``."""
    h = hashlib.sha256(f"{video_name}\0{ref}".encode("utf-8")).hexdigest()
    return h[:16]


def frame_index_from_image_path(img_path: Path, warn_counter: list[int]) -> int:
    """Frame index from filename stem (e.g. ``00042.jpg`` -> 42). Fallback logs via ``warn_counter``."""
    stem = img_path.stem
    if stem.isdigit():
        return int(stem)
    nums = re.findall(r"\d+", stem)
    if nums:
        return int(nums[-1])
    warn_counter[0] += 1
    return 0


def load_done_steps_from_output_dir(output_dir: Path, *, full_metrics: bool) -> set[int]:
    """Resume: done if ``summary.csv`` (default) or ``global_summary.txt`` (``--full_metrics``)."""
    done: set[int] = set()
    if not output_dir.is_dir():
        return done
    marker = GLOBAL_SUMMARY_NAME if full_metrics else SUMMARY_CSV_NAME
    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and (p / marker).is_file():
            done.add(int(m.group(1)))
    return done


def baseline_artifacts_complete(baseline_dir: Path, *, full_metrics: bool) -> bool:
    marker = GLOBAL_SUMMARY_NAME if full_metrics else SUMMARY_CSV_NAME
    return (baseline_dir / marker).is_file()


def extract_ref_expr(text: str) -> Optional[str]:
    m = re.search(r"<ref>(.*?)</ref>", text, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip()


def parse_gt_norm_xyxy(assistant_content: str) -> Optional[list[int]]:
    return BBoxParser._from_box_tags(assistant_content)  # type: ignore[attr-defined]


def strip_thinking(text: str) -> str:
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def primary_cuda_device(gpu_arg: str) -> str:
    """vLLM uses a single visible device; take the first index from a list."""
    parts = [p.strip() for p in str(gpu_arg).split(",") if p.strip()]
    return parts[0] if parts else "0"


def load_mevis_records(
    json_path: Path,
    max_samples: Optional[int],
    seed: int,
) -> list[dict[str, Any]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {json_path}")
    if max_samples is not None and max_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, max_samples)
    return data


def evaluate_checkpoint_or_baseline(
    *,
    adapter_path: Optional[Path],
    records: list[dict[str, Any]],
    batch_size: int,
    model_path: str,
    full_metrics: bool,
    step: int,
    checkpoint_path_for_csv: str,
    progress_label: str = "run",
    progress_every_batches: int = 5,
    use_vllm_tqdm: bool = False,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> dict[str, Any]:
    """If ``adapter_path`` is None, evaluate base model without LoRA. CUDA must be set before import."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    frame_warns = [0]

    use_lora = adapter_path is not None
    t_wall0 = time.time()
    lora_request = None
    if use_lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("mevis_lora", 1, str(adapter_path.resolve()))

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        enable_lora=use_lora,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    _stop_names = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = list(
        {
            tid
            for name in _stop_names
            for tid in [tokenizer.convert_tokens_to_ids(name)]
            if isinstance(tid, int) and tid != tokenizer.unk_token_id
        }
        | {tokenizer.eos_token_id}
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
    )

    ious: list[float] = []
    gious: list[float] = []
    parse_ok: list[bool] = []
    per_frame_rows: list[dict[str, Any]] = []

    n_batches = (len(records) + batch_size - 1) // batch_size
    recent_gen_seconds: deque[float] = deque(maxlen=BATCH_TIME_WINDOW)
    t_phase0 = time.perf_counter()

    batch_pbar: Any = None
    ascii_batch_bar: Optional[_TerminalAsciiBatchBar] = None
    if progress_every_batches == 0 and n_batches > 0:
        try:
            from tqdm import tqdm

            batch_pbar = tqdm(
                total=n_batches,
                desc=f"{PROGRESS_PREFIX} {progress_label}",
                unit="batch",
                leave=True,
                mininterval=0.2,
                file=sys.stdout,
                smoothing=0.05,
            )
        except ImportError:
            ascii_batch_bar = _TerminalAsciiBatchBar(progress_label, n_batches)

    if progress_every_batches == 0 and n_batches > 0:
        print(
            f"{PROGRESS_PREFIX} {progress_label}: {len(records)} records, batch_size={batch_size}, "
            f"{n_batches} batch(es) — batch progress bar below",
            flush=True,
        )
    else:
        print(
            f"{PROGRESS_PREFIX} {progress_label}: {len(records)} records, "
            f"{n_batches} batch(es), batch_size={batch_size} (ETA rough until GPU warm)",
            flush=True,
        )

    for b in range(n_batches):
        batch = records[b * batch_size : (b + 1) * batch_size]
        vllm_inputs: list[Optional[dict]] = []
        meta: list[Optional[dict]] = []

        for local_i, rec in enumerate(batch):
            try:
                img_path = Path(rec["images"][0])
            except (KeyError, TypeError, IndexError):
                vllm_inputs.append(None)
                meta.append(None)
                continue

            msgs = rec.get("messages") or []
            if len(msgs) < 2:
                vllm_inputs.append(None)
                meta.append(None)
                continue

            user_c = msgs[0].get("content") or ""
            asst_c = msgs[1].get("content") or ""
            expr = extract_ref_expr(user_c)
            if not expr:
                vllm_inputs.append(None)
                meta.append(None)
                continue

            gt_norm = parse_gt_norm_xyxy(asst_c)
            if gt_norm is None:
                vllm_inputs.append(None)
                meta.append(None)
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                vllm_inputs.append(None)
                meta.append(None)
                continue

            h, w = img.size[1], img.size[0]
            gt_px = denormalize(gt_norm, h, w)
            if gt_px == [0, 0, 0, 0]:
                vllm_inputs.append(None)
                meta.append(None)
                continue

            video_name = img_path.parent.name
            expr_id = synthetic_expr_id(video_name, expr)
            n_frame = frame_index_from_image_path(img_path, frame_warns)

            question = format_grounding_question(expr)
            prompt_str = build_prompt(tokenizer, question)
            vllm_inputs.append({"prompt": prompt_str, "multi_modal_data": {"image": img}})
            meta.append(
                {
                    "img_path": str(img_path),
                    "h": h,
                    "w": w,
                    "gt_px": gt_px,
                    "sample_idx": b * batch_size + local_i,
                    "video_name": video_name,
                    "expr_id": expr_id,
                    "prompt": expr,
                    "n_frame": int(n_frame),
                }
            )

        valid_idx = [i for i, x in enumerate(vllm_inputs) if x is not None]
        valid_inputs = [vllm_inputs[i] for i in valid_idx]

        batch_infer_seconds = 0.0
        if valid_inputs:
            gen_kw: dict[str, Any] = {"sampling_params": sampling_params}
            if lora_request is not None:
                gen_kw["lora_request"] = lora_request
            t_gen0 = time.perf_counter()
            raw_outputs = llm_generate_with_tqdm_option(
                llm,
                valid_inputs,
                gen_kw,
                use_vllm_tqdm=use_vllm_tqdm,
            )
            batch_infer_seconds = time.perf_counter() - t_gen0
            recent_gen_seconds.append(batch_infer_seconds)
        else:
            raw_outputs = []

        infer_share = (
            batch_infer_seconds / len(valid_inputs) if valid_inputs else 0.0
        )

        out_map = {valid_idx[j]: raw_outputs[j] for j in range(len(valid_idx))}

        for i, rec in enumerate(batch):
            m = meta[i]
            inp = vllm_inputs[i]
            if m is None or inp is None:
                continue

            out = out_map.get(i)
            raw_response = out.outputs[0].text if out is not None else ""
            clean = strip_thinking(raw_response)
            pred_raw = BBoxParser.extract_bbox(clean)
            ok = pred_raw is not None

            if ok and pred_raw is not None:
                pred_px = denormalize(pred_raw, m["h"], m["w"])
                pred_bbox_raw = json.dumps(list(pred_raw))
            else:
                pred_px = [0, 0, 0, 0]
                pred_bbox_raw = "[]"

            iou, giou = compute_giou(m["gt_px"], pred_px)
            ious.append(float(iou))
            gious.append(float(giou))
            parse_ok.append(ok)

            if full_metrics:
                per_frame_rows.append(
                    {
                        "video_name": m["video_name"],
                        "expr_id": m["expr_id"],
                        "prompt": m["prompt"],
                        "n_frame": m["n_frame"],
                        "raw_response": raw_response,
                        "clean_response": clean,
                        "success": ok,
                        "pred_bbox_raw": pred_bbox_raw,
                        "height": m["h"],
                        "width": m["w"],
                        "x_min_gt": m["gt_px"][0],
                        "x_max_gt": m["gt_px"][1],
                        "y_min_gt": m["gt_px"][2],
                        "y_max_gt": m["gt_px"][3],
                        "inference_time": float(infer_share),
                        "x_min_pred_px": pred_px[0],
                        "x_max_pred_px": pred_px[1],
                        "y_min_pred_px": pred_px[2],
                        "y_max_pred_px": pred_px[3],
                    }
                )

        done_b = b + 1
        rem_b = n_batches - done_b
        if progress_every_batches > 0 and (
            done_b % progress_every_batches == 0 or done_b == n_batches
        ):
            elapsed = time.perf_counter() - t_phase0
            if recent_gen_seconds and rem_b > 0:
                avg_g = sum(recent_gen_seconds) / len(recent_gen_seconds)
                eta_b = avg_g * rem_b
                eta_s = format_eta(eta_b)
            elif rem_b == 0:
                eta_s = "0s"
            else:
                eta_s = "?"
            print(
                f"{PROGRESS_PREFIX} {progress_label}: batch {done_b}/{n_batches} "
                f"(valid_in_batch={len(valid_inputs)}) elapsed={format_eta(elapsed)} "
                f"ETA_batches≈{eta_s}",
                flush=True,
            )

        if batch_pbar is not None:
            batch_pbar.update(1)
        elif ascii_batch_bar is not None:
            ascii_batch_bar.update(1)

    if batch_pbar is not None:
        batch_pbar.close()
    if ascii_batch_bar is not None:
        ascii_batch_bar.close()

    del llm
    gc.collect()

    if frame_warns[0] > 0:
        print(
            f"[mevis_val_bbox] WARNING: {frame_warns[0]} image(s) used n_frame=0 fallback "
            f"(non-numeric filename stem); check JPEG naming.",
            flush=True,
        )

    n = len(ious)
    wall = time.time() - t_wall0
    if n == 0:
        return {
            "step": step,
            "checkpoint_path": checkpoint_path_for_csv,
            "n_eval": 0,
            "mIoU": 0.0,
            "mean_GIoU": 0.0,
            "acc_50": 0.0,
            "acc_75": 0.0,
            "parse_success_rate": 0.0,
            "wall_seconds": round(wall, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "_per_frame_rows": per_frame_rows,
        }

    iou_arr = np.array(ious, dtype=np.float64)
    acc_50 = float(np.mean(iou_arr >= 0.5))
    acc_75 = float(np.mean(iou_arr >= 0.75))
    return {
        "step": step,
        "checkpoint_path": checkpoint_path_for_csv,
        "n_eval": n,
        "mIoU": float(np.mean(iou_arr)),
        "mean_GIoU": float(np.mean(gious)),
        "acc_50": acc_50,
        "acc_75": acc_75,
        "parse_success_rate": float(np.mean(parse_ok)) if parse_ok else 0.0,
        "wall_seconds": round(wall, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "_per_frame_rows": per_frame_rows,
    }


def write_summary_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_row = {k: row[k] for k in CSV_COLUMNS}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerow(out_row)


def write_full_metrics_artifacts(run_dir: Path, rows: list[dict[str, Any]], run_name: str) -> None:
    """``per_frame`` / ``per_expression`` / ``per_video`` / ``global_summary`` (``metrics_finetune``-style)."""
    import pandas as pd

    from mevis_bbox_metrics_aggregate import aggregate_bbox_frames_to_run_dir

    aggregate_bbox_frames_to_run_dir(run_dir, pd.DataFrame(rows), run_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="mevis_val bbox IoU/GIoU per LoRA checkpoint via vLLM",
    )
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing checkpoint-NNNN with adapter_model.safetensors",
    )
    p.add_argument(
        "--mevis_val",
        type=str,
        default=str(DEFAULT_MEVIS_VAL),
        help="Path to mevis_val.json",
    )
    p.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES (first index used for vLLM), e.g. 0 or 0,1,2,3",
    )
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Random subset size (not comparable to full ~30k eval)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --max_samples")
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Root directory for per-checkpoint folders. If omitted: "
            "{checkpoint_dir}/vllm_bbox_eval_mevis_val/"
        ),
    )
    p.add_argument(
        "--full_metrics",
        action="store_true",
        help=(
            "Write metrics_finetune-style artifacts: per_frame/per_expression/per_video CSV "
            f"and {GLOBAL_SUMMARY_NAME} under each run_dir (~30k rows per checkpoint on full mevis_val)"
        ),
    )
    p.add_argument(
        "--save_per_frame",
        action="store_true",
        help="Deprecated alias for --full_metrics (emits a warning)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            f"Skip steps where output_dir/checkpoint-{{step}}/{SUMMARY_CSV_NAME} exists "
            f"(or {GLOBAL_SUMMARY_NAME} when --full_metrics); with --eval_baseline skip baseline "
            f"when output_dir/{BASELINE_DIR_NAME}/ has the same marker"
        ),
    )
    p.add_argument("--step_from", type=int, default=None)
    p.add_argument("--step_to", type=int, default=None)
    p.add_argument(
        "--model_path",
        type=str,
        default=MODEL_ID,
        help="Base model id or local path",
    )
    p.add_argument(
        "--eval_baseline",
        action="store_true",
        help=(
            f"Also run base model (--model_path) without LoRA; write "
            f"output_dir/{BASELINE_DIR_NAME}/{SUMMARY_CSV_NAME} (runs first, same mevis_val subset)"
        ),
    )
    p.add_argument(
        "--subprocess_per_checkpoint",
        action="store_true",
        help="Spawn a fresh Python process per checkpoint (safer GPU memory across many checkpoints)",
    )
    p.add_argument(
        "--vllm_tqdm",
        action="store_true",
        help="Show vLLM tqdm bars on each generate() batch (default: off for quieter logs)",
    )
    p.add_argument(
        "--progress_every_batches",
        type=int,
        default=5,
        metavar="N",
        help=(
            f"Batch progress: N>=1 emits {PROGRESS_PREFIX} text lines every N batches (default: 5); "
            "N=0 uses one tqdm bar over batches (falls back to ASCII \\r if tqdm missing); "
            "checkpoint-level lines in main are unchanged"
        ),
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        metavar="F",
        help=(
            "vLLM ``gpu_memory_utilization`` (fraction of **total** VRAM to target); "
            f"default {DEFAULT_GPU_MEMORY_UTILIZATION} leaves headroom on shared GPUs. "
            "Raise only if plenty of free memory at engine init."
        ),
    )
    p.add_argument(
        "--one_checkpoint_internal",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--baseline_internal",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ns = p.parse_args()
    if ns.save_per_frame:
        print(
            "[mevis_val_bbox] WARNING: --save_per_frame is deprecated; use --full_metrics.",
            flush=True,
        )
    ns.full_metrics = bool(ns.full_metrics or ns.save_per_frame)
    if ns.progress_every_batches < 0:
        p.error("--progress_every_batches must be >= 0")
    gmu = float(ns.gpu_memory_utilization)
    if not (0.0 < gmu <= 0.99):
        p.error("--gpu_memory_utilization must be in (0, 0.99]")
    ns.gpu_memory_utilization = gmu
    return ns


def build_baseline_worker_argv(base: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint_dir",
        str(Path(base.checkpoint_dir).resolve()),
        "--mevis_val",
        str(base.mevis_val),
        "--gpu",
        base.gpu,
        "--batch_size",
        str(base.batch_size),
        "--seed",
        str(base.seed),
        "--output_dir",
        str(base.output_dir),
        "--model_path",
        str(base.model_path),
        "--baseline_internal",
        *(
            ["--max_samples", str(base.max_samples)]
            if base.max_samples is not None
            else []
        ),
        *([] if not base.full_metrics else ["--full_metrics"]),
        "--progress_every_batches",
        str(base.progress_every_batches),
        *([] if not base.vllm_tqdm else ["--vllm_tqdm"]),
        "--gpu_memory_utilization",
        str(base.gpu_memory_utilization),
    ]


def build_worker_argv(base: argparse.Namespace, ckpt_path: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--checkpoint_dir",
        str(ckpt_path.parent),
        "--mevis_val",
        str(base.mevis_val),
        "--gpu",
        base.gpu,
        "--batch_size",
        str(base.batch_size),
        "--seed",
        str(base.seed),
        "--output_dir",
        str(base.output_dir),
        "--model_path",
        str(base.model_path),
        "--one_checkpoint_internal",
        str(ckpt_path),
        *(
            ["--max_samples", str(base.max_samples)]
            if base.max_samples is not None
            else []
        ),
        *([] if not base.full_metrics else ["--full_metrics"]),
        "--progress_every_batches",
        str(base.progress_every_batches),
        *([] if not base.vllm_tqdm else ["--vllm_tqdm"]),
        "--gpu_memory_utilization",
        str(base.gpu_memory_utilization),
    ]


def _run_baseline_eval(
    *,
    args: argparse.Namespace,
    mevis_path: Path,
    output_dir: Path,
    records: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Write ``baseline/summary.csv`` and optional full-metrics artifacts."""
    baseline_dir = output_dir / BASELINE_DIR_NAME
    marker_path = baseline_dir / (
        GLOBAL_SUMMARY_NAME if args.full_metrics else SUMMARY_CSV_NAME
    )
    if args.resume and baseline_artifacts_complete(baseline_dir, full_metrics=args.full_metrics):
        print(f"[mevis_val_bbox] skip baseline (exists {marker_path})", flush=True)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = primary_cuda_device(args.gpu)
    if records is None:
        records = load_mevis_records(mevis_path, args.max_samples, args.seed)
    row = evaluate_checkpoint_or_baseline(
        adapter_path=None,
        records=records,
        batch_size=args.batch_size,
        model_path=args.model_path,
        full_metrics=args.full_metrics,
        step=-1,
        checkpoint_path_for_csv=args.model_path,
        progress_label="baseline",
        progress_every_batches=args.progress_every_batches,
        use_vllm_tqdm=args.vllm_tqdm,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    pf = row.pop("_per_frame_rows", [])
    summary_path = baseline_dir / SUMMARY_CSV_NAME
    write_summary_csv(summary_path, row)
    if args.full_metrics:
        write_full_metrics_artifacts(baseline_dir, pf, "mevis_val_baseline")
    print(
        f"[mevis_val_bbox] baseline n_eval={row['n_eval']} mIoU={row['mIoU']:.4f} "
        f"mean_GIoU={row['mean_GIoU']:.4f} parse_ok={row['parse_success_rate']:.4f} "
        f"wall={row['wall_seconds']}s",
        flush=True,
    )
    print(f"[mevis_val_bbox] Wrote {summary_path}", flush=True)
    if args.full_metrics:
        print(f"[mevis_val_bbox] Wrote full metrics under {baseline_dir}", flush=True)
    gc.collect()


def main() -> None:
    args = parse_args()
    args.gpu = re.sub(r"\s+", "", str(args.gpu))
    mevis_path = Path(args.mevis_val).resolve()
    ckpt_root = Path(args.checkpoint_dir).resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (ckpt_root / METRICS_SUBDIR_NAME).resolve()
    args.output_dir = str(output_dir)
    if output_dir.resolve() == ckpt_root.resolve():
        print(
            "ERROR: --output_dir must not equal --checkpoint_dir "
            "(subfolders checkpoint-NNNN would overwrite LoRA weights).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.baseline_internal:
        os.environ["CUDA_VISIBLE_DEVICES"] = primary_cuda_device(args.gpu)
        records = load_mevis_records(mevis_path, args.max_samples, args.seed)
        baseline_dir = output_dir / BASELINE_DIR_NAME
        row = evaluate_checkpoint_or_baseline(
            adapter_path=None,
            records=records,
            batch_size=args.batch_size,
            model_path=args.model_path,
            full_metrics=args.full_metrics,
            step=-1,
            checkpoint_path_for_csv=args.model_path,
            progress_label="baseline",
            progress_every_batches=args.progress_every_batches,
            use_vllm_tqdm=args.vllm_tqdm,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        pf = row.pop("_per_frame_rows", [])
        write_summary_csv(baseline_dir / SUMMARY_CSV_NAME, row)
        if args.full_metrics:
            write_full_metrics_artifacts(baseline_dir, pf, "mevis_val_baseline")
        print(
            f"[mevis_val_bbox] baseline (subprocess) n_eval={row['n_eval']} mIoU={row['mIoU']:.4f} "
            f"mean_GIoU={row['mean_GIoU']:.4f} wall={row['wall_seconds']}s",
            flush=True,
        )
        print(f"[mevis_val_bbox] Wrote {baseline_dir / SUMMARY_CSV_NAME}", flush=True)
        if args.full_metrics:
            print(f"[mevis_val_bbox] Wrote full metrics under {baseline_dir}", flush=True)
        return

    if args.one_checkpoint_internal:
        ckpt_path = Path(args.one_checkpoint_internal).resolve()
        m = re.fullmatch(r"checkpoint-(\d+)", ckpt_path.name)
        step = int(m.group(1)) if m else -1

        os.environ["CUDA_VISIBLE_DEVICES"] = primary_cuda_device(args.gpu)

        run_dir = checkpoint_run_dir(output_dir, step)
        records = load_mevis_records(mevis_path, args.max_samples, args.seed)
        row = evaluate_checkpoint_or_baseline(
            adapter_path=ckpt_path,
            records=records,
            batch_size=args.batch_size,
            model_path=args.model_path,
            full_metrics=args.full_metrics,
            step=step,
            checkpoint_path_for_csv=str(ckpt_path.resolve()),
            progress_label=f"checkpoint-{step}",
            progress_every_batches=args.progress_every_batches,
            use_vllm_tqdm=args.vllm_tqdm,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        pf = row.pop("_per_frame_rows", [])
        write_summary_csv(run_dir / SUMMARY_CSV_NAME, row)
        if args.full_metrics:
            write_full_metrics_artifacts(run_dir, pf, f"mevis_val_checkpoint_{step}")
        print(
            f"[mevis_val_bbox] step={step} n_eval={row['n_eval']} "
            f"mIoU={row['mIoU']:.4f} mean_GIoU={row['mean_GIoU']:.4f} "
            f"parse_ok={row['parse_success_rate']:.4f} wall={row['wall_seconds']}s",
            flush=True,
        )
        print(f"[mevis_val_bbox] Wrote {run_dir / SUMMARY_CSV_NAME}", flush=True)
        if args.full_metrics:
            print(f"[mevis_val_bbox] Wrote full metrics under {run_dir}", flush=True)
        return

    pairs = discover_checkpoints(ckpt_root)
    if args.step_from is not None:
        pairs = [(s, p) for s, p in pairs if s >= args.step_from]
    if args.step_to is not None:
        pairs = [(s, p) for s, p in pairs if s <= args.step_to]

    print(f"[mevis_val_bbox] Found {len(pairs)} checkpoints under {ckpt_root}", flush=True)
    print(f"[mevis_val_bbox] Output root (per-checkpoint subdirs): {output_dir}", flush=True)
    done = (
        load_done_steps_from_output_dir(output_dir, full_metrics=args.full_metrics)
        if args.resume
        else set()
    )

    to_run = [(s, p) for s, p in pairs if s not in done]
    print(
        f"{PROGRESS_PREFIX} checkpoints: {len(to_run)} to run this session, "
        f"{len(pairs) - len(to_run)} skipped (resume/filters) of {len(pairs)} listed",
        flush=True,
    )

    records: Optional[list[dict[str, Any]]] = None
    if not args.subprocess_per_checkpoint:
        records = load_mevis_records(mevis_path, args.max_samples, args.seed)
        print(f"[mevis_val_bbox] Loaded {len(records)} mevis_val rows from {mevis_path}", flush=True)

    if args.eval_baseline:
        baseline_dir = output_dir / BASELINE_DIR_NAME
        skip_bl = args.resume and baseline_artifacts_complete(
            baseline_dir, full_metrics=args.full_metrics
        )
        if skip_bl:
            marker = baseline_dir / (
                GLOBAL_SUMMARY_NAME if args.full_metrics else SUMMARY_CSV_NAME
            )
            print(f"[mevis_val_bbox] skip baseline (exists {marker})", flush=True)
        elif args.subprocess_per_checkpoint:
            cmd = build_baseline_worker_argv(args)
            print(
                f"{PROGRESS_PREFIX} baseline subprocess starting "
                f"(then {len(to_run)} checkpoint worker(s) if any)",
                flush=True,
            )
            print(f"[mevis_val_bbox] subprocess (baseline): {' '.join(cmd)}", flush=True)
            t_bl0 = time.perf_counter()
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
            print(
                f"{PROGRESS_PREFIX} baseline subprocess done wall="
                f"{format_eta(time.perf_counter() - t_bl0)}",
                flush=True,
            )
        else:
            assert records is not None
            _run_baseline_eval(
                args=args,
                mevis_path=mevis_path,
                output_dir=output_dir,
                records=records,
            )

    ck_run_wall: list[float] = []
    n_to_run = len(to_run)

    for step, ckpt_path in pairs:
        if step in done:
            marker = GLOBAL_SUMMARY_NAME if args.full_metrics else SUMMARY_CSV_NAME
            print(
                f"[mevis_val_bbox] skip step={step} (exists {checkpoint_run_dir(output_dir, step) / marker})",
                flush=True,
            )
            continue

        if args.subprocess_per_checkpoint:
            cmd = build_worker_argv(args, ckpt_path)
            idx = len(ck_run_wall) + 1
            if ck_run_wall and n_to_run > 0:
                avg_c = sum(ck_run_wall) / len(ck_run_wall)
                rem_inc = n_to_run - idx + 1
                eta_before = format_eta(avg_c * rem_inc)
            else:
                eta_before = "?"
            print(
                f"{PROGRESS_PREFIX} checkpoint subprocess {idx}/{n_to_run} step={step} "
                f"ETA_this+remaining≈{eta_before}",
                flush=True,
            )
            print(f"[mevis_val_bbox] subprocess: {' '.join(cmd)}", flush=True)
            t_ck0 = time.perf_counter()
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
            dt_ck = time.perf_counter() - t_ck0
            ck_run_wall.append(dt_ck)
            rem_after = n_to_run - len(ck_run_wall)
            if ck_run_wall and rem_after > 0:
                avg_c = sum(ck_run_wall) / len(ck_run_wall)
                eta_rest = format_eta(avg_c * rem_after)
            else:
                eta_rest = "0s" if rem_after == 0 else "?"
            print(
                f"{PROGRESS_PREFIX} checkpoint step={step} subprocess wall={format_eta(dt_ck)} "
                f"ETA_remaining≈{eta_rest}",
                flush=True,
            )
        else:
            assert records is not None
            os.environ["CUDA_VISIBLE_DEVICES"] = primary_cuda_device(args.gpu)
            run_dir = checkpoint_run_dir(output_dir, step)
            idx = len(ck_run_wall) + 1
            if ck_run_wall and n_to_run > 0:
                avg_c = sum(ck_run_wall) / len(ck_run_wall)
                rem_inc = n_to_run - idx + 1
                eta_before = format_eta(avg_c * rem_inc)
            else:
                eta_before = "?"
            print(
                f"{PROGRESS_PREFIX} in-process eval {idx}/{n_to_run} step={step} "
                f"ETA_this+remaining≈{eta_before}",
                flush=True,
            )
            t_ck0 = time.perf_counter()
            row = evaluate_checkpoint_or_baseline(
                adapter_path=ckpt_path,
                records=records,
                batch_size=args.batch_size,
                model_path=args.model_path,
                full_metrics=args.full_metrics,
                step=step,
                checkpoint_path_for_csv=str(ckpt_path.resolve()),
                progress_label=f"checkpoint-{step}",
                progress_every_batches=args.progress_every_batches,
                use_vllm_tqdm=args.vllm_tqdm,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            dt_ck = time.perf_counter() - t_ck0
            ck_run_wall.append(dt_ck)
            rem_after = n_to_run - len(ck_run_wall)
            if ck_run_wall and rem_after > 0:
                avg_c = sum(ck_run_wall) / len(ck_run_wall)
                eta_rest = format_eta(avg_c * rem_after)
            else:
                eta_rest = "0s" if rem_after == 0 else "?"
            print(
                f"{PROGRESS_PREFIX} in-process step={step} eval wall={format_eta(dt_ck)} "
                f"ETA_remaining≈{eta_rest}",
                flush=True,
            )
            pf = row.pop("_per_frame_rows", [])
            write_summary_csv(run_dir / SUMMARY_CSV_NAME, row)
            if args.full_metrics:
                write_full_metrics_artifacts(run_dir, pf, f"mevis_val_checkpoint_{step}")
            print(
                f"[mevis_val_bbox] step={step} n_eval={row['n_eval']} "
                f"mIoU={row['mIoU']:.4f} mean_GIoU={row['mean_GIoU']:.4f} wall={row['wall_seconds']}s",
                flush=True,
            )
            print(f"[mevis_val_bbox] Wrote {run_dir / SUMMARY_CSV_NAME}", flush=True)
            if args.full_metrics:
                print(f"[mevis_val_bbox] Wrote full metrics under {run_dir}", flush=True)
            gc.collect()


if __name__ == "__main__":
    main()
