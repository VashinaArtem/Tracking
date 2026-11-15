"""
bbox_eval_checkpoint.py

Evaluates a single LoRA checkpoint on 500 fixed valid_u frame-expression pairs.
Computes mean GIoU and appends the result to a CSV log.

Usage:
    python bbox_eval_checkpoint.py \
        --checkpoint_path /path/to/lora_v2/checkpoint-5000 \
        --gpu 1 \
        --csv_path /path/to/bbox_eval_log.csv
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

N_SAMPLES   = 500
SAMPLE_SEED = 42
MODEL_ID    = "openbmb/MiniCPM-V-4_5"

SCRIPT_DIR  = Path(__file__).resolve().parent
VALID_U_DIR = SCRIPT_DIR.parent / "data_sets" / "valid_u"
META_PATH   = VALID_U_DIR / "meta_expressions_v2.json"
MASK_PATH   = VALID_U_DIR / "mask_dict.json"
IMG_DIR     = VALID_U_DIR / "JPEGImages"

CSV_COLUMNS = ["step", "checkpoint_path", "n_samples", "mean_iou", "mean_giou", "timestamp"]


# ---------------------------------------------------------------------------
# GT bbox from RLE mask  (mirrors run_bbox_eval_vllm.py)
# ---------------------------------------------------------------------------

def extract_gt_bbox(rle) -> List[int]:
    """Returns [x_min, x_max, y_min, y_max] in pixel coords, or [0,0,0,0] if empty."""
    import numpy as np
    from pycocotools import mask as mask_utils
    if rle is None:
        return [0, 0, 0, 0]
    decoded = mask_utils.decode(rle)
    rows = np.any(decoded, axis=1)
    cols = np.any(decoded, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y_min = int(rows.argmax())
    y_max = int(len(rows) - 1 - rows[::-1].argmax())
    x_min = int(cols.argmax())
    x_max = int(len(cols) - 1 - cols[::-1].argmax())
    return [x_min, x_max, y_min, y_max]


# ---------------------------------------------------------------------------
# Denormalize 0-1000 model output → pixel coords  (mirrors run_bbox_eval_vllm.py)
# ---------------------------------------------------------------------------

def denormalize(bbox: List[int], height: int, width: int) -> List[int]:
    """Input: [x1, y1, x2, y2] in 0-1000 range. Output: [x_min, x_max, y_min, y_max] px."""
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * width  / 1000),
        round(x2 * width  / 1000),
        round(y1 * height / 1000),
        round(y2 * height / 1000),
    ]


# ---------------------------------------------------------------------------
# BBox parser  (mirrors run_bbox_eval_vllm.py)
# ---------------------------------------------------------------------------

def extract_bbox(response: str) -> Optional[List[int]]:
    m = re.search(r"<box>(.*?)</box>", response, flags=re.DOTALL)
    if m:
        nums = re.findall(r"[-+]?\d*\.?\d+", m.group(1))
        if len(nums) >= 4:
            return [int(round(float(nums[i]))) for i in range(4)]
    nums = re.findall(r"\d+", response)
    if len(nums) >= 4:
        return [int(nums[i]) for i in range(4)]
    return None


# ---------------------------------------------------------------------------
# GIoU  (mirrors metrics_finetune.py _bbox_metrics)
# ---------------------------------------------------------------------------

def compute_giou(gt: List[int], pred: List[int]) -> Tuple[float, float]:
    """
    gt / pred: [x_min, x_max, y_min, y_max] in pixel coords.
    Returns (iou, giou).
    """
    x1g, x2g, y1g, y2g = gt
    x1p, x2p, y1p, y2p = pred

    xi1, yi1 = max(x1g, x1p), max(y1g, y1p)
    xi2, yi2 = min(x2g, x2p), min(y2g, y2p)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    gt_area   = (x2g - x1g) * (y2g - y1g)
    pred_area = (x2p - x1p) * (y2p - y1p)
    union = gt_area + pred_area - inter
    iou   = inter / union if union > 0 else 0.0

    cx1, cy1 = min(x1g, x1p), min(y1g, y1p)
    cx2, cy2 = max(x2g, x2p), max(y2g, y2p)
    c_area = (cx2 - cx1) * (cy2 - cy1)
    giou = iou - (c_area - union) / c_area if c_area > 0 else iou
    return iou, giou


# ---------------------------------------------------------------------------
# Prompt  (mirrors run_bbox_eval_vllm.py)
# ---------------------------------------------------------------------------

def format_question(expr: str) -> str:
    return (
        f"Please provide the bounding box coordinate of the region this sentence describes: "
        f"<ref>{expr}</ref>. Return format: <box>x1 y1 x2 y2</box>"
    )


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": f"(<image>./</image>)\n{question}"}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )


# ---------------------------------------------------------------------------
# Build fixed sample list (deterministic, seed=42)
# ---------------------------------------------------------------------------

def build_sample_list() -> list:
    """
    Builds a fixed list of N_SAMPLES valid (video_id, expr_id, frame_idx, anno_id, frame_name)
    tuples from valid_u, filtering out multi-object expressions and empty GT masks.
    """
    import numpy as np
    from pycocotools import mask as mask_utils

    with open(META_PATH) as f:
        meta = json.load(f)
    with open(MASK_PATH) as f:
        mask_dict = json.load(f)

    all_pairs = []
    for video_id, vmeta in meta["videos"].items():
        frames = vmeta["frames"]
        for expr_id, ex in vmeta["expressions"].items():
            if len(ex["obj_id"]) != 1:
                continue
            anno_id = str(ex["anno_id"][0])
            for frame_idx, frame_name in enumerate(frames):
                img_path = IMG_DIR / video_id / f"{frame_name}.jpg"
                if not img_path.exists():
                    continue
                if anno_id not in mask_dict or frame_idx >= len(mask_dict[anno_id]):
                    continue
                rle = mask_dict[anno_id][frame_idx]
                if rle is None:
                    continue
                decoded = mask_utils.decode(rle)
                if not np.any(decoded):
                    continue
                all_pairs.append((video_id, expr_id, frame_idx, anno_id, frame_name, ex["exp"]))

    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(all_pairs)
    return all_pairs[:N_SAMPLES]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA checkpoint on 500 fixed valid_u pairs (GIoU)"
    )
    parser.add_argument("--checkpoint_path", required=True,
                        help="Path to LoRA checkpoint directory (e.g. .../lora_v2/checkpoint-5000)")
    parser.add_argument("--gpu",  type=str, required=True,
                        help="CUDA_VISIBLE_DEVICES value, e.g. '0'")
    parser.add_argument("--csv_path", required=True,
                        help="Path to output CSV log file")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Must happen before any torch / vllm import
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    checkpoint_path = Path(args.checkpoint_path).resolve()
    csv_path        = Path(args.csv_path).resolve()

    # Sanity check: adapter weights must exist and be at least 60s old
    adapter_file = checkpoint_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        print(f"[bbox_eval] ERROR: adapter not found: {adapter_file}", flush=True)
        sys.exit(1)
    age = time.time() - adapter_file.stat().st_mtime
    if age < 60:
        print(f"[bbox_eval] ERROR: adapter written {age:.0f}s ago, too recent. Aborting.", flush=True)
        sys.exit(1)

    # Extract step number from checkpoint dir name
    try:
        step = int(checkpoint_path.name.split("-")[-1])
    except ValueError:
        step = -1

    print(f"[bbox_eval] checkpoint={checkpoint_path.name}  step={step}  gpu={args.gpu}", flush=True)

    # -----------------------------------------------------------------------
    # Build samples (loaded before vllm to keep GPU free during CPU work)
    # -----------------------------------------------------------------------
    print("[bbox_eval] Building sample list ...", flush=True)
    from PIL import Image

    samples = build_sample_list()
    print(f"[bbox_eval] {len(samples)} samples ready.", flush=True)

    # -----------------------------------------------------------------------
    # Lazy imports (after CUDA_VISIBLE_DEVICES is set)
    # -----------------------------------------------------------------------
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print(f"[bbox_eval] Loading base model {MODEL_ID} + LoRA ...", flush=True)
    t0 = time.time()

    lora_request = LoRARequest("mevis_lora", 1, str(checkpoint_path))

    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        enable_lora=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    _stop_names = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = list({
        tid
        for name in _stop_names
        for tid in [tokenizer.convert_tokens_to_ids(name)]
        if isinstance(tid, int) and tid != tokenizer.unk_token_id
    } | {tokenizer.eos_token_id})

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
    )
    print(f"[bbox_eval] Model ready in {time.time() - t0:.1f}s.", flush=True)

    # -----------------------------------------------------------------------
    # Batched inference
    # -----------------------------------------------------------------------
    import numpy as np
    from pycocotools import mask as mask_utils

    with open(MASK_PATH) as f:
        mask_dict = json.load(f)

    ious, gious = [], []
    n_batches = (len(samples) + args.batch_size - 1) // args.batch_size
    t_infer = time.time()

    for b in range(n_batches):
        batch = samples[b * args.batch_size : (b + 1) * args.batch_size]

        vllm_inputs, images_pil, gt_bboxes = [], [], []
        for (video_id, expr_id, frame_idx, anno_id, frame_name, expr_text) in batch:
            img_path = IMG_DIR / video_id / f"{frame_name}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                images_pil.append(None)
                vllm_inputs.append(None)
                gt_bboxes.append(None)
                continue

            rle = mask_dict.get(anno_id, [None])[frame_idx] \
                if anno_id in mask_dict and frame_idx < len(mask_dict[anno_id]) else None
            gt = extract_gt_bbox(rle)

            question   = format_question(expr_text)
            prompt_str = build_prompt(tokenizer, question)

            images_pil.append(img)
            vllm_inputs.append({"prompt": prompt_str, "multi_modal_data": {"image": img}})
            gt_bboxes.append(gt)

        valid_idx    = [i for i, x in enumerate(vllm_inputs) if x is not None]
        valid_inputs = [vllm_inputs[i] for i in valid_idx]

        if valid_inputs:
            outputs = llm.generate(valid_inputs, sampling_params=sampling_params,
                                   lora_request=lora_request)
        else:
            outputs = []

        out_map = {valid_idx[j]: outputs[j] for j in range(len(valid_idx))}

        for i, (video_id, expr_id, frame_idx, anno_id, frame_name, _) in enumerate(batch):
            img = images_pil[i]
            gt  = gt_bboxes[i]
            if img is None or gt is None:
                continue

            out = out_map.get(i)
            if out is None:
                continue

            raw = out.outputs[0].text
            clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            pred_raw = extract_bbox(clean)

            if pred_raw is None:
                # Failed parse → pred at origin = worst GIoU
                pred_px = [0, 0, 0, 0]
            else:
                h, w = img.size[1], img.size[0]
                pred_px = denormalize(pred_raw, h, w)

            iou, giou = compute_giou(gt, pred_px)
            ious.append(iou)
            gious.append(giou)

        if (b + 1) % 5 == 0 or b == n_batches - 1:
            print(
                f"[bbox_eval] batch {b+1}/{n_batches}  "
                f"samples_so_far={len(gious)}  "
                f"mean_giou={sum(gious)/len(gious):.4f}",
                flush=True,
            )

    elapsed = time.time() - t_infer
    mean_iou  = float(sum(ious)  / len(ious))  if ious  else 0.0
    mean_giou = float(sum(gious) / len(gious)) if gious else 0.0

    print(
        f"[bbox_eval] DONE  step={step}  n={len(gious)}  "
        f"mean_iou={mean_iou:.4f}  mean_giou={mean_giou:.4f}  "
        f"elapsed={elapsed:.0f}s",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Append to CSV
    # -----------------------------------------------------------------------
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(CSV_COLUMNS)
        writer.writerow([
            step,
            str(checkpoint_path),
            len(gious),
            round(mean_iou,  6),
            round(mean_giou, 6),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])

    print(f"[bbox_eval] Result appended to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
