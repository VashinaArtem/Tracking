"""
sample_100_raw_responses.py

Takes exactly 100 samples from FirstMetricsData/human_animal_object,
runs them through MiniCPM-V-4.5 (vLLM), and saves raw_response to CSV.

Usage:
    python sample_100_raw_responses.py --gpu 1
    python sample_100_raw_responses.py --gpu 1 --n_samples 100 --output my_results.csv
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

SCRIPT_DIR = Path(__file__).parent
MEVIS_DIR  = SCRIPT_DIR.parent
DATA_DIR   = MEVIS_DIR / "data_sets/train"
FM_DIR     = DATA_DIR  / "FirstMetricsData"
META_PATH  = DATA_DIR  / "meta_expressions_v2.json"
MASK_PATH  = DATA_DIR  / "mask_dict.json"

MODEL_ID = "openbmb/MiniCPM-V-4_5"
CATEGORY = "human_animal_object"

CSV_COLUMNS = [
    "video_name", "expr_id", "prompt",
    "frame_idx", "raw_response", "clean_response",
    "success", "pred_bbox_raw",
    "height", "width",
    "x_min_gt", "x_max_gt", "y_min_gt", "y_max_gt",
    "inference_time",
]


# ---------------------------------------------------------------------------
# Helpers (copied from run_bbox_eval_vllm.py)
# ---------------------------------------------------------------------------

class BBoxParser:
    @staticmethod
    def extract_bbox(response: str) -> Optional[List[int]]:
        bbox = BBoxParser._from_box_tags(response)
        if bbox:
            return bbox
        return BBoxParser._from_consecutive_numbers(response)

    @staticmethod
    def _from_box_tags(response: str) -> Optional[List[int]]:
        m = re.search(r"<box>(.*?)</box>", response, flags=re.DOTALL)
        if m:
            nums = re.findall(r"[-+]?\d*\.?\d+", m.group(1))
            if len(nums) >= 4:
                return [int(round(float(nums[i]))) for i in range(4)]
        return None

    @staticmethod
    def _from_consecutive_numbers(response: str) -> Optional[List[int]]:
        nums = re.findall(r"\d+", response)
        if len(nums) >= 4:
            return [int(nums[i]) for i in range(4)]
        return None


def extract_gt_bbox(rle) -> List[int]:
    if rle is None:
        return [0, 0, 0, 0]
    decoded = mask_utils.decode(rle)
    rows = np.any(decoded, axis=1)
    cols = np.any(decoded, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y_min = int(np.argmax(rows))
    y_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x_min = int(np.argmax(cols))
    x_max = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return [x_min, x_max, y_min, y_max]


def format_grounding_question(expr: str) -> str:
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",        type=int, required=True)
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_path", type=str, default=MODEL_ID)
    parser.add_argument("--output",     type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    out_path = Path(args.output) if args.output else (
        MEVIS_DIR / f"results_finetune/sample100_human_animal_object_raw.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_path}")
    print(f"Loading metadata...")

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    with open(MASK_PATH, encoding="utf-8") as f:
        mask_dict = json.load(f)

    # -- collect all available work-items from human_animal_object --
    cat_dir = FM_DIR / CATEGORY
    all_videos = sorted(x.name for x in cat_dir.iterdir() if x.is_dir())

    work_items = []
    for video_id in all_videos:
        if video_id not in meta.get("videos", {}):
            continue
        vmeta  = meta["videos"][video_id]
        frames = vmeta["frames"]
        eligible = [
            (eid, ex)
            for eid, ex in vmeta["expressions"].items()
            if len(ex["obj_id"]) == 1
        ]
        for expr_id, expr in eligible:
            anno_id = str(expr["anno_id"][0])
            for i, frame_name in enumerate(frames):
                img_path = cat_dir / video_id / f"{frame_name}.jpg"
                if not img_path.exists():
                    continue
                work_items.append({
                    "video_id":   video_id,
                    "expr_id":    expr_id,
                    "prompt_txt": expr["exp"],
                    "anno_id":    anno_id,
                    "frame_idx":  i,
                    "n_frames":   len(frames),
                    "img_path":   img_path,
                })

    print(f"Total available work items: {len(work_items)}")

    # uniformly select n_samples
    import random
    random.seed(42)
    selected = random.sample(work_items, min(args.n_samples, len(work_items)))
    selected.sort(key=lambda x: (x["video_id"], x["expr_id"], x["frame_idx"]))
    print(f"Selected {len(selected)} samples")

    # -- vLLM init --
    print(f"Initialising vLLM (model={args.model_path}) ...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

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
    print("vLLM ready.")

    # -- batched inference --
    batch_size   = args.batch_size
    n_batches    = (len(selected) + batch_size - 1) // batch_size
    results      = []
    total_start  = time.time()

    for batch_idx in range(n_batches):
        batch = selected[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        vllm_inputs   = []
        loaded_images = []
        valid_batch_items = []

        for item in batch:
            try:
                img = Image.open(item["img_path"]).convert("RGB")
            except Exception as e:
                print(f"  WARN: cannot open {item['img_path']}: {e}")
                continue
            question   = format_grounding_question(item["prompt_txt"])
            prompt_str = build_prompt(tokenizer, question)
            loaded_images.append(img)
            vllm_inputs.append({
                "prompt":           prompt_str,
                "multi_modal_data": {"image": img},
            })
            valid_batch_items.append(item)

        if not vllm_inputs:
            continue

        t0 = time.time()
        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
        elapsed = time.time() - t0
        per_item = elapsed / len(vllm_inputs)

        for item, img, out in zip(valid_batch_items, loaded_images, outputs):
            raw_response = out.outputs[0].text
            clean        = re.sub(r"<think>.*?</think>", "", raw_response,
                                  flags=re.DOTALL).strip()
            pred_raw     = BBoxParser.extract_bbox(clean)
            ok           = pred_raw is not None

            height, width = img.size[1], img.size[0]

            rle = (
                mask_dict[item["anno_id"]][item["frame_idx"]]
                if item["anno_id"] in mask_dict and item["frame_idx"] < len(mask_dict[item["anno_id"]])
                else None
            )
            gt_bbox = extract_gt_bbox(rle)

            results.append({
                "video_name":     item["video_id"],
                "expr_id":        item["expr_id"],
                "prompt":         item["prompt_txt"],
                "frame_idx":      item["frame_idx"],
                "raw_response":   raw_response,
                "clean_response": clean,
                "success":        ok,
                "pred_bbox_raw":  str(pred_raw) if pred_raw else "",
                "height":         height,
                "width":          width,
                "x_min_gt":       gt_bbox[0],
                "x_max_gt":       gt_bbox[1],
                "y_min_gt":       gt_bbox[2],
                "y_max_gt":       gt_bbox[3],
                "inference_time": round(per_item, 4),
            })

        done = len(results)
        print(f"Batch {batch_idx+1}/{n_batches}  processed={done}/{len(selected)}  "
              f"batch_t={elapsed:.2f}s")

    # -- write CSV --
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

    total_elapsed = time.time() - total_start
    n_ok = sum(1 for r in results if r["success"])
    print(f"\nDone in {total_elapsed:.1f}s  |  success={n_ok}/{len(results)}")
    print(f"CSV saved to: {out_path}")

    # -- brief raw_response statistics --
    print("\n=== Sample raw_response examples ===")
    for r in results[:20]:
        print(f"[{r['video_name']} / expr={r['expr_id']} / f={r['frame_idx']}]")
        print(f"  prompt : {r['prompt']}")
        print(f"  raw    : {repr(r['raw_response'])}")
        print(f"  clean  : {repr(r['clean_response'])}")
        print(f"  success: {r['success']}  pred={r['pred_bbox_raw']}")
        print()


if __name__ == "__main__":
    main()
8