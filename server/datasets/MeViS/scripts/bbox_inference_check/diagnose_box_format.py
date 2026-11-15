"""
diagnose_box_format.py

Диагностика формата ответов MiniCPM-V-4.5 через vLLM на valid_u.
Проверяет: есть ли <box>...</box> теги в ответах или модель использует fallback.

Usage:
    # базовая модель без LoRA (диагностика):
    python diagnose_box_format.py --gpu 0

    # с LoRA — для сравнения:
    python diagnose_box_format.py --gpu 0 --lora_path /path/to/saves/lora

    # кастомное кол-во сэмплов:
    python diagnose_box_format.py --gpu 0 --n_samples 200
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
MEVIS_DIR  = SCRIPT_DIR.parent
VALID_U_DIR = MEVIS_DIR / "data_sets/valid_u"
META_PATH   = VALID_U_DIR / "meta_expressions_v2.json"
MASK_PATH   = VALID_U_DIR / "mask_dict.json"
IMAGES_DIR  = VALID_U_DIR / "JPEGImages"

MODEL_ID = "openbmb/MiniCPM-V-4_5"

OUTPUT_DIR = SCRIPT_DIR / "diagnose_box_format_results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class BBoxParser:
    @staticmethod
    def extract_bbox(response: str) -> Optional[List[int]]:
        bbox = BBoxParser._from_box_tags(response)
        if bbox:
            return bbox, "box_tags"
        bbox = BBoxParser._from_consecutive_numbers(response)
        if bbox:
            return bbox, "fallback"
        return None, "failed"

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
    parser = argparse.ArgumentParser(
        description="Diagnose <box> tag format in MiniCPM-V-4.5 responses on valid_u"
    )
    parser.add_argument("--gpu",        type=int, required=True,
                        help="GPU index for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--n_samples",  type=int, default=200,
                        help="Number of frames to sample (default: 200)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_path", type=str, default=MODEL_ID)
    parser.add_argument("--lora_path",  type=str, default=None,
                        help="LoRA adapter path (optional; omit for base model diagnosis)")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Output paths
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "lora" if args.lora_path else "baseline"
    csv_path     = OUTPUT_DIR / f"diagnose_{tag}_n{args.n_samples}.csv"
    report_path  = OUTPUT_DIR / f"diagnose_{tag}_n{args.n_samples}_report.txt"

    print("=" * 60)
    print(f"Diagnose box format  |  model={args.model_path}")
    print(f"LoRA: {args.lora_path or 'None (base model)'}")
    print(f"Samples: {args.n_samples}  |  GPU: {args.gpu}  |  seed: {args.seed}")
    print(f"Output CSV   : {csv_path}")
    print(f"Output report: {report_path}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Load metadata
    # -----------------------------------------------------------------------
    print("Loading valid_u metadata...")
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    with open(MASK_PATH, encoding="utf-8") as f:
        mask_dict = json.load(f)

    # -----------------------------------------------------------------------
    # Build work items from valid_u
    # -----------------------------------------------------------------------
    work_items = []
    for video_id, vmeta in meta["videos"].items():
        frames = vmeta["frames"]
        for expr_id, expr in vmeta["expressions"].items():
            obj_ids  = expr.get("obj_id", [])
            anno_ids = expr.get("anno_id", [])
            if len(obj_ids) != 1:
                continue  # skip multi-object expressions
            anno_id = str(anno_ids[0])
            for i, frame_name in enumerate(frames):
                img_path = IMAGES_DIR / video_id / f"{frame_name}.jpg"
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

    print(f"Total available work items in valid_u: {len(work_items)}")

    import random
    random.seed(args.seed)
    selected = random.sample(work_items, min(args.n_samples, len(work_items)))
    selected.sort(key=lambda x: (x["video_id"], x["expr_id"], x["frame_idx"]))
    print(f"Sampled {len(selected)} items (seed={args.seed})")

    # -----------------------------------------------------------------------
    # Init vLLM
    # -----------------------------------------------------------------------
    print(f"\nInitialising vLLM (model={args.model_path}) ...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("diag_lora", 1, args.lora_path)
        print(f"LoRA adapter: {args.lora_path}")

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        enable_lora=lora_request is not None,
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
    print("vLLM ready.\n")

    # -----------------------------------------------------------------------
    # Batched inference
    # -----------------------------------------------------------------------
    CSV_COLUMNS = [
        "video_id", "expr_id", "prompt", "frame_idx",
        "raw_response", "clean_response",
        "parse_method",   # "box_tags" | "fallback" | "failed"
        "pred_bbox",
        "height", "width",
        "x_min_gt", "x_max_gt", "y_min_gt", "y_max_gt",
        "inference_time",
    ]

    results    = []
    total_start = time.time()
    n_batches   = (len(selected) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(n_batches):
        batch = selected[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

        vllm_inputs   = []
        loaded_images = []
        valid_items   = []

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
            valid_items.append(item)

        if not vllm_inputs:
            continue

        t0 = time.time()
        gen_kwargs = {"sampling_params": sampling_params}
        if lora_request:
            gen_kwargs["lora_request"] = lora_request
        outputs = llm.generate(vllm_inputs, **gen_kwargs)
        elapsed = time.time() - t0
        per_item = elapsed / len(vllm_inputs)

        for item, img, out in zip(valid_items, loaded_images, outputs):
            raw_response  = out.outputs[0].text
            clean         = re.sub(r"<think>.*?</think>", "", raw_response,
                                   flags=re.DOTALL).strip()
            pred_raw, parse_method = BBoxParser.extract_bbox(clean)

            height, width = img.size[1], img.size[0]

            rle = (
                mask_dict[item["anno_id"]][item["frame_idx"]]
                if item["anno_id"] in mask_dict
                   and item["frame_idx"] < len(mask_dict[item["anno_id"]])
                else None
            )
            gt_bbox = extract_gt_bbox(rle)

            results.append({
                "video_id":       item["video_id"],
                "expr_id":        item["expr_id"],
                "prompt":         item["prompt_txt"],
                "frame_idx":      item["frame_idx"],
                "raw_response":   raw_response,
                "clean_response": clean,
                "parse_method":   parse_method,
                "pred_bbox":      str(pred_raw) if pred_raw else "",
                "height":         height,
                "width":          width,
                "x_min_gt":       gt_bbox[0],
                "x_max_gt":       gt_bbox[1],
                "y_min_gt":       gt_bbox[2],
                "y_max_gt":       gt_bbox[3],
                "inference_time": round(per_item, 4),
            })

        print(f"Batch {batch_idx+1}/{n_batches}  "
              f"processed={len(results)}/{len(selected)}  "
              f"batch_t={elapsed:.2f}s")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Statistics & report
    # -----------------------------------------------------------------------
    total          = len(results)
    n_box_tags     = sum(1 for r in results if r["parse_method"] == "box_tags")
    n_fallback     = sum(1 for r in results if r["parse_method"] == "fallback")
    n_failed       = sum(1 for r in results if r["parse_method"] == "failed")
    elapsed_total  = time.time() - total_start

    report_lines = [
        "=" * 60,
        f"DIAGNOSE BOX FORMAT REPORT",
        f"Model      : {args.model_path}",
        f"LoRA       : {args.lora_path or 'None (base model)'}",
        f"Samples    : {total}",
        f"Elapsed    : {elapsed_total:.1f}s",
        "=" * 60,
        "",
        "=== Parse method breakdown ===",
        f"  <box> tags (correct format) : {n_box_tags:>5} / {total}  ({100*n_box_tags/total:.1f}%)",
        f"  Fallback (first 4 numbers)  : {n_fallback:>5} / {total}  ({100*n_fallback/total:.1f}%)",
        f"  Failed (no bbox found)      : {n_failed:>5} / {total}  ({100*n_failed/total:.1f}%)",
        "",
        "=== Diagnosis ===",
    ]

    if n_box_tags / total >= 0.95:
        report_lines.append("  ✓ GOOD: >95% responses use <box> tags — format is correct.")
    elif n_box_tags / total >= 0.5:
        report_lines.append("  ~ PARTIAL: 50-95% responses use <box> tags — partial format issue.")
    else:
        report_lines.append("  ✗ BAD: <50% responses use <box> tags — format problem detected.")
        if args.lora_path is None:
            report_lines.append("    → Problem is in apply_chat_template, NOT in LoRA weights.")
            report_lines.append("    → Fix the chat template BEFORE training.")
        else:
            report_lines.append("    → Problem may be in LoRA weights or chat template.")
            report_lines.append("    → Run WITHOUT --lora_path to isolate the issue.")

    report_lines += [
        "",
        "=== Example responses (first 10) ===",
    ]
    for r in results[:10]:
        report_lines += [
            f"  [{r['video_id']} / expr={r['expr_id']} / frame={r['frame_idx']}]",
            f"  prompt : {r['prompt']}",
            f"  raw    : {repr(r['raw_response'][:120])}",
            f"  method : {r['parse_method']}  |  pred={r['pred_bbox']}",
            "",
        ]

    report_lines += [
        "=== Box-tag examples (up to 3) ===",
    ]
    box_examples = [r for r in results if r["parse_method"] == "box_tags"][:3]
    for r in box_examples:
        report_lines += [
            f"  raw: {repr(r['raw_response'][:120])}",
            "",
        ]

    report_lines += [
        "=== Fallback examples (up to 3) ===",
    ]
    fallback_examples = [r for r in results if r["parse_method"] == "fallback"][:3]
    for r in fallback_examples:
        report_lines += [
            f"  raw: {repr(r['raw_response'][:120])}",
            "",
        ]

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
