"""
run_bbox_eval_vllm.py

Evaluate MiniCPM-V-4.5 bounding box predictions on MeViS FirstMetricsData videos
using vLLM batched inference — faster than the sequential model.chat() approach.

Usage:
    python run_bbox_eval_vllm.py --category humans_frequency --part 0 --gpu 0
    python run_bbox_eval_vllm.py --category human_object --part 1 --gpu 3 --batch_size 32
    python run_bbox_eval_vllm.py --category humans_frequency --part 0 --gpu 0 \\
        --lora_path /path/to/saves/minicpmv4_5/mevis/lora
    python run_bbox_eval_vllm.py --category humans_frequency --part 0 --gpu 0 --test

Output CSV columns are identical to run_bbox_eval.py, so metrics.ipynb works unchanged.
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR          = Path(__file__).parent           # .../MeViS/scripts/
MEVIS_DIR           = SCRIPT_DIR.parent               # .../MeViS/
DATA_DIR            = MEVIS_DIR / "data_sets/train"
FM_DIR              = DATA_DIR / "FirstMetricsData"
META_PATH           = DATA_DIR / "meta_expressions_v2.json"
MASK_PATH           = DATA_DIR / "mask_dict.json"
DEFAULT_RESULTS_DIR = MEVIS_DIR / "results_finetune"

MODEL_ID = "openbmb/MiniCPM-V-4_5"

CATEGORY_DIRS = {
    "humans_frequency":    "humans_frequency",
    "human_object":        "human_object",
    "human_animal_object": "human_animal_object",
}

CSV_COLUMNS = [
    "video_name", "category", "expr_id", "prompt",
    "success", "raw_response", "inference_time",
    "height", "width", "total_frames", "n_frame",
    "x_min_gt", "x_max_gt", "y_min_gt", "y_max_gt",
    "x_min_pred_norm", "x_max_pred_norm", "y_min_pred_norm", "y_max_pred_norm",
    "x_min_pred_px",   "x_max_pred_px",   "y_min_pred_px",   "y_max_pred_px",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt     = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("bbox_eval_vllm")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# BBoxParser  (identical to run_bbox_eval.py)
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


# ---------------------------------------------------------------------------
# GT bbox from RLE mask  (identical to run_bbox_eval.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Denormalization: 0-1000 → pixels  (identical to run_bbox_eval.py)
# ---------------------------------------------------------------------------

def denormalize(bbox: List[int], height: int, width: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * width  / 1000),
        round(x2 * width  / 1000),
        round(y1 * height / 1000),
        round(y2 * height / 1000),
    ]


# ---------------------------------------------------------------------------
# Prompt  (identical to run_bbox_eval.py)
# ---------------------------------------------------------------------------

def format_grounding_question(expr: str) -> str:
    return (
        f"Please provide the bounding box coordinate of the region this sentence describes: "
        f"<ref>{expr}</ref>. Return format: <box>x1 y1 x2 y2</box>"
    )


# ---------------------------------------------------------------------------
# Video deduplication  (identical to run_bbox_eval.py)
# ---------------------------------------------------------------------------

def get_deduplicated_videos(category: str) -> List[str]:
    def load_videos(cat: str) -> set:
        d = FM_DIR / cat
        return {x.name for x in d.iterdir() if x.is_dir()} if d.exists() else set()

    hf  = load_videos("humans_frequency")
    ho  = load_videos("human_object")
    hao = load_videos("human_animal_object")

    if   category == "humans_frequency":    videos = hf
    elif category == "human_object":        videos = ho - hf
    elif category == "human_animal_object": videos = hao - hf - ho
    else: raise ValueError(f"Unknown category: {category}")

    return sorted(videos)


# ---------------------------------------------------------------------------
# Checkpoint  (identical to run_bbox_eval.py)
# ---------------------------------------------------------------------------

def load_checkpoint(csv_path: Path, logger: logging.Logger):
    done: set = set()
    if not csv_path.exists():
        logger.info("No existing checkpoint — starting fresh.")
        return done
    logger.info("Reading checkpoint from %s ...", csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done.add((row["video_name"], row["expr_id"], int(row["n_frame"])))
            except (KeyError, ValueError):
                pass
    logger.info("Checkpoint loaded: %d rows already done.", len(done))
    return done


# ---------------------------------------------------------------------------
# Build vLLM prompt string
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, question: str) -> str:
    """
    Apply MiniCPM-V chat template with thinking disabled.
    Image placeholder (<image>./</image>) is the format expected by vLLM's
    MiniCPM-V processor.
    """
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
        description="BBox evaluation with MiniCPM-V-4.5 via vLLM (batched)"
    )
    parser.add_argument("--category", required=True, choices=list(CATEGORY_DIRS.keys()))
    parser.add_argument("--part",       type=int,  required=True, choices=[0, 1],
                        help="0 = first 50%%, 1 = second 50%%")
    parser.add_argument("--gpu",        type=int,  required=True,
                        help="Physical GPU index (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--batch_size", type=int,  default=16,
                        help="Frames per vLLM batch (default 16)")
    parser.add_argument("--model_path", type=str,  default=MODEL_ID,
                        help=f"HuggingFace ID or local path (default: {MODEL_ID})")
    parser.add_argument("--lora_path",  type=str,  default=None,
                        help="Path to LoRA adapter directory (optional)")
    parser.add_argument("--output",     type=str,  default=None,
                        help="Custom output CSV path")
    parser.add_argument("--test",       action="store_true",
                        help="Process only first 3 videos")
    args = parser.parse_args()

    # Must be set before any torch/vllm import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Output paths
    if args.output:
        csv_path = Path(args.output).resolve()
    else:
        tag = "_vllm" + ("_lora" if args.lora_path else "")
        csv_path = DEFAULT_RESULTS_DIR / f"{args.category}_part{args.part}{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = csv_path.with_suffix(".log")

    logger  = setup_logging(log_path)
    run_tag = f"{args.category}/part{args.part}"

    logger.info("=" * 60)
    logger.info(
        "BBox Eval (vLLM) started  |  %s  |  GPU=%d  |  batch_size=%d%s%s",
        run_tag, args.gpu, args.batch_size,
        f"  |  LoRA={args.lora_path}" if args.lora_path else "",
        "  [TEST MODE]" if args.test else "",
    )
    logger.info("Model      : %s", args.model_path)
    logger.info("Output CSV : %s", csv_path)
    logger.info("Log file   : %s", log_path)

    # Video list
    all_videos = get_deduplicated_videos(args.category)
    mid = len(all_videos) // 2
    selected = all_videos[:mid] if args.part == 0 else all_videos[mid:]
    if args.test:
        selected = selected[:3]
    logger.info("Videos in this part: %d%s", len(selected),
                " (first 3 only — TEST)" if args.test else "")

    # Checkpoint
    done = load_checkpoint(csv_path, logger)

    # Metadata
    logger.info("Loading meta_expressions_v2.json ...")
    t0 = time.time()
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    logger.info("Loaded in %.1f s.", time.time() - t0)

    logger.info("Loading mask_dict.json ...")
    t0 = time.time()
    with open(MASK_PATH, encoding="utf-8") as f:
        mask_dict = json.load(f)
    logger.info("Loaded in %.1f s.", time.time() - t0)

    # -----------------------------------------------------------------------
    # vLLM model + tokenizer
    # -----------------------------------------------------------------------
    logger.info("Initialising vLLM engine (model=%s) ...", args.model_path)
    t0 = time.time()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    lora_request = None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("mevis_lora", 1, args.lora_path)
        logger.info("LoRA adapter: %s", args.lora_path)

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        enable_lora=lora_request is not None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )

    # Stop tokens: <|im_end|> and EOS
    _stop_names = ["<|im_end|>", "<|endoftext|>"]
    stop_token_ids = list({
        tid
        for name in _stop_names
        for tid in [tokenizer.convert_tokens_to_ids(name)]
        if isinstance(tid, int) and tid != tokenizer.unk_token_id
    } | {tokenizer.eos_token_id})

    sampling_params = SamplingParams(
        temperature=0.0,      # greedy — matches sampling=False in original
        max_tokens=128,       # <box>x1 y1 x2 y2</box> fits in ~15 tokens; 128 is safe buffer
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
    )

    logger.info("vLLM engine ready in %.1f s.", time.time() - t0)

    # -----------------------------------------------------------------------
    # Build work-item list (skipping checkpoint)
    # -----------------------------------------------------------------------
    logger.info("Building work item list ...")
    WorkItem = Dict  # video_id, expr_id, prompt_txt, anno_id, frame_idx, n_frames, img_path

    work_items: List[WorkItem] = []
    for video_id in selected:
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
                if (video_id, expr_id, i) in done:
                    continue
                img_path = FM_DIR / args.category / video_id / f"{frame_name}.jpg"
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

    total_to_process = len(work_items)
    logger.info("Total frames to process: %d", total_to_process)

    # -----------------------------------------------------------------------
    # Open CSV
    # -----------------------------------------------------------------------
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer   = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    if is_new:
        writer.writerow(CSV_COLUMNS)
        csv_file.flush()
        logger.info("Created new CSV with header.")
    else:
        logger.info("Appending to existing CSV.")

    # -----------------------------------------------------------------------
    # Batched inference
    # -----------------------------------------------------------------------
    run_start       = time.time()
    total_processed = 0
    total_failed    = 0
    n_batches       = (total_to_process + args.batch_size - 1) // args.batch_size

    logger.info("-" * 60)
    logger.info("Starting batched inference: %d batches of up to %d frames ...",
                n_batches, args.batch_size)

    try:
        for batch_idx in range(n_batches):
            batch = work_items[
                batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size
            ]

            # --- load images & build prompts ---
            vllm_inputs:  List[Optional[dict]] = []
            loaded_images: List[Optional[Image.Image]] = []

            for item in batch:
                try:
                    img = Image.open(item["img_path"]).convert("RGB")
                except Exception as e:
                    logger.warning("Cannot open %s: %s", item["img_path"], e)
                    loaded_images.append(None)
                    vllm_inputs.append(None)
                    continue

                question   = format_grounding_question(item["prompt_txt"])
                prompt_str = build_prompt(tokenizer, question)

                loaded_images.append(img)
                vllm_inputs.append({
                    "prompt":           prompt_str,
                    "multi_modal_data": {"image": img},
                })

            # --- run vLLM on valid inputs only ---
            valid_idx    = [i for i, x in enumerate(vllm_inputs) if x is not None]
            valid_inputs = [vllm_inputs[i] for i in valid_idx]

            t_batch = time.time()
            if valid_inputs:
                gen_kwargs: dict = {"sampling_params": sampling_params}
                if lora_request:
                    gen_kwargs["lora_request"] = lora_request
                raw_outputs = llm.generate(valid_inputs, **gen_kwargs)
            else:
                raw_outputs = []
            batch_elapsed = time.time() - t_batch

            # per-item time reported in CSV (amortised batch time)
            per_item_time = (batch_elapsed / len(valid_inputs)) if valid_inputs else 0.0

            output_map = {valid_idx[j]: raw_outputs[j] for j in range(len(valid_idx))}

            # --- write results ---
            for i, item in enumerate(batch):
                img = loaded_images[i]
                if img is None:
                    continue

                out = output_map.get(i)
                if out is not None:
                    raw_response = out.outputs[0].text
                    clean        = re.sub(r"<think>.*?</think>", "", raw_response,
                                         flags=re.DOTALL).strip()
                    pred_raw     = BBoxParser.extract_bbox(clean)
                    ok           = pred_raw is not None
                else:
                    raw_response = "LOAD_ERROR"
                    pred_raw     = None
                    ok           = False

                height, width = img.size[1], img.size[0]

                # GT bbox
                anno_id   = item["anno_id"]
                frame_idx = item["frame_idx"]
                rle = (
                    mask_dict[anno_id][frame_idx]
                    if anno_id in mask_dict and frame_idx < len(mask_dict[anno_id])
                    else None
                )
                gt_bbox = extract_gt_bbox(rle)

                if ok and pred_raw is not None:
                    pred_norm_csv = [pred_raw[0], pred_raw[2], pred_raw[1], pred_raw[3]]
                    pred_px       = denormalize(pred_raw, height, width)
                else:
                    pred_norm_csv = [0, 0, 0, 0]
                    pred_px       = [0, 0, 0, 0]
                    total_failed += 1

                writer.writerow([
                    item["video_id"],
                    args.category,
                    item["expr_id"],
                    item["prompt_txt"],
                    ok,
                    raw_response,
                    round(per_item_time, 4),
                    height,
                    width,
                    item["n_frames"],
                    frame_idx,
                    *gt_bbox,
                    *pred_norm_csv,
                    *pred_px,
                ])

                done.add((item["video_id"], item["expr_id"], frame_idx))
                total_processed += 1

                logger.debug(
                    "  [%s] expr=%s frame=%d  ok=%s  pred=%s",
                    item["video_id"], item["expr_id"], frame_idx, ok, pred_raw,
                )

            csv_file.flush()

            # progress log every 10 batches
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                elapsed = time.time() - run_start
                fps     = total_processed / elapsed if elapsed > 0 else 0
                remain  = total_to_process - total_processed
                eta_s   = remain / fps if fps > 0 else 0
                logger.info(
                    "Batch %d/%d  |  processed=%d/%d  |  failed=%d  |  "
                    "%.2f fps  |  ETA=%dh%02dm  |  batch_t=%.2fs",
                    batch_idx + 1, n_batches,
                    total_processed, total_to_process,
                    total_failed,
                    fps,
                    int(eta_s // 3600), int((eta_s % 3600) // 60),
                    batch_elapsed,
                )

    finally:
        csv_file.close()
        elapsed_total = time.time() - run_start
        fps = total_processed / elapsed_total if elapsed_total > 0 else 0
        logger.info("=" * 60)
        logger.info(
            "DONE  |  %s  |  total=%d  |  failed=%d  |  elapsed=%.0fs  |  avg_fps=%.2f",
            run_tag, total_processed, total_failed, elapsed_total, fps,
        )
        logger.info("CSV closed.")


if __name__ == "__main__":
    main()
