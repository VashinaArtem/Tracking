"""
run_bbox_eval.py

Evaluate MiniCPM-V-4.5 bounding box predictions on MeViS FirstMetricsData videos.

Usage:
    python run_bbox_eval.py --category humans_frequency --part 0 --gpu 0
    python run_bbox_eval.py --category human_object --part 1 --gpu 3 --test
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths (all absolute, relative to this script's location)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent           # .../datasets/MeViS/scripts/
MEVIS_DIR  = SCRIPT_DIR.parent               # .../datasets/MeViS/
DATA_DIR   = MEVIS_DIR / "data_sets/train"   # .../data_sets/train/
FM_DIR     = DATA_DIR / "FirstMetricsData"   # .../FirstMetricsData/
META_PATH  = DATA_DIR / "meta_expressions_v2.json"
MASK_PATH  = DATA_DIR / "mask_dict.json"
DEFAULT_RESULTS_DIR = MEVIS_DIR / "results" # то куда сохраняются данные 

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
    "x_min_pred_px",  "x_max_pred_px",  "y_min_pred_px",  "y_max_pred_px",
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("bbox_eval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# BBoxParser (copied from tracking_client_server/server_gpu_inference/bbox_parser.py)
# ---------------------------------------------------------------------------

class BBoxParser:
    """Parser for extracting bounding boxes from model responses."""

    @staticmethod
    def extract_bbox(response: str) -> Optional[List[int]]:
        bbox = BBoxParser._extract_from_box_tags(response)
        if bbox:
            return bbox
        bbox = BBoxParser._extract_consecutive_numbers(response)
        if bbox:
            return bbox
        return None

    @staticmethod
    def _extract_from_box_tags(response: str) -> Optional[List[int]]:
        match = re.search(r"<box>(.*?)</box>", response, flags=re.DOTALL)
        if match:
            bbox_str = match.group(1)
            nums = re.findall(r"[-+]?\d*\.?\d+", bbox_str)
            if len(nums) >= 4:
                return [int(round(float(nums[i]))) for i in range(4)]
        return None

    @staticmethod
    def _extract_consecutive_numbers(response: str) -> Optional[List[int]]:
        nums = re.findall(r"\d+", response)
        if len(nums) >= 4:
            return [int(nums[i]) for i in range(4)]
        return None


# ---------------------------------------------------------------------------
# Ground truth bbox from RLE mask
# ---------------------------------------------------------------------------

def extract_gt_bbox(rle) -> List[int]:
    """Return [x_min, x_max, y_min, y_max] from an RLE mask, or [0,0,0,0] if None."""
    if rle is None:
        return [0, 0, 0, 0]
    decoded = mask_utils.decode(rle)   # shape (H, W), dtype uint8
    rows = np.any(decoded, axis=1)     # shape (H,)
    cols = np.any(decoded, axis=0)     # shape (W,)
    if not rows.any():
        return [0, 0, 0, 0]
    y_min = int(np.argmax(rows))
    y_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
    x_min = int(np.argmax(cols))
    x_max = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return [x_min, x_max, y_min, y_max]


# ---------------------------------------------------------------------------
# Denormalization: 0-1000 -> pixel coordinates
# ---------------------------------------------------------------------------

def denormalize(bbox: List[int], height: int, width: int) -> List[int]:
    """
    bbox = [x1, y1, x2, y2] in 0-1000 range.
    Returns [x_min_px, x_max_px, y_min_px, y_max_px].
    """
    x1, y1, x2, y2 = bbox
    return [
        round(x1 * width  / 1000),
        round(x2 * width  / 1000),
        round(y1 * height / 1000),
        round(y2 * height / 1000),
    ]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_grounding_question(prompt: str) -> str:
    return (
        f"Please provide the bounding box coordinate of the region this sentence describes: "
        f"<ref>{prompt}</ref>. Return format: <box>x1 y1 x2 y2</box>"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tokenizer, processor, image: Image.Image, prompt: str):
    """
    Returns (pred_bbox_or_None, raw_response, success, elapsed_seconds).
    pred_bbox = [x1, y1, x2, y2] in 0-1000 range (as returned by BBoxParser).
    """
    question = format_grounding_question(prompt)
    t0 = time.time()
    try:
        response = model.chat(
            image=image,
            msgs=[{"role": "user", "content": question}],
            tokenizer=tokenizer,
            processor=processor,
            sampling=False,
            max_new_tokens=64,
            enable_thinking=False,
        )
        elapsed = time.time() - t0
        # Strip <think>...</think> blocks defensively
        clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        bbox = BBoxParser.extract_bbox(clean)
        return bbox, response, bbox is not None, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return None, f"EXCEPTION: {e}", False, elapsed


# ---------------------------------------------------------------------------
# Video deduplication
# ---------------------------------------------------------------------------

def get_deduplicated_videos(category: str) -> List[str]:
    """
    Return sorted list of video IDs belonging to `category` after deduplication.
    Priority: hf > ho > hao
    """
    def load_videos(cat: str) -> set:
        cat_dir = FM_DIR / cat
        if not cat_dir.exists():
            return set()
        return {d.name for d in cat_dir.iterdir() if d.is_dir()}

    hf  = load_videos("humans_frequency")
    ho  = load_videos("human_object")
    hao = load_videos("human_animal_object")

    if category == "humans_frequency":
        videos = hf
    elif category == "human_object":
        videos = ho - hf
    elif category == "human_animal_object":
        videos = hao - hf - ho
    else:
        raise ValueError(f"Unknown category: {category}")

    return sorted(videos)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(csv_path: Path, logger: logging.Logger):
    """Return set of (video_name, expr_id, n_frame) already processed."""
    done = set()
    if not csv_path.exists():
        logger.info("No existing checkpoint found — starting fresh.")
        return done
    logger.info("Reading checkpoint from %s ...", csv_path)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add((row["video_name"], row["expr_id"], int(row["n_frame"])))
            except (KeyError, ValueError):
                pass
    logger.info("Checkpoint loaded: %d rows already done.", len(done))
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BBox evaluation with MiniCPM-V-4.5")
    parser.add_argument("--category", required=True,
                        choices=list(CATEGORY_DIRS.keys()),
                        help="Video category to process")
    parser.add_argument("--part", type=int, required=True, choices=[0, 1],
                        help="0 = first 50%%, 1 = second 50%%")
    parser.add_argument("--gpu", type=int, required=True,
                        help="Physical GPU index (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output CSV (default: results/{category}_part{part}.csv)")
    parser.add_argument("--test", action="store_true",
                        help="Process only first 3 videos (for testing)")
    args = parser.parse_args()

    # Set GPU before any torch import side-effects
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Resolve output CSV path
    if args.output:
        csv_path = Path(args.output).resolve()
    else:
        csv_path = DEFAULT_RESULTS_DIR / f"{args.category}_part{args.part}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Log file sits next to the CSV
    log_path = csv_path.with_suffix(".log")
    logger = setup_logging(log_path)

    run_tag = f"{args.category}/part{args.part}"
    logger.info("=" * 60)
    logger.info("BBox Eval started  |  %s  |  GPU=%d%s",
                run_tag, args.gpu, "  [TEST MODE]" if args.test else "")
    logger.info("Output CSV : %s", csv_path)
    logger.info("Log file   : %s", log_path)

    # Select videos
    logger.info("Collecting video list for category '%s' ...", args.category)
    all_videos = get_deduplicated_videos(args.category)
    mid = len(all_videos) // 2
    selected = all_videos[:mid] if args.part == 0 else all_videos[mid:]
    if args.test:
        selected = selected[:3]
    logger.info("Videos in this part: %d%s", len(selected),
                " (first 3 only — TEST)" if args.test else "")

    # Load checkpoint
    done = load_checkpoint(csv_path, logger)

    # Load metadata
    logger.info("Loading meta_expressions_v2.json  (%s) ...", META_PATH)
    t0 = time.time()
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    logger.info("meta_expressions_v2.json loaded in %.1f s.", time.time() - t0)

    logger.info("Loading mask_dict.json  (%s) ...", MASK_PATH)
    t0 = time.time()
    with open(MASK_PATH, encoding="utf-8") as f:
        mask_dict = json.load(f)
    logger.info("mask_dict.json loaded in %.1f s.", time.time() - t0)

    # Load model
    logger.info("Loading tokenizer (openbmb/MiniCPM-V-4_5) ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-V-4_5", trust_remote_code=True
    )
    logger.info("Tokenizer loaded in %.1f s.", time.time() - t0)

    logger.info("Loading processor (openbmb/MiniCPM-V-4_5) ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(
        "openbmb/MiniCPM-V-4_5", trust_remote_code=True
    )
    logger.info("Processor loaded in %.1f s.", time.time() - t0)

    logger.info("Loading model weights (openbmb/MiniCPM-V-4_5, bfloat16) ...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-V-4_5",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda:0")
    model.eval()
    logger.info("Model loaded and moved to cuda:0 in %.1f s.", time.time() - t0)

    # Open CSV for appending
    is_new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
    if is_new_file:
        writer.writerow(CSV_COLUMNS)
        csv_file.flush()
        logger.info("Created new CSV with header.")
    else:
        logger.info("Appending to existing CSV.")

    logger.info("-" * 60)
    logger.info("Starting main loop over %d videos ...", len(selected))

    # Pre-compute total number of frames to process (excluding checkpoint)
    total_frames_to_process = 0
    for vid in selected:
        if vid not in meta.get("videos", {}):
            continue
        vmeta = meta["videos"][vid]
        n_frames = len(vmeta["frames"])
        eligible = [
            eid for eid, ex in vmeta["expressions"].items()
            if len(ex["obj_id"]) == 1
        ]
        for eid in eligible:
            already = sum(1 for i in range(n_frames) if (vid, eid, i) in done)
            total_frames_to_process += n_frames - already
    logger.info("Total frames to process (this run): %d", total_frames_to_process)

    run_start = time.time()
    total_processed = 0
    total_failed = 0

    try:
        for video_idx, video_id in enumerate(selected, 1):
            if video_id not in meta.get("videos", {}):
                logger.warning("[%d/%d] Video %s not found in meta_expressions — skipped.",
                               video_idx, len(selected), video_id)
                continue

            video_meta  = meta["videos"][video_id]
            frames      = video_meta["frames"]
            total_frames = len(frames)

            # Count eligible expressions for this video
            eligible_exprs = [
                (eid, ex) for eid, ex in video_meta["expressions"].items()
                if len(ex["obj_id"]) == 1
            ]
            logger.info("[%d/%d] Video %s  |  frames=%d  |  eligible_exprs=%d",
                        video_idx, len(selected), video_id,
                        total_frames, len(eligible_exprs))

            for expr_id, expr in eligible_exprs:
                prompt  = expr["exp"]
                anno_id = str(expr["anno_id"][0])

                already = sum(
                    1 for i in range(total_frames)
                    if (video_id, expr_id, i) in done
                )
                remaining = total_frames - already
                if remaining == 0:
                    logger.debug("  expr %s — all %d frames already in checkpoint, skipping.",
                                 expr_id, total_frames)
                    continue

                logger.info("  expr %s  |  prompt='%.60s'  |  frames=%d  |  remaining=%d",
                            expr_id, prompt, total_frames, remaining)

                expr_ok = 0
                expr_fail = 0

                for i, frame_name in enumerate(frames):
                    if (video_id, expr_id, i) in done:
                        continue

                    # Load image
                    img_path = FM_DIR / args.category / video_id / f"{frame_name}.jpg"
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except Exception as e:
                        logger.warning("    [WARN] Cannot open %s: %s", img_path, e)
                        continue

                    height = image.size[1]   # PIL: (W, H)
                    width  = image.size[0]

                    # Ground truth bbox
                    rle = None
                    if anno_id in mask_dict and i < len(mask_dict[anno_id]):
                        rle = mask_dict[anno_id][i]
                    gt_bbox = extract_gt_bbox(rle)

                    # Inference
                    pred_raw, raw_response, ok, elapsed = run_inference(
                        model, tokenizer, processor, image, prompt
                    )

                    if ok and pred_raw is not None:
                        pred_norm_csv = [pred_raw[0], pred_raw[2], pred_raw[1], pred_raw[3]]
                        pred_px = denormalize(pred_raw, height, width)
                        expr_ok += 1
                    else:
                        pred_norm_csv = [0, 0, 0, 0]
                        pred_px       = [0, 0, 0, 0]
                        expr_fail += 1
                        if "EXCEPTION" in raw_response:
                            logger.error("    frame %d/%d — EXCEPTION: %s",
                                         i, total_frames - 1, raw_response)

                    writer.writerow([
                        video_id,
                        args.category,
                        expr_id,
                        prompt,
                        ok,
                        raw_response,
                        round(elapsed, 4),
                        height,
                        width,
                        total_frames,
                        i,
                        *gt_bbox,
                        *pred_norm_csv,
                        *pred_px,
                    ])
                    csv_file.flush()

                    done.add((video_id, expr_id, i))
                    total_processed += 1
                    if not ok:
                        total_failed += 1

                    # Per-frame debug log (visible in file, not noisy on console)
                    logger.debug("    frame %d/%d  ok=%s  t=%.2fs  pred=%s",
                                 i, total_frames - 1, ok, elapsed, pred_raw)

                    # Progress every 100 frames
                    if total_processed % 100 == 0:
                        elapsed_total = time.time() - run_start
                        fps = total_processed / elapsed_total if elapsed_total > 0 else 0
                        remaining_frames = total_frames_to_process - total_processed
                        eta_s = remaining_frames / fps if fps > 0 else 0
                        eta_h = int(eta_s // 3600)
                        eta_m = int((eta_s % 3600) // 60)
                        logger.info(
                            "  >> Progress: %d/%d  |  remaining=%d  |  %.2f fps  |  "
                            "ETA=%dh%02dm  |  failed=%d  |  elapsed=%.0fs",
                            total_processed, total_frames_to_process,
                            remaining_frames, fps,
                            eta_h, eta_m, total_failed, elapsed_total,
                        )

                logger.info("  expr %s done  |  ok=%d  failed=%d",
                            expr_id, expr_ok, expr_fail)

        elapsed_total = time.time() - run_start
        fps = total_processed / elapsed_total if elapsed_total > 0 else 0
        logger.info("=" * 60)
        logger.info("DONE  |  %s  |  total_frames=%d  |  failed=%d  |  "
                    "elapsed=%.0fs  |  avg_fps=%.2f",
                    run_tag, total_processed, total_failed, elapsed_total, fps)

    finally:
        csv_file.close()
        logger.info("CSV closed.")


if __name__ == "__main__":
    main()
