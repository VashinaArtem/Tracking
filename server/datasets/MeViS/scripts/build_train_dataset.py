#!/usr/bin/env python3
"""
Build MeViS train dataset for LLaMA-Factory in ShareGPT format.

Output:
  LLaMA-Factory/data/mevis_train.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Generator, List, Set

import numpy as np
from pycocotools import mask as mask_utils


SCRIPT_DIR = Path(__file__).resolve().parent
MEVIS_DIR = SCRIPT_DIR.parent
REPO_ROOT = MEVIS_DIR.parent.parent
DATA_DIR = MEVIS_DIR / "data_sets" / "train"
FM_DIR = DATA_DIR / "FirstMetricsData"
META_PATH = DATA_DIR / "meta_expressions_v2.json"
MASK_PATH = DATA_DIR / "mask_dict.json"
JPEG_DIR = DATA_DIR / "JPEGImages_fequency" / "JPEGImages"
DEFAULT_OUT = REPO_ROOT / "LLaMA-Factory" / "data" / "mevis_train.json"

FM_CATEGORIES = ("humans_frequency", "human_object", "human_animal_object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MeViS train dataset JSON for LLaMA-Factory.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUT),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debug generation.",
    )
    parser.add_argument(
        "--test-videos",
        type=str,
        default=None,
        help=(
            "Path to JSON file containing a list of video_ids to exclude (test set). "
            "If not set, uses get_test_ids() which excludes all FirstMetricsData videos."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_gt_bbox(rle) -> List[int]:
    """Return [x_min, x_max, y_min, y_max] from an RLE mask, or [0,0,0,0] if empty."""
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


def get_test_ids() -> Set[str]:
    test_ids: Set[str] = set()
    for category in FM_CATEGORIES:
        d = FM_DIR / category
        if d.exists():
            test_ids.update(p.name for p in d.iterdir() if p.is_dir())
    return test_ids


def build_user_prompt(expr_text: str) -> str:
    return (
        "<image>"
        "Please provide the bounding box coordinate of the region this sentence describes: "
        f"<ref>{expr_text}</ref>. Return format: <box>x1 y1 x2 y2</box>"
    )


def normalize_bbox(gt_bbox: List[int], height: int, width: int) -> List[int]:
    x1 = max(0, min(1000, round(gt_bbox[0] / width * 1000)))
    y1 = max(0, min(1000, round(gt_bbox[2] / height * 1000)))
    x2 = max(0, min(1000, round(gt_bbox[1] / width * 1000)))
    y2 = max(0, min(1000, round(gt_bbox[3] / height * 1000)))
    return [x1, y1, x2, y2]


def generate_samples(
    meta: Dict,
    mask_dict: Dict,
    available_vids: Set[str],
    test_ids: Set[str],
) -> Generator[Dict, None, None]:
    videos = meta.get("videos", {})
    for video_id, vdata in videos.items():
        if video_id in test_ids:
            continue
        if video_id not in available_vids:
            continue

        frames = vdata.get("frames", [])
        expressions = vdata.get("expressions", {})

        for ex in expressions.values():
            obj_id = ex.get("obj_id", [])
            if not isinstance(obj_id, list) or len(obj_id) != 1:
                continue
            anno_list = ex.get("anno_id")
            if not isinstance(anno_list, list) or not anno_list:
                continue

            anno_id = str(anno_list[0])  # int -> str
            if anno_id not in mask_dict:
                continue

            expr_text = ex.get("exp", "").strip()
            if not expr_text:
                continue

            frame_masks = mask_dict[anno_id]
            for frame_idx, frame_name in enumerate(frames):
                img_path = JPEG_DIR / video_id / f"{frame_name}.jpg"
                if not img_path.exists():
                    continue
                if frame_idx >= len(frame_masks):
                    continue

                rle = frame_masks[frame_idx]
                if rle is None or "size" not in rle or len(rle["size"]) != 2:
                    continue

                height, width = rle["size"][0], rle["size"][1]
                if not isinstance(height, int) or not isinstance(width, int):
                    continue
                if height <= 0 or width <= 0:
                    continue

                gt_bbox = extract_gt_bbox(rle)
                x1, y1, x2, y2 = normalize_bbox(gt_bbox, height, width)
                if x1 > x2 or y1 > y2:
                    continue

                answer = f"<ref>{expr_text}</ref><box>{x1} {y1} {x2} {y2}</box>"
                sample = {
                    "messages": [
                        {"role": "user", "content": build_user_prompt(expr_text)},
                        {"role": "assistant", "content": answer},
                    ],
                    "images": [str(img_path.resolve())],
                }
                yield sample


def main() -> None:
    args = parse_args()

    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta file not found: {META_PATH}")
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Mask file not found: {MASK_PATH}")
    if not JPEG_DIR.exists():
        raise FileNotFoundError(
            f"JPEG directory not found: {JPEG_DIR}. Run extract_train_frames.py first."
        )

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading meta_expressions_v2.json ...")
    meta = load_json(META_PATH)
    print("Loading mask_dict.json ...")
    mask_dict = load_json(MASK_PATH)
    if args.test_videos:
        test_ids = set(json.load(open(args.test_videos, encoding="utf-8")))
    else:
        test_ids = get_test_ids()
    available_vids = {p.name for p in JPEG_DIR.iterdir() if p.is_dir()}

    print(f"Meta videos: {len(meta.get('videos', {}))}")
    print(f"Test videos excluded: {len(test_ids)}")
    print(f"Available extracted videos: {len(available_vids)}")

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        f.write("[\n")
        for sample in generate_samples(meta, mask_dict, available_vids, test_ids):
            if args.max_samples is not None and count >= args.max_samples:
                break
            sep = ",\n" if count > 0 else ""
            f.write(sep + json.dumps(sample, ensure_ascii=False))
            count += 1
            if count % 50000 == 0:
                print(f"Generated samples: {count}")
        f.write("\n]")

    print(f"Saved {count} samples to: {out_path}")


if __name__ == "__main__":
    main()

