#!/usr/bin/env python3
"""
Extract MeViS train video frames from JPEGImages.tar.

Train split definition (per plan):
- all videos from meta_expressions_v2.json
- exclude test videos from FirstMetricsData
- keep only videos with at least one single-object expression (len(obj_id) == 1)

Output root:
  data_sets/train/JPEGImages_fequency/JPEGImages/<video_id>/*.jpg
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Set


SCRIPT_DIR = Path(__file__).resolve().parent
MEVIS_DIR = SCRIPT_DIR.parent
DATA_DIR = MEVIS_DIR / "data_sets" / "train"
FM_DIR = DATA_DIR / "FirstMetricsData"
META_PATH = DATA_DIR / "meta_expressions_v2.json"
TAR_PATH = DATA_DIR / "JPEGImages.tar"
JPEG_ROOT = DATA_DIR / "JPEGImages_fequency"
JPEG_DIR = JPEG_ROOT / "JPEGImages"

FM_CATEGORIES = ("humans_frequency", "human_object", "human_animal_object")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract train frames for MeViS.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print extraction stats without writing files.",
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


def load_meta(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_test_ids() -> Set[str]:
    test_ids: Set[str] = set()
    for category in FM_CATEGORIES:
        d = FM_DIR / category
        if d.exists():
            test_ids.update(p.name for p in d.iterdir() if p.is_dir())
    return test_ids


def is_single_object_video(video_data: Dict) -> bool:
    expressions = video_data.get("expressions", {})
    for ex in expressions.values():
        obj_id = ex.get("obj_id", [])
        if isinstance(obj_id, list) and len(obj_id) == 1:
            return True
    return False


def get_train_video_ids(meta: Dict, test_ids: Set[str]) -> Set[str]:
    train_ids: Set[str] = set()
    videos = meta.get("videos", {})
    for vid_id, vdata in videos.items():
        if vid_id in test_ids:
            continue
        if is_single_object_video(vdata):
            train_ids.add(vid_id)
    return train_ids


def collect_members_to_extract(
    tar: tarfile.TarFile,
    train_ids: Set[str],
    already_extracted: Set[str],
) -> Iterable[tarfile.TarInfo]:
    for member in tar.getmembers():
        parts = Path(member.name).parts
        # Expected: JPEGImages/<video_id>/<frame>.jpg
        if len(parts) < 2:
            continue
        if parts[0] != "JPEGImages":
            continue
        video_id = parts[1]
        if video_id not in train_ids:
            continue
        if video_id in already_extracted:
            continue
        yield member


def main() -> None:
    args = parse_args()

    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta file not found: {META_PATH}")
    if not TAR_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {TAR_PATH}")

    JPEG_DIR.mkdir(parents=True, exist_ok=True)
    already_extracted = {p.name for p in JPEG_DIR.iterdir() if p.is_dir()}

    meta = load_meta(META_PATH)
    if args.test_videos:
        test_ids = set(json.loads(Path(args.test_videos).read_text(encoding="utf-8")))
    else:
        test_ids = get_test_ids()
    train_ids = get_train_video_ids(meta, test_ids)
    pending_ids = train_ids - already_extracted

    print(f"Total videos in meta: {len(meta.get('videos', {}))}")
    print(f"Test videos (excluded): {len(test_ids)}")
    print(f"Train videos (single-obj only): {len(train_ids)}")
    print(f"Already extracted: {len(train_ids & already_extracted)}")
    print(f"Pending extraction: {len(pending_ids)}")

    if not pending_ids:
        print("Nothing to extract. Done.")
        return

    with tarfile.open(TAR_PATH) as tar:
        to_extract = list(collect_members_to_extract(tar, train_ids, already_extracted))
        print(f"Archive members to extract: {len(to_extract)}")
        if args.dry_run:
            print("Dry-run mode enabled. Exiting without extraction.")
            return

        try:
            tar.extractall(path=JPEG_ROOT, members=to_extract, filter="data")
        except TypeError:
            # Fallback for Python versions where `filter` is unsupported.
            tar.extractall(path=JPEG_ROOT, members=to_extract)

    after_extracted = {p.name for p in JPEG_DIR.iterdir() if p.is_dir()}
    print(f"Extracted train videos now present: {len(train_ids & after_extracted)} / {len(train_ids)}")
    print("Done.")


if __name__ == "__main__":
    main()

