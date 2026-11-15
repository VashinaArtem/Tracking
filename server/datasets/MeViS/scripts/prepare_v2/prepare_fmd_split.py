#!/usr/bin/env python3
"""
Split FirstMetricsData videos into test and train sets.

Output (saved next to this script):
  fmd_test_videos.json   — 90 video_ids  (held-out test set)
  fmd_train_videos.json  — 191 video_ids (to be added to training)

Selection: seed=42, shuffle of all 281 unique FMD video_ids.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MEVIS_DIR = SCRIPT_DIR.parent.parent
FM_DIR = MEVIS_DIR / "data_sets" / "train" / "FirstMetricsData"
FM_CATEGORIES = ("humans_frequency", "human_object", "human_animal_object")

N_TEST = 90
SEED = 42


def collect_fmd_video_ids() -> list[str]:
    unique: set[str] = set()
    for cat in FM_CATEGORIES:
        cat_dir = FM_DIR / cat
        if cat_dir.exists():
            unique.update(p.name for p in cat_dir.iterdir() if p.is_dir())
    return sorted(unique)


def main() -> None:
    all_vids = collect_fmd_video_ids()
    print(f"Unique FMD video_ids found: {len(all_vids)}")

    rng = random.Random(SEED)
    rng.shuffle(all_vids)

    test_vids  = all_vids[:N_TEST]
    train_vids = all_vids[N_TEST:]

    test_path  = SCRIPT_DIR / "fmd_test_videos.json"
    train_path = SCRIPT_DIR / "fmd_train_videos.json"

    test_path.write_text(json.dumps(test_vids, indent=2, ensure_ascii=False))
    train_path.write_text(json.dumps(train_vids, indent=2, ensure_ascii=False))

    print(f"Test  ({len(test_vids):3d} videos) → {test_path}")
    print(f"Train ({len(train_vids):3d} videos) → {train_path}")


if __name__ == "__main__":
    main()
