"""
Shared aggregation for mevis_val bbox eval: same logic as ``metrics_finetune.process_csv``,
without matplotlib / file paths from FMD. Input is a DataFrame with columns matching
raw bbox-eval CSV (plus video_name, expr_id, prompt, n_frame, height, width, success).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _bbox_metrics(row: pd.Series) -> tuple[float, float]:
    gt_empty = (
        row["x_min_gt"] == 0
        and row["x_max_gt"] == 0
        and row["y_min_gt"] == 0
        and row["y_max_gt"] == 0
    )
    if gt_empty:
        return np.nan, np.nan

    x1g, x2g = row["x_min_gt"], row["x_max_gt"]
    y1g, y2g = row["y_min_gt"], row["y_max_gt"]
    x1p, x2p = row["x_min_pred_px"], row["x_max_pred_px"]
    y1p, y2p = row["y_min_pred_px"], row["y_max_pred_px"]

    xi1, yi1 = max(x1g, x1p), max(y1g, y1p)
    xi2, yi2 = min(x2g, x2p), min(y2g, y2p)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    gt_area = (x2g - x1g) * (y2g - y1g)
    pred_area = (x2p - x1p) * (y2p - y1p)
    union = gt_area + pred_area - inter
    iou = inter / union if union > 0 else 0.0

    cx1, cy1 = min(x1g, x1p), min(y1g, y1p)
    cx2, cy2 = max(x2g, x2p), max(y2g, y2p)
    c_area = (cx2 - cx1) * (cy2 - cy1)
    giou = iou - (c_area - union) / c_area if c_area > 0 else iou

    return float(iou), float(giou)


def _iou_tg(group: pd.DataFrame) -> float:
    iou_seq = group[group["gt_is_valid"]].sort_values("n_frame")["iou"].dropna()
    if len(iou_seq) < 2:
        return float("nan")
    return float(iou_seq.diff().abs().mean())


# ``frame_idx`` mirrors ``n_frame`` (same as raw MiniCPM bbox-eval CSV ``frame_idx``).
# Order matches ``results_finetune/*_raw.csv`` then pixel preds + aggregate metrics.
PER_FRAME_COLS = [
    "video_name",
    "expr_id",
    "prompt",
    "frame_idx",
    "raw_response",
    "clean_response",
    "success",
    "pred_bbox_raw",
    "height",
    "width",
    "x_min_gt",
    "x_max_gt",
    "y_min_gt",
    "y_max_gt",
    "inference_time",
    "x_min_pred_px",
    "x_max_pred_px",
    "y_min_pred_px",
    "y_max_pred_px",
    "iou",
    "giou",
    "gt_is_valid",
]


def aggregate_bbox_frames_to_run_dir(
    run_dir: Path,
    df: pd.DataFrame,
    run_name: str,
) -> dict[str, Any]:
    """
    Writes ``per_frame.csv``, ``per_expression.csv``, ``per_video.csv``, ``global_summary.txt``
    under ``run_dir``. Mirrors ``metrics_finetune.process_csv`` (lines 80–182).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        summary = {
            "name": run_name,
            "n_frames_total": 0,
            "n_frames_valid": 0,
            "n_videos": 0,
            "n_expressions": 0,
            "mIoU_global": 0.0,
            "mIoU_per_video": 0.0,
            "mIoU_per_expr": 0.0,
            "mean_GIoU": 0.0,
            "acc_50": 0.0,
            "acc_75": 0.0,
            "iou_tg_per_expr": float("nan"),
            "success_rate": 0.0,
        }
        write_global_summary(summary, run_dir)
        return summary

    metrics = df.apply(lambda r: _bbox_metrics(r), axis=1, result_type="expand")
    df = df.copy()
    df["iou"] = metrics[0]
    df["giou"] = metrics[1]
    df["gt_is_valid"] = ~(
        (df["x_min_gt"] == 0)
        & (df["x_max_gt"] == 0)
        & (df["y_min_gt"] == 0)
        & (df["y_max_gt"] == 0)
    )

    if "frame_idx" not in df.columns:
        df["frame_idx"] = df["n_frame"]

    for col in (
        "raw_response",
        "clean_response",
        "pred_bbox_raw",
        "inference_time",
    ):
        if col not in df.columns:
            if col == "inference_time":
                df[col] = 0.0
            else:
                df[col] = ""

    df_frame = df[PER_FRAME_COLS].copy()
    df_frame.to_csv(run_dir / "per_frame.csv", index=False)

    valid = df[df["gt_is_valid"]]

    iou_tg_map = (
        valid.groupby(["video_name", "expr_id"], sort=False)
        .apply(_iou_tg)
        .rename("iou_tg")
        .reset_index()
    )

    def agg_expr(g: pd.DataFrame) -> pd.Series:
        iou = g["iou"]
        giou = g["giou"]
        return pd.Series(
            {
                "prompt": g["prompt"].iloc[0],
                "n_frames_total": len(g),
                "n_frames_valid": g["gt_is_valid"].sum(),
                "mIoU": iou.mean(),
                "mean_GIoU": giou.mean(),
                "acc_50": (iou >= 0.50).mean(),
                "acc_75": (iou >= 0.75).mean(),
                "success_rate": g["success"].mean(),
            }
        )

    df_expr = (
        valid.groupby(["video_name", "expr_id"], sort=False)
        .apply(agg_expr)
        .reset_index()
        .merge(iou_tg_map, on=["video_name", "expr_id"], how="left")
    )
    df_expr.to_csv(run_dir / "per_expression.csv", index=False)

    def agg_video(g: pd.DataFrame) -> pd.Series:
        iou = g["iou"]
        giou = g["giou"]
        return pd.Series(
            {
                "n_expressions": g["expr_id"].nunique(),
                "n_frames_total": len(g),
                "n_frames_valid": g["gt_is_valid"].sum(),
                "mIoU": iou.mean(),
                "mean_GIoU": giou.mean(),
                "acc_50": (iou >= 0.50).mean(),
                "acc_75": (iou >= 0.75).mean(),
                "iou_tg": _iou_tg(g),
                "success_rate": g["success"].mean(),
            }
        )

    df_video = valid.groupby("video_name", sort=False).apply(agg_video).reset_index()
    df_video.to_csv(run_dir / "per_video.csv", index=False)

    iou_valid = valid["iou"]
    giou_valid = valid["giou"]
    summary = {
        "name": run_name,
        "n_frames_total": len(df),
        "n_frames_valid": len(valid),
        "n_videos": df["video_name"].nunique(),
        "n_expressions": df.groupby(["video_name", "expr_id"]).ngroups,
        "mIoU_global": round(float(iou_valid.mean()), 4),
        "mIoU_per_video": round(float(df_video["mIoU"].mean()), 4),
        "mIoU_per_expr": round(float(df_expr["mIoU"].mean()), 4),
        "mean_GIoU": round(float(giou_valid.mean()), 4),
        "acc_50": round(float((iou_valid >= 0.50).mean()), 4),
        "acc_75": round(float((iou_valid >= 0.75).mean()), 4),
        "iou_tg_per_expr": round(float(df_expr["iou_tg"].mean()), 4),
        "success_rate": round(float(df["success"].mean()), 4),
    }
    write_global_summary(summary, run_dir)
    return summary


def write_global_summary(summary: dict[str, Any], output_dir: Path) -> None:
    lines = ["=== Global summary ==="]
    for k, v in summary.items():
        lines.append(f"  {k:<22} {v}")
    text = "\n".join(lines) + "\n"
    (output_dir / "global_summary.txt").write_text(text, encoding="utf-8")
