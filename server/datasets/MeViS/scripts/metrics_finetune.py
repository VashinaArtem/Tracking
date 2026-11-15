from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from PIL import Image

RESULTS_DIR = Path("/home/jovyan/shares/SR006.nfs2/vashina/datasets/MeViS/results_finetune")
METRICS_DIR = RESULTS_DIR / "metrics"
MEVIS_DIR = Path("/home/jovyan/shares/SR006.nfs2/vashina/datasets/MeViS")
FM_DIR = MEVIS_DIR / "data_sets/train/FirstMetricsData"
META_PATH = MEVIS_DIR / "data_sets/train/meta_expressions_v2.json"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "metrics_finetune"

ALL_CSVS = [
    "humans_frequency_part0",
    "humans_frequency_part1",
    "human_object_part0",
    "human_object_part1",
    "human_animal_object_part0",
    "human_animal_object_part1",
]


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

    return iou, giou


def _iou_tg(group: pd.DataFrame) -> float:
    iou_seq = group[group["gt_is_valid"]].sort_values("n_frame")["iou"].dropna()
    if len(iou_seq) < 2:
        return np.nan
    return iou_seq.diff().abs().mean()


def process_csv(name: str) -> dict:
    csv_path = RESULTS_DIR / f"{name}.csv"
    out_dir = METRICS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    print(f"  Rows: {len(df):,}")

    metrics = df.apply(lambda r: _bbox_metrics(r), axis=1, result_type="expand")
    df["iou"] = metrics[0]
    df["giou"] = metrics[1]
    df["gt_is_valid"] = ~(
        (df["x_min_gt"] == 0)
        & (df["x_max_gt"] == 0)
        & (df["y_min_gt"] == 0)
        & (df["y_max_gt"] == 0)
    )

    per_frame_cols = [
        "video_name",
        "expr_id",
        "prompt",
        "n_frame",
        "height",
        "width",
        "x_min_gt",
        "x_max_gt",
        "y_min_gt",
        "y_max_gt",
        "x_min_pred_px",
        "x_max_pred_px",
        "y_min_pred_px",
        "y_max_pred_px",
        "success",
        "iou",
        "giou",
        "gt_is_valid",
    ]
    df_frame = df[per_frame_cols].copy()
    df_frame.to_csv(out_dir / "per_frame.csv", index=False)
    print(f"  Saved per_frame.csv  ({len(df_frame):,} rows)")

    valid = df[df["gt_is_valid"]]

    iou_tg_map = (
        valid.groupby(["video_name", "expr_id"]).apply(_iou_tg).rename("iou_tg").reset_index()
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
    df_expr.to_csv(out_dir / "per_expression.csv", index=False)
    print(f"  Saved per_expression.csv  ({len(df_expr):,} rows)")

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
    df_video.to_csv(out_dir / "per_video.csv", index=False)
    print(f"  Saved per_video.csv  ({len(df_video):,} rows)")

    iou_valid = valid["iou"]
    giou_valid = valid["giou"]
    summary = {
        "name": name,
        "n_frames_total": len(df),
        "n_frames_valid": len(valid),
        "n_videos": df["video_name"].nunique(),
        "n_expressions": df.groupby(["video_name", "expr_id"]).ngroups,
        "mIoU_global": round(iou_valid.mean(), 4),
        "mIoU_per_video": round(df_video["mIoU"].mean(), 4),
        "mIoU_per_expr": round(df_expr["mIoU"].mean(), 4),
        "mean_GIoU": round(giou_valid.mean(), 4),
        "acc_50": round((iou_valid >= 0.50).mean(), 4),
        "acc_75": round((iou_valid >= 0.75).mean(), 4),
        "iou_tg_per_expr": round(df_expr["iou_tg"].mean(), 4),
        "success_rate": round(df["success"].mean(), 4),
    }
    return summary


def write_global_summary(summary: dict, output_dir: Path) -> None:
    lines = ["=== Global summary ==="]
    for k, v in summary.items():
        lines.append(f"  {k:<22} {v}")
    text = "\n".join(lines) + "\n"
    print("\n" + text.strip())
    (output_dir / "global_summary.txt").write_text(text, encoding="utf-8")


def plot_single_part(name: str, output_dir: Path) -> None:
    df_expr = pd.read_csv(METRICS_DIR / name / "per_expression.csv")
    df_video = pd.read_csv(METRICS_DIR / name / "per_video.csv")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, data, label, color in [
        (axes[0, 0], df_expr["mIoU"], "per expression", "steelblue"),
        (axes[0, 1], df_video["mIoU"], "per video", "darkorange"),
    ]:
        d = data.dropna()
        ax.hist(d, bins=40, color=color, edgecolor="white")
        ax.axvline(d.mean(), color="red", linestyle="--", label=f"mean={d.mean():.3f}")
        ax.set_title(f"mIoU distribution - {label}")
        ax.set_xlabel("mIoU")
        ax.set_ylabel("count")
        ax.legend()

    giou_data = df_expr["mean_GIoU"].dropna()
    axes[1, 0].hist(giou_data, bins=40, color="mediumpurple", edgecolor="white")
    axes[1, 0].axvline(
        giou_data.mean(), color="red", linestyle="--", label=f"mean={giou_data.mean():.3f}"
    )
    axes[1, 0].axvline(0, color="gray", linestyle=":", linewidth=1, label="GIoU=0")
    axes[1, 0].set_title("mean GIoU distribution - per expression")
    axes[1, 0].set_xlabel("mean GIoU (negative = non-overlapping)")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].legend()

    tg_data = df_expr["iou_tg"].dropna()
    axes[1, 1].hist(tg_data, bins=40, color="tomato", edgecolor="white")
    axes[1, 1].axvline(
        tg_data.mean(), color="darkred", linestyle="--", label=f"mean={tg_data.mean():.3f}"
    )
    axes[1, 1].set_title("IoU-TG distribution - per expression")
    axes[1, 1].set_xlabel("IoU-TG (lower = more stable)")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].legend()

    plt.suptitle(name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "plots" / f"{name}_distributions.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for col in ["mIoU", "mean_GIoU", "acc_50", "acc_75", "iou_tg", "success_rate"]:
        e = df_expr[col].mean() if col in df_expr else np.nan
        v = df_video[col].mean() if col in df_video else np.nan
        rows.append({"metric": col, "per_expr_mean": e, "per_video_mean": v})
    metric_table = pd.DataFrame(rows)
    metric_table.to_csv(output_dir / "tables" / f"{name}_metric_summary.csv", index=False)


def summary_across_all_parts(output_dir: Path) -> pd.DataFrame:
    all_summaries = []
    for name in ALL_CSVS:
        csv_path = RESULTS_DIR / f"{name}.csv"
        if not csv_path.exists():
            print(f"SKIP {name} - CSV not found")
            continue
        s = process_csv(name)
        all_summaries.append(s)
        print()

    df_summary = pd.DataFrame(all_summaries)
    print("\n=== Summary across all parts ===")
    print(df_summary.to_string(index=False))
    df_summary.to_csv(output_dir / "summary_across_all_parts.csv", index=False)
    (output_dir / "summary_across_all_parts.txt").write_text(
        "=== Summary across all parts ===\n"
        + df_summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    return df_summary


def plot_all_parts(output_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ALL_CSVS:
        path = METRICS_DIR / name / "per_expression.csv"
        if path.exists():
            df_tmp = pd.read_csv(path)
            df_tmp["source"] = name
            frames.append(df_tmp)

    if not frames:
        raise RuntimeError("No per_expression.csv found - run process_csv first.")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df_all):,} expressions from {len(frames)} file(s):")
    for f in frames:
        src = f["source"].iloc[0]
        print(f"  {src:<40}  {len(f):>4} expressions")

    metrics = [
        ("mIoU", "mIoU", "steelblue", (0, 1), True),
        ("mean_GIoU", "mean GIoU", "mediumpurple", (-1, 1), True),
        ("acc_50", "Acc@0.5", "seagreen", (0, 1), True),
        ("acc_75", "Acc@0.75", "darkcyan", (0, 1), True),
        ("iou_tg", "IoU-TG", "tomato", None, False),
        ("success_rate", "Success rate", "goldenrod", (0, 1), True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (col, label, color, xlim, add_vline_zero) in zip(axes, metrics):
        data = df_all[col].dropna()
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(
            mean_val, color="red", linestyle="--", linewidth=1.4, label=f"mean={mean_val:.3f}"
        )
        ax.axvline(
            median_val,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label=f"median={median_val:.3f}",
        )
        if add_vline_zero and col == "mean_GIoU":
            ax.axvline(0, color="gray", linestyle="-", linewidth=0.8, label="0")

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("expressions")
        if xlim:
            ax.set_xlim(*xlim)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    n_sources = len(frames)
    src_label = ", ".join(f["source"].iloc[0] for f in frames)
    fig.suptitle(
        f"Metric distributions - {n_sources} part(s): {src_label}\n"
        f"({len(df_all):,} expressions total)",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = output_dir / "plots" / "all_parts_distributions.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for col, label, *_ in metrics:
        d = df_all[col].dropna()
        rows.append(
            {
                "metric": label,
                "mean": d.mean(),
                "median": d.median(),
                "std": d.std(),
                "min": d.min(),
                "max": d.max(),
            }
        )
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(output_dir / "tables" / "all_parts_metric_stats.csv", index=False)
    return stats_df


def iou_single(row: pd.Series) -> float:
    gt_empty = (
        row.x_min_gt == 0 and row.x_max_gt == 0 and row.y_min_gt == 0 and row.y_max_gt == 0
    )
    if gt_empty:
        return float("nan")
    if not row.success:
        return 0.0
    xi1 = max(row.x_min_gt, row.x_min_pred_px)
    yi1 = max(row.y_min_gt, row.y_min_pred_px)
    xi2 = min(row.x_max_gt, row.x_max_pred_px)
    yi2 = min(row.y_max_gt, row.y_max_pred_px)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    gt_a = (row.x_max_gt - row.x_min_gt) * (row.y_max_gt - row.y_min_gt)
    pred_a = (row.x_max_pred_px - row.x_min_pred_px) * (row.y_max_pred_px - row.y_min_pred_px)
    union = gt_a + pred_a - inter
    return inter / union if union > 0 else 0.0


def save_failed_examples(csv_name: str, output_dir: Path, window: int = 2, limit: int = 50) -> None:
    if not META_PATH.exists():
        print(f"Skip failed examples: meta file not found at {META_PATH}")
        return
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    df_raw = pd.read_csv(RESULTS_DIR / f"{csv_name}.csv")
    failed_rows = df_raw[df_raw["success"] == False]
    print(f"Failed rows: {len(failed_rows)}")

    failed_dir = output_dir / "failed_examples" / csv_name
    failed_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, failed) in enumerate(failed_rows.iterrows()):
        if i >= limit:
            break

        vid = failed["video_name"]
        eid = int(failed["expr_id"])
        fail_idx = int(failed["n_frame"])
        cat = failed["category"]
        prompt = failed["prompt"]
        frames = meta["videos"][vid]["frames"]

        window_indices = range(max(0, fail_idx - window), min(len(frames), fail_idx + window + 1))
        n_cols = len(list(window_indices))
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 3.8, 4.2))
        if n_cols == 1:
            axes = [axes]

        for ax, frame_idx in zip(axes, window_indices):
            frame_name = frames[frame_idx]
            img_path = FM_DIR / cat / vid / f"{frame_name}.jpg"
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                ax.set_title(f"f{frame_idx}\nNO IMAGE")
                ax.axis("off")
                continue

            ax.imshow(img)
            ax.axis("off")
            rows = df_raw[
                (df_raw["video_name"] == vid)
                & (df_raw["expr_id"] == eid)
                & (df_raw["n_frame"] == frame_idx)
            ]
            if rows.empty:
                ax.set_title(f"f{frame_idx}\nno data")
                continue

            row = rows.iloc[0]
            iou = iou_single(row)
            gt_empty = (
                row.x_min_gt == 0 and row.x_max_gt == 0 and row.y_min_gt == 0 and row.y_max_gt == 0
            )
            if not gt_empty:
                ax.add_patch(
                    patches.Rectangle(
                        (row.x_min_gt, row.y_min_gt),
                        row.x_max_gt - row.x_min_gt,
                        row.y_max_gt - row.y_min_gt,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="none",
                    )
                )
            if row.success:
                ax.add_patch(
                    patches.Rectangle(
                        (row.x_min_pred_px, row.y_min_pred_px),
                        row.x_max_pred_px - row.x_min_pred_px,
                        row.y_max_pred_px - row.y_min_pred_px,
                        linewidth=2,
                        edgecolor="lime",
                        facecolor="none",
                    )
                )

            is_fail = frame_idx == fail_idx
            iou_str = f"{iou:.3f}" if not np.isnan(iou) else "GT empty"
            status = "X NO BBOX" if not row.success else "OK"
            color = "red" if is_fail else "black"
            ax.set_title(
                f"frame {frame_idx}  {status}\nIoU={iou_str}",
                fontsize=8,
                color=color,
                fontweight="bold" if is_fail else "normal",
            )

        fig.suptitle(
            f"video={vid} | expr_id={eid} | failed frame={fail_idx}\n"
            f'"{prompt}"\n'
            f"red=GT green=Pred X=no bbox predicted",
            fontsize=9,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(failed_dir / f"failed_{i:04d}_{vid}_expr{eid}_frame{fail_idx}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "failed_examples").mkdir(parents=True, exist_ok=True)

    target_name = "humans_frequency_part0"
    summary = process_csv(target_name)
    write_global_summary(summary, OUTPUT_DIR)
    plot_single_part(target_name, OUTPUT_DIR)

    summary_across_all_parts(OUTPUT_DIR)
    all_stats = plot_all_parts(OUTPUT_DIR)
    all_stats.to_csv(OUTPUT_DIR / "tables" / "all_parts_metric_stats.csv", index=False)

    save_failed_examples(target_name, OUTPUT_DIR, window=2, limit=50)
    print(f"\nArtifacts are saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
