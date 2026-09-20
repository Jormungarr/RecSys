"""看一眼 yambda 的 listens 数据：它长什么样，每个字段是什么意思。

运行：
    uv run python scripts/01_inspect_listens.py

字段含义的官方出处与推导过程见 docs/dataset_notes.md。
"""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "flat" / "50m" / "listens.parquet"

SECONDS_PER_DAY = 86_400

COLUMNS = {
    "uid": "用户 id（不连续，不是 0..N-1）",
    "item_id": "曲目 id（指单曲，不是专辑/艺人；不连续）",
    "timestamp": "事件时间，单位 = 秒，量化到 5 秒的倍数，起点被官方隐去",
    "is_organic": "0 = 推荐/算法驱动，1 = 用户主动发现",
    "played_ratio_pct": "播放百分比：0~159，100 = 完整听完，>100 = 回放/拖动",
    "track_length_seconds": "曲目总时长（秒），也是 5 秒的倍数",
}


def main() -> None:
    print("file:", DATA)
    metadata = pq.ParquetFile(DATA).metadata
    print(
        "rows:",
        f"{metadata.num_rows:,}",
        "cols:",
        metadata.num_columns,
        "row_groups:",
        metadata.num_row_groups,
        "size:",
        f"{DATA.stat().st_size / 1e6:.1f} MB",
    )

    print("\nschema:")
    for field in pq.read_schema(DATA):
        print(f"  {field.name:<22}{field.type}")

    print("\ncolumn meaning:")
    for name, meaning in COLUMNS.items():
        print(f"  {name:<22}{meaning}")

    frame = pd.read_parquet(DATA, columns=list(COLUMNS))
    print("\nmemory after loading these columns:", f"{frame.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    print("\nfirst 8 rows (raw):")
    print(frame.head(8).to_string(index=False))

    print("\nthe same rows, read as sentences:")
    for _, row in frame.head(3).iterrows():
        listened_seconds = row.track_length_seconds * row.played_ratio_pct / 100
        source = "用户主动" if row.is_organic == 1 else "算法推荐"
        print(
            f"  用户 {row.uid} 在第 {row.timestamp} 秒（第 {row.timestamp / SECONDS_PER_DAY:.3f} 天）"
            f"播放曲目 {row.item_id}（曲长 {row.track_length_seconds} 秒），"
            f"听了 {row.played_ratio_pct}%（约 {listened_seconds:.0f} 秒），来源：{source}"
        )

    user_id = int(frame.uid.iloc[0])
    user = frame[frame.uid == user_id].head(8).copy()
    user["day"] = (user.timestamp / SECONDS_PER_DAY).round(3)
    user["gap_seconds"] = user.timestamp.diff()
    print(f"\nuser {user_id} 的前 8 条记录（文件已按时间排好序，这就是一条行为序列）:")
    print(user[["item_id", "timestamp", "day", "gap_seconds", "played_ratio_pct"]].to_string(index=False))

    ratio = frame.played_ratio_pct
    print("\nbasic counts:")
    print("  events:", f"{len(frame):,}")
    print("  users:", f"{frame.uid.nunique():,}")
    print("  items:", f"{frame.item_id.nunique():,}")
    print(
        "  timestamp range:",
        f"{frame.timestamp.min():,} ~ {frame.timestamp.max():,}",
        f"= 第 0 ~ {frame.timestamp.max() / SECONDS_PER_DAY:.1f} 天",
    )
    print("  played_ratio_pct: min", int(ratio.min()), "median", int(ratio.median()), "max", int(ratio.max()))
    print("  完整听完 (=100%):", f"{(ratio == 100).mean():.1%}")
    print("  Listen+ (>=50%):", f"{(ratio >= 50).mean():.1%}", "  <- 官方定义的正样本")
    print("  organic (is_organic=1):", f"{(frame.is_organic == 1).mean():.1%}")


if __name__ == "__main__":
    main()
