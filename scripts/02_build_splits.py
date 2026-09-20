"""把 listens 变成我们自己的 train/val/test。

运行：
    uv run python scripts/02_build_splits.py

口径全部对齐官方 benchmark 代码（vendor/yambda-benchmarks/benchmarks/）：
- 正样本：`played_ratio_pct >= 50`（`constants.TRACK_LISTEN_THRESHOLD`）
- 时间切分：`processing/timesplit.py::flat_split_train_val_test`（val_size = VAL_SIZE）
- 只保留训练集出现过的 uid；训练集没出现过的物品仍留在 val/test（官方默认 drop_non_train_items=False）

验收数字见 docs/benchmark_repro.md 的"一致性核对"。
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "flat" / "50m" / "listens.parquet"
OUT = ROOT / "artifacts" / "splits"

# 官方常量（秒），出处 vendor/yambda-benchmarks/benchmarks/yambda/constants.py
HOUR = 60 * 60
DAY = 24 * HOUR
GAP_SIZE = HOUR // 2  # 1,800
VAL_SIZE = DAY  # 86,400
TEST_TIMESTAMP = 26_000_000 - DAY  # 25,913,600

LISTEN_THRESHOLD = 50

TRAIN_END = TEST_TIMESTAMP - GAP_SIZE - VAL_SIZE - GAP_SIZE  # 25,823,600
VAL_START = TEST_TIMESTAMP - VAL_SIZE - GAP_SIZE  # 25,825,400
VAL_END = TEST_TIMESTAMP - GAP_SIZE  # 25,911,800

COLUMNS = ["uid", "item_id", "timestamp", "played_ratio_pct"]


def is_sorted(uid: np.ndarray, timestamp: np.ndarray) -> bool:
    """(uid, timestamp) 升序。"""
    return bool(np.all((uid[1:] > uid[:-1]) | ((uid[1:] == uid[:-1]) & (timestamp[1:] >= timestamp[:-1]))))


def densify(raw: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """原始 id -> 0..N-1；不在 universe 里的记为 -1。"""
    pos = np.searchsorted(universe, raw)
    pos_clipped = np.clip(pos, 0, len(universe) - 1)
    return np.where(universe[pos_clipped] == raw, pos_clipped, -1).astype("int32")


def main() -> None:
    frame = pd.read_parquet(SRC, columns=COLUMNS)
    print(f"raw listens            : {len(frame):,} 行")

    assert is_sorted(frame.uid.to_numpy(), frame.timestamp.to_numpy()), "输入应已按 (uid, timestamp) 升序"
    print("input sorted           : yes（(uid, timestamp) 升序，管线依赖这一点）")

    listen_plus = frame[frame.played_ratio_pct >= LISTEN_THRESHOLD]
    print(f"Listen+                : {len(listen_plus):,} 行（{len(listen_plus) / len(frame):.2%}）")

    timestamp = listen_plus.timestamp
    train_raw = listen_plus[timestamp < TRAIN_END]
    val_raw = listen_plus[(timestamp >= VAL_START) & (timestamp < VAL_END)]
    test_raw = listen_plus[timestamp >= TEST_TIMESTAMP]
    gaps = len(listen_plus) - len(train_raw) - len(val_raw) - len(test_raw)
    print(f"切分（留 uid 之前）      : train {len(train_raw):,} / val {len(val_raw):,} / test {len(test_raw):,}（两个 gap 共 {gaps:,} 行）")

    train_uids = np.sort(train_raw.uid.unique())
    train_items = np.sort(train_raw.item_id.unique())
    val = val_raw[np.isin(val_raw.uid.to_numpy(), train_uids)]
    test = test_raw[np.isin(test_raw.uid.to_numpy(), train_uids)]
    print(f"切分（只留训练集用户）    : val {len(val):,} / test {len(test):,}")

    parts = {}
    for name, part in (("train", train_raw), ("val", val), ("test", test)):
        dense = pd.DataFrame(
            {
                "uid": densify(part.uid.to_numpy(), train_uids),
                "item_id": densify(part.item_id.to_numpy(), train_items),
                "timestamp": part.timestamp.to_numpy().astype("int32"),
            }
        )
        assert is_sorted(dense.uid.to_numpy(), dense.timestamp.to_numpy()), f"{name} 应保持 (uid, timestamp) 升序"
        parts[name] = dense

    print(f"重编号后 id 空间         : 用户 {len(train_uids):,}（0..{len(train_uids) - 1}）、物品 {len(train_items):,}（0..{len(train_items) - 1}）")
    for name in ("val", "test"):
        unseen = int((parts[name].item_id == -1).sum())
        print(f"{name} 目标               : 用户 {parts[name].uid.nunique():,}，行 {len(parts[name]):,}，其中物品不在训练集（记为 -1）{unseen:,} 行")

    OUT.mkdir(parents=True, exist_ok=True)
    for name, dense in parts.items():
        path = OUT / f"{name}.parquet"
        dense.to_parquet(path, index=False)
        print(f"写出                   : {path.relative_to(ROOT)}（{path.stat().st_size / 1e6:.1f} MB）")
    uid_map = pd.DataFrame({"raw_uid": train_uids, "uid": np.arange(len(train_uids), dtype="int32")})
    item_map = pd.DataFrame({"raw_item_id": train_items, "item_id": np.arange(len(train_items), dtype="int32")})
    uid_map.to_parquet(OUT / "uid_map.parquet", index=False)
    item_map.to_parquet(OUT / "item_map.parquet", index=False)
    print(f"写出                   : {OUT.relative_to(ROOT)}/uid_map.parquet、item_map.parquet")


if __name__ == "__main__":
    main()
