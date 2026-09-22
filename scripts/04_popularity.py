"""popularity 基线：自己实现一份，评测口径对齐官方 benchmark。

运行：
    uv run python scripts/04_popularity.py                # 在 val 上扫 hour 网格并评测
    uv run python scripts/04_popularity.py test 1.0       # 在 test 上评测，hour 直接用 val 选出的 1.0
    uv run python scripts/04_popularity.py test 1.0 b     # 版本 B 口径（官方表口径）：无 val，候选池 629,298

口径出处（vendor/yambda-benchmarks/benchmarks/）：
- 打分（models/popularity/main.py::training）：
      score(i) = Σ tau ** (max_ts - ts)，tau = 0.9 ** (1 / 86400 / (hour / 24))；hour=0 即不衰减的计数
- 排序：所有用户共用同一份全局 top-K（官方不过滤已交互物品、不做 per-user 排除）
- 评测（yambda/evaluation/）：口径与实现抽到 scripts/rec_eval.py（与 05_itemknn.py 共用同一份），
      定义与出处见该文件 docstring
- 超参：官方在 val 上扫 hour 网格（默认 7 档），按 val ndcg@100（即命中率@100）选最优

注意：版本 A（默认）的两个 future 窗口：val 用来选 hour，test 用来报告（候选池 627,648）。
排序只由训练集决定（score 只用到 train 的 timestamp），所以换 split 只换评测目标，top-100 不变。
第二个参数是 hour：给了就跳过网格扫描。在 test 上评测时必须传 val 上选出的那一档，
否则等于在 test 上选参。
官方表格里的 test 数字用的是第二遍训练集（val_size=0，train 延伸到 test 前 30 分钟），默认口径 A
不能直接与它比；要并排比就加第三个参数 `b`（口径 B，见 architecture.md「实验流程与两套口径（A / B）」）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rec_eval import evaluate, history_flags, item_frequency, show, split_by_user, target_axes

ROOT = Path(__file__).resolve().parents[1]
SPLITS_ID = sys.argv[3] if len(sys.argv) > 3 else "a"  # 口径：a = 第一遍（有 val），b = 第二遍（val_size = 0）
assert SPLITS_ID in ("a", "b"), f"口径只能是 a 或 b，收到 {SPLITS_ID!r}"
SPLITS = ROOT / "artifacts" / ("splits" if SPLITS_ID == "a" else "splits_b")

DAY = 86_400
HOURS = (0.5, 1.0, 2.0, 3.0, 6.0, 12.0, 24.0)  # 官方 popularity 的网格
SELECT_K = 100  # 官方 --validation_metric 默认 ndcg@100
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"  # 评测目标：val 或 test
HOUR = float(sys.argv[2]) if len(sys.argv) > 2 else None  # 给了就跳过网格扫描
assert not (SPLITS_ID == "b" and SPLIT != "test"), "版本 B 没有 val（val_size = 0），只能评测 test"


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS / f"{name}.parquet")


def item_scores(item: np.ndarray, timestamp: np.ndarray, max_ts: int, n_items: int, hour: float) -> np.ndarray:
    """官方 popularity 的打分（hour=0 为不衰减的计数）。"""
    if hour == 0:
        return np.bincount(item, minlength=n_items).astype(np.float64)
    tau = 0.9 ** (1 / DAY / (hour / 24))
    return np.bincount(item, weights=tau ** (max_ts - timestamp), minlength=n_items)


def rank_all(scores: np.ndarray) -> np.ndarray:
    """全部物品按分数降序（同分时按 item id 升序，保证可重复）。"""
    return np.lexsort((np.arange(scores.shape[0]), -scores))


def main() -> None:
    train = load("train")
    target = load(SPLIT)

    n_items = int(train.item_id.max()) + 1
    n_users = int(train.uid.max()) + 1
    max_ts = int(train.timestamp.max())
    train_uid = train.uid.to_numpy()
    train_item = train.item_id.to_numpy()
    train_ts = train.timestamp.to_numpy()
    print(f"口径 {SPLITS_ID.upper()}；train {len(train):,} 行 / {n_users:,} 用户 / {n_items:,} 候选物品；train max_ts = {max_ts:,}")

    target_users, target_chunks = split_by_user(target.uid.to_numpy(), target.item_id.to_numpy())
    n_unmappable = sum(int((chunk < 0).sum()) for chunk in target_chunks)
    print(
        f"{SPLIT:<5} {len(target):,} 行 / {len(target_users):,} 个有目标的用户"
        f"（其中不可排名的 -1 共 {n_unmappable:,} 行）"
    )

    # 每用户训练期交互过的物品：uid 空间稠密（就是训练集用户），所以「按 uid 分组后的列表」下标就是 uid
    train_history = [np.unique(chunk) for chunk in split_by_user(train_uid, train_item)[1]]
    target_history = [train_history[u] for u in target_users]

    # 分层轴（is_organic / 参与度 / 回访×来源 / 回访×稀疏度）：口径见 rec_eval.py 文件头，列来自 2026-09-22 补进 splits 的三个字段
    back = history_flags(target.uid.to_numpy(), target.item_id.to_numpy(), train_uid, train_item)
    counts = np.bincount(train_item, minlength=n_items)  # 训练集物品频次（回访×稀疏度 轴用）
    frequency = item_frequency(counts, target.item_id.to_numpy())
    axes = target_axes(
        target.uid.to_numpy(), target.is_organic.to_numpy(), target.played_ratio_pct.to_numpy(), back, frequency
    )

    if HOUR is None:
        print("\n[扫描] val 上的 hour 网格（官方验证指标 = ndcg@100，即命中率@100）")
        scan = []
        for hour in HOURS:
            order = rank_all(item_scores(train_item, train_ts, max_ts, n_items, hour))
            all_tops = np.tile(order[:SELECT_K], (n_users, 1))  # 全用户共用同一份全局 top-100
            metrics, _ = evaluate(
                [order[:SELECT_K]] * len(target_chunks),
                target_chunks,
                n_items,
                all_tops=all_tops,
                history=target_history,
                axes=axes,
            )
            scan.append((hour, metrics))
            print(f"  hour={hour:<5} 命中率@100={metrics['hitrate'][SELECT_K]:.6f}  recall@100={metrics['recall'][SELECT_K]:.6f}")
        best_hour, best_metrics = max(scan, key=lambda x: x[1]["hitrate"][SELECT_K])
        print(f"  → 最优 hour = {best_hour}（与官方一致：按命中率@100 选）")
    else:
        best_hour = HOUR
        order = rank_all(item_scores(train_item, train_ts, max_ts, n_items, best_hour))
        all_tops = np.tile(order[:SELECT_K], (n_users, 1))
        best_metrics, _ = evaluate(
            [order[:SELECT_K]] * len(target_chunks),
            target_chunks,
            n_items,
            all_tops=all_tops,
            history=target_history,
            axes=axes,
        )
        print(f"\n[跳过扫描] hour={best_hour}（val 上选出的档，不在 {SPLIT} 上选参）")

    show(f"[{SPLIT}] 官方口径（不过滤已交互物品、目标保留不可排名行）", best_metrics, len(target_users), n_items)

    # 对照 1：过滤掉用户训练期已交互的物品（候选与目标都过滤；官方不做这件事）
    order = rank_all(item_scores(train_item, train_ts, max_ts, n_items, best_hour))
    all_tops = np.tile(order[:SELECT_K], (n_users, 1))
    position = np.empty(n_items, dtype=np.int32)
    position[order] = np.arange(n_items)
    history = [np.unique(chunk) for chunk in split_by_user(train_uid, train_item)[1]]

    filtered_tops, filtered_chunks = [], []
    for uid_value, chunk in zip(target_users, target_chunks):
        seen = history[uid_value]
        blocked = np.sort(position[seen])
        free = np.setdiff1d(np.arange(SELECT_K + blocked.size), blocked, assume_unique=True)
        filtered_tops.append(order[free[:SELECT_K]])
        rankable = chunk[chunk >= 0]
        position_in_history = np.searchsorted(seen, rankable)
        filtered_chunks.append(rankable[seen[np.clip(position_in_history, 0, seen.size - 1)] != rankable])
    metrics_filtered, n_filtered = evaluate(filtered_tops, filtered_chunks, n_items)
    show(f"[{SPLIT}] 对照：过滤已交互物品（候选+目标都去掉历史）", metrics_filtered, n_filtered, n_items)

    # 对照 2：只看可排名的目标（把 -1 行从目标里丢掉）
    rankable_chunks = [chunk[chunk >= 0] for chunk in target_chunks]
    metrics_rankable, n_rankable = evaluate(
        [order[:SELECT_K]] * len(rankable_chunks), rankable_chunks, n_items, all_tops=all_tops
    )
    show(f"[{SPLIT}] 对照：只看可排名目标（丢掉 -1 行）", metrics_rankable, n_rankable, n_items)


if __name__ == "__main__":
    main()
