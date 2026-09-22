"""分区检查（只读、不训练）：把"模型在什么样的目标上弱"拆成可量的几个数。

运行：
    uv run python scripts/07_zone_checks.py

四块（口径、结论与出处见 `docs/baselines.md`「长尾/稀疏度分层 + D 可行性检查」）：
① 目标构成：回访 / 新歌 × 物品稀疏度（目标物品在**训练集**的交互数）
② 冷物品（外推）：训练期没出现过的物品占目标多少、能不能拿到艺人 / 专辑
③ 艺人 / 专辑的共现强度：同实体物品对的用户集合余弦 vs 随机物品对
④ 死区可及性：新歌·稀疏里，用户历史**已含同艺人**的比例

读的是口径 A 的 `artifacts/splits/` 与 `data/raw/` 的原始映射表；口径（回访 / 稀疏度）复用 `rec_eval`，
所以和评测里打印的 `回访×稀疏度` 轴是同一套定义。**这个脚本是 2026-09-22 的检查，用户要求入库留档**
（此前这类检查只在 /tmp，不入库）。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rec_eval import history_flags, item_frequency

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "splits"
RAW = ROOT / "data" / "raw"
SAMPLE = 400  # ③ 采样多少个艺人 / 专辑（seed 固定，可复现）

SPARSITY = ("稀疏(≤5)", "中(6~100)", "热门(>100)")


def load_artist_album(raw_to_dense: dict[int, np.uint32]) -> tuple[np.ndarray, np.ndarray]:
    """把原始 item_id 上的艺人 / 专辑映射，搬到本口径的 id 空间（形状与 item_map 对齐）。"""
    artist = pd.read_parquet(RAW / "artist_item_mapping.parquet")
    album = pd.read_parquet(RAW / "album_item_mapping.parquet")
    n_items = len(raw_to_dense)
    item_artist = np.full(n_items, -1, dtype=np.int64)
    item_album = np.full(n_items, -1, dtype=np.int64)
    for frame, out, column in ((artist, item_artist, "artist_id"), (album, item_album, "album_id")):
        dense = frame.item_id.map(raw_to_dense)
        keep = dense.notna().to_numpy()
        out[dense[keep].to_numpy().astype(np.int64)] = frame[column].to_numpy()[keep]
    return item_artist, item_album


def main() -> None:
    train = pd.read_parquet(SPLITS / "train.parquet", columns=["uid", "item_id"])
    counts = np.bincount(train.item_id.to_numpy(), minlength=int(train.item_id.max()) + 1)
    item_map = pd.read_parquet(SPLITS / "item_map.parquet")
    raw_to_dense = dict(zip(item_map.raw_item_id.to_numpy(), item_map.item_id.to_numpy()))
    item_artist, item_album = load_artist_album(raw_to_dense)
    print(
        f"训练集 {len(train):,} 行 / 物品 {len(counts):,} 个；"
        f"能拿到艺人映射的物品 {int((item_artist >= 0).sum()):,}（{(item_artist >= 0).mean():.2%}）、"
        f"专辑 {int((item_album >= 0).sum()):,}（{(item_album >= 0).mean():.2%}）"
    )

    for split in ("val", "test"):
        target = pd.read_parquet(SPLITS / f"{split}.parquet", columns=["uid", "item_id"])
        uid, item = target.uid.to_numpy(), target.item_id.to_numpy()
        back = history_flags(uid, item, train.uid.to_numpy(), train.item_id.to_numpy())
        frequency = item_frequency(counts, item)
        sparsity = np.where(frequency <= 5, 0, np.where(frequency <= 100, 1, 2))
        row = np.where(back, 0, 1)
        table = np.zeros((2, 3), dtype=np.int64)
        for r, s in zip(row, sparsity):
            table[r, s] += 1
        print(f"\n=== {split}：{len(target):,} 行 ===")
        print("① 目标构成（回访 / 新歌 × 稀疏度）")
        for i, name in enumerate(("回访", "新歌")):
            cells = "  ".join(f"{SPARSITY[j]} {table[i, j]:,}" for j in range(3))
            print(f"   {name}: {cells}（占全部目标 {table[i].sum() / len(target):.1%}）")

        dead = (row == 1) & (sparsity == 0) & (item >= 0)  # 新歌 + 稀疏，排除冷物品
        cold = item < 0
        print("② 冷物品：目标里训练期没出现过的物品")
        print(
            f"   {int(cold.sum()):,} 行（{cold.mean():.2%}）记 -1、必然未命中。"
            "艺人/专辑方面的拆分（拿得到映射的比例、其中多少是真新艺人）需要回到原始 listens 去算，"
            "结果记在 docs/dataset_notes.md「冷物品（外推）」一节"
        )

        rows = np.flatnonzero(dead)
        user_artists = pd.DataFrame({"uid": train.uid.to_numpy(), "a": item_artist[train.item_id.to_numpy()]})
        user_artists = user_artists.groupby("uid").a.apply(set).to_dict()
        reachable = np.array([item_artist[item[i]] in user_artists.get(uid[i], set()) for i in rows])
        print(f"④ 死区（新歌·稀疏，排除冷物品 {len(rows):,} 行）：用户历史里已有同艺人其他歌的占 {reachable.mean():.1%}")

    print("\n③ 艺人 / 专辑的共现强度（二值用户集合的余弦）")
    train_all = pd.read_parquet(SPLITS / "train.parquet", columns=["uid", "item_id"])
    n_items = int(train_all.item_id.max()) + 1
    n_users = int(train_all.uid.max()) + 1
    matrix = sp.csr_matrix(
        (np.ones(len(train_all), dtype=np.float32), (train_all.item_id.to_numpy(), train_all.uid.to_numpy())),
        shape=(n_items, n_users),
    )
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    normalized = sp.diags(1.0 / norm) @ matrix

    rng = np.random.default_rng(0)
    for name, mapping in (("艺人", item_artist), ("专辑", item_album)):
        groups = [np.flatnonzero(mapping == value) for value in np.unique(mapping[mapping >= 0])]
        groups = [g for g in groups if 2 <= len(g) <= 50]
        picks = rng.choice(len(groups), size=min(SAMPLE, len(groups)), replace=False)
        within = []
        for index in picks:
            sub = normalized[groups[index]]
            similarity = (sub @ sub.T).toarray()
            within.append(similarity[np.triu_indices(len(groups[index]), 1)].mean())
        a = rng.integers(0, n_items, 200_000)
        b = rng.integers(0, n_items, 200_000)
        random_pair = float(np.asarray(normalized[a].multiply(normalized[b]).sum(axis=1)).ravel().mean())
        print(
            f"   {name}：可比较 {len(groups):,} 个（2~50 个物品），采样 {len(picks)} 个；"
            f"同{name}物品对 {np.mean(within):.5f} vs 随机物品对 {random_pair:.5f} = {np.mean(within) / random_pair:.0f}×"
        )


if __name__ == "__main__":
    main()
