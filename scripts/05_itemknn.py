"""itemknn 基线：自己实现一份，评测口径对齐官方 benchmark。

运行：
    uv run python scripts/05_itemknn.py                  # 在 val 上扫 13 档 hour 网格并评测（约 16 分钟）
    uv run python scripts/05_itemknn.py test 0.5         # 在 test 上评测，hour 直接用 val 选出的 0.5（约 1.5 分钟）
    uv run python scripts/05_itemknn.py test 0.5 b       # 版本 B 口径（官方表口径）：无 val，候选池 629,298

口径出处（vendor/yambda-benchmarks/benchmarks/models/itemknn/main.py）：
- C = 训练集「用户×物品」计数矩阵：同一 (uid, item) 的多次交互相加
- 物品表示：item_emb[i, v] = C[v, i] / ||C[v, :]||_2         （sparse_normalize(user_item.T, dim=-1)）
- 用户表示：A = W @ C.T，权重 tau ** Δ，Δ = 该用户的最后一次 - 该次时间，
            tau = 0.9 ** (1 / 86400 / (hour / 24))；hour=0 → tau=0，只剩「每个用户最后一次事件」那些对；
            官方随后对 A 按行 L2 归一 —— 逐行除以一个正常数不改变该行排序，而指标只看 top-k，故省略；
            W 的支撑比 C 小：tau ** Δ 会下溢，官方用 eliminate_zeros(1e-9) 把那些对清掉
- 打分：score(u, i) = Σ_v A[u, v] · C[v, i] / ||C[v, :]||_2，取全局 top-100（不过滤已交互物品）
- 超参：官方在 val 上扫 13 档 hour 网格，按 val ndcg@100（即命中率@100，官方 bug）选最优
- 评测：与 04_popularity.py 共用 scripts/rec_eval.py

与官方实现的两处已知差异（只可能影响并列项的先后，不影响分数）：
- 官方在 torch 里 float32 逐批累加，求和顺序与这里不同 → 分数末位可能有别
- 并列时官方 torch.topk 的取舍未定义；这里固定按 item id 升序

实现取舍：A = W @ C.T 用「按物品累加外积」而不是稀疏×稀疏乘法 —— 后者是逐元素哈希累加，在本机慢
一到两个数量级，而外积走 BLAS。外积覆盖 W 的行集 × C 的**整列**（两者支撑不同：C(v, i) 与
W(u, i) 不必同时非零），大物品按块组合、避免整列外积撑爆内存。

注意：版本 A（默认）的两个 future 窗口：val 用来选 hour，test 用来报告（候选池 627,648）。
排序只由训练集决定（C 与 W 都只从 train 算），所以换 split 只换评测目标，top-100 不变。
第二个参数是 hour：给了就跳过 13 档网格扫描。在 test 上评测时必须传 val 上选出的那一档，
否则等于在 test 上选参。
官方表格里的 test 数字用的是第二遍训练集（val_size=0），默认口径 A 不能直接与它比；
要并排比就加第三个参数 `b`（口径 B，见 architecture.md「实验流程与两套口径（A / B）」）。
"""

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from rec_eval import evaluate, show, split_by_user

ROOT = Path(__file__).resolve().parents[1]
SPLITS_ID = sys.argv[3] if len(sys.argv) > 3 else "a"  # 口径：a = 第一遍（有 val），b = 第二遍（val_size = 0）
assert SPLITS_ID in ("a", "b"), f"口径只能是 a 或 b，收到 {SPLITS_ID!r}"
SPLITS = ROOT / "artifacts" / ("splits" if SPLITS_ID == "a" else "splits_b")

DAY = 86_400
HOURS = (0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.5, 1.0, 2.0)  # 官方 itemknn 的网格
SELECT_K = 100  # 官方 --validation_metric 默认 ndcg@100
ZERO_EPS = 1e-9  # 官方 eliminate_zeros 的阈值
BLOCK = 512  # 外积分块的边长（要么两边都 ≤ BLOCK 直接算，要么按 BLOCK 切块两两组合）
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"  # 评测目标：val 或 test
HOUR = float(sys.argv[2]) if len(sys.argv) > 2 else None  # 给了就跳过网格扫描
assert not (SPLITS_ID == "b" and SPLIT != "test"), "版本 B 没有 val（val_size = 0），只能评测 test"


@dataclasses.dataclass
class Pairs:
    """训练集压成 (uid, item) 对，按 CSC 规范序（主键 item、次键 uid）排好。"""

    uid: np.ndarray  # 每个对的用户，长度 = 对数
    item: np.ndarray  # 每个对的物品，升序
    count: np.ndarray  # 每个对的交互次数
    starts: np.ndarray  # delta 里每个对的起点
    delta: np.ndarray  # 每次交互的 Δ = 该用户的最后一次 - 该次时间，按对分组、组内按时间


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS / f"{name}.parquet")


def user_max_timestamp(train: pd.DataFrame, n_users: int) -> np.ndarray:
    """每个用户自己的最后一次交互时间（官方 pl.max("timestamp").over("uid")）。"""
    per_user = train.groupby("uid")["timestamp"].max().to_numpy()
    assert len(per_user) == n_users, "uid 应是 0..n_users-1 的连续编号"
    return per_user


def build_pairs(train: pd.DataFrame, n_users: int) -> Pairs:
    uid = train.uid.to_numpy().astype(np.int32)
    item = train.item_id.to_numpy().astype(np.int32)
    delta = (user_max_timestamp(train, n_users)[uid] - train.timestamp.to_numpy()).astype(np.int32)

    order = np.lexsort((uid, item))
    uid, item, delta = uid[order], item[order], delta[order]

    starts = np.flatnonzero(np.r_[True, (item[1:] != item[:-1]) | (uid[1:] != uid[:-1])])
    return Pairs(
        uid=uid[starts],
        item=item[starts],
        count=np.diff(np.r_[starts, len(item)]).astype(np.float32),
        starts=starts,
        delta=delta,
    )


def column_matrix(
    rows: np.ndarray, items: np.ndarray, values: np.ndarray, n_users: int, n_items: int
) -> sp.csc_matrix:
    """按物品分列、列内按 uid 升序的 CSC 矩阵（rows/items 需已是 CSC 规范序）。"""
    indptr = np.r_[0, np.cumsum(np.bincount(items, minlength=n_items))]
    return sp.csc_matrix((values, rows, indptr), shape=(n_users, n_items))


def hour_weights(pairs: Pairs, hour: float) -> np.ndarray:
    """官方 create_weighted_sparse_tensor：tau ** Δ 按 (uid, item) 对求和。"""
    if hour == 0:  # 官方 tau = 0.0，0.0 ** 0 = 1、0.0 ** 正数 = 0
        weights = (pairs.delta == 0).astype(np.float64)
    else:
        tau = 0.9 ** (1 / DAY / (hour / 24))
        weights = tau ** pairs.delta
    return np.add.reduceat(weights, pairs.starts).astype(np.float32)


def user_user_matrix(C: sp.csc_matrix, W: sp.csc_matrix, n_users: int) -> np.ndarray:
    """A = W @ C.T：对每个物品，把外积 w ⊗ c 累加进 A。

    W 与 C 同列但支撑不同，且求和要对 C 的整列做，所以是「W 的行集 × C 的行集」的矩形外积；
    两边都不超过 BLOCK 时直接算，否则按 BLOCK 切块两两组合（含非对角块）。"""
    A = np.zeros((n_users, n_users), dtype=np.float32)
    for item in range(C.shape[1]):
        ws, we = int(W.indptr[item]), int(W.indptr[item + 1])
        if ws == we:  # W 在这一列没有非零（权重下溢），对 A 无贡献
            continue
        cs, ce = int(C.indptr[item]), int(C.indptr[item + 1])
        if we - ws <= BLOCK and ce - cs <= BLOCK:
            A[np.ix_(W.indices[ws:we], C.indices[cs:ce])] += np.outer(W.data[ws:we], C.data[cs:ce])
            continue
        for a in range(ws, we, BLOCK):
            a_stop = min(a + BLOCK, we)
            for b in range(cs, ce, BLOCK):
                b_stop = min(b + BLOCK, ce)
                A[np.ix_(W.indices[a:a_stop], C.indices[b:b_stop])] += np.outer(
                    W.data[a:a_stop], C.data[b:b_stop]
                )
    return A


def top_k(A: np.ndarray, Cn_t: sp.csr_matrix, batch: int = 128) -> np.ndarray:
    """用户表示 × 物品表示，取每个用户的 top-k（同分按 item id 升序）。"""
    n_users = A.shape[0]
    tops = np.empty((n_users, SELECT_K), dtype=np.int32)
    for s in range(0, n_users, batch):
        e = min(s + batch, n_users)
        scores = (Cn_t @ A[s:e, :].T).T  # (batch, n_items)，稠密 × 稀疏
        n = e - s
        part = np.argpartition(-scores, SELECT_K, axis=1)[:, :SELECT_K]
        rows = np.arange(n)[:, None]
        tops[s:e] = part[rows, np.lexsort((part, -scores[rows, part]), axis=1)]
    return tops


def main() -> None:
    train = load("train")
    target = load(SPLIT)

    n_users = int(train.uid.max()) + 1
    n_items = int(train.item_id.max()) + 1
    print(f"口径 {SPLITS_ID.upper()}；train {len(train):,} 行 / {n_users:,} 用户 / {n_items:,} 候选物品")

    pairs = build_pairs(train, n_users)
    print(f"(uid,item) 对        : {len(pairs.uid):,}")

    C = column_matrix(pairs.uid, pairs.item, pairs.count, n_users, n_items)
    # 物品表示（官方 sparse_normalize(user_item.T, dim=-1)）：每个非零除以「它所属用户」的 L2 范数。
    # C 是 CSC，indices 就是行号（uid）。
    scale = np.sqrt(np.bincount(C.indices, weights=C.data.astype(np.float64) ** 2, minlength=n_users) + 1e-12)
    Cn = C.copy()
    Cn.data = Cn.data / scale[Cn.indices]
    Cn_t = Cn.tocsr().T.tocsr()

    target_users, target_chunks = split_by_user(target.uid.to_numpy(), target.item_id.to_numpy())
    print(f"{SPLIT:<5} {len(target):,} 行 / {len(target_users):,} 个有目标的用户")

    # 每用户训练期交互过的物品（uid 空间稠密，所以「按 uid 分组后的列表」下标就是 uid）
    train_history = [np.unique(chunk) for chunk in split_by_user(train.uid.to_numpy(), train.item_id.to_numpy())[1]]
    target_history = [train_history[u] for u in target_users]

    if HOUR is None:
        print("\n[扫描] val 上的 hour 网格（官方验证指标 = ndcg@100，即命中率@100）")
        scan = []
        for hour in HOURS:
            started = time.perf_counter()
            w = hour_weights(pairs, hour)
            keep = w > ZERO_EPS  # 官方 eliminate_zeros
            W = column_matrix(pairs.uid[keep], pairs.item[keep], w[keep], n_users, n_items)
            A = user_user_matrix(C, W, n_users)
            tops = top_k(A, Cn_t)
            metrics, _ = evaluate(
                list(tops[target_users]), target_chunks, n_items, all_tops=tops, history=target_history
            )
            scan.append((hour, metrics))
            print(
                f"  hour={hour:<6} 命中率@100={metrics['hitrate'][SELECT_K]:.6f}"
                f"  recall@100={metrics['recall'][SELECT_K]:.6f}  ({time.perf_counter() - started:.1f}s)"
            )

        best_hour, best_metrics = max(scan, key=lambda x: x[1]["hitrate"][SELECT_K])
        print(f"  → 最优 hour = {best_hour}（与官方一致：按命中率@100 选）")
    else:
        best_hour = HOUR
        print(f"\n[跳过扫描] hour={best_hour}（val 上选出的档，不在 {SPLIT} 上选参）")
        w = hour_weights(pairs, best_hour)
        keep = w > ZERO_EPS  # 官方 eliminate_zeros
        W = column_matrix(pairs.uid[keep], pairs.item[keep], w[keep], n_users, n_items)
        tops = top_k(user_user_matrix(C, W, n_users), Cn_t)
        best_metrics, _ = evaluate(
            list(tops[target_users]), target_chunks, n_items, all_tops=tops, history=target_history
        )

    show(f"[{SPLIT}] 官方口径（不过滤已交互物品、目标保留不可排名行）", best_metrics, len(target_users), n_items)


if __name__ == "__main__":
    main()
