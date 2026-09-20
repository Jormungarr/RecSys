"""评测指标：与官方 benchmark 同一份口径，供各 baseline 脚本共用。

用法（被 scripts/ 下的脚本 import，不是可执行脚本）：
    from rec_eval import evaluate, show, split_by_user

输入约定：物品 id 已重编号（0..n_items-1），tops[i] 是第 i 个用户的 top-100（降序），
chunks[i] 是同一用户的目标行（含重复、含 -1）；-1 = 训练集没出现过的物品，永远不会命中。

口径出处（vendor/yambda-benchmarks/benchmarks/）：
- 分母只统计「有目标的用户」（官方 yambda/evaluation/metrics.py::cut_off_ranked）
- recall@k   = 命中数 / min(目标行数, k)      ← 官方 clamp(num_positives, max=k)，分母是"行数"
- dcg@k      = Σ_{命中的位次} 1 / log2(位次 + 2)
- coverage@k = top-k 里出现过的不重复物品数 / 候选池大小
- 官方 ndcg@k 是 real_dcg / real_dcg（已知 bug），实际含义 = 命中率@k；这里同时报正确 NDCG 和命中率
"""

import numpy as np

KS = (10, 50, 100)


def split_by_user(uid: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """输入已按 uid 升序排好；返回 (每个分组的 uid, 每组的 values)。"""
    starts = np.concatenate(([0], np.flatnonzero(np.diff(uid)) + 1))
    return uid[starts], np.split(values, starts[1:])


def evaluate(tops: list[np.ndarray], chunks: list[np.ndarray], n_items: int) -> tuple[dict, int]:
    """按官方定义算指标；tops 与 chunks（每用户的目标行，含 -1）一一对应。"""
    discount = 1.0 / np.log2(np.arange(2, max(KS) + 2))
    sums = {name: dict.fromkeys(KS, 0.0) for name in ("recall", "dcg", "hitrate", "ndcg")}
    recommended: dict[int, set] = {k: set() for k in KS}
    n_users = 0

    for top, chunk in zip(tops, chunks):
        if chunk.size == 0:
            continue
        n_users += 1
        hit_positions = np.flatnonzero(np.isin(top, chunk))
        n_distinct = int(np.unique(chunk).size)
        for k in KS:
            hits = hit_positions[hit_positions < k]
            gain = float(discount[hits].sum())
            sums["recall"][k] += hits.size / min(chunk.size, k)
            sums["dcg"][k] += gain
            sums["hitrate"][k] += 1.0 if hits.size else 0.0
            sums["ndcg"][k] += gain / float(discount[: min(n_distinct, k)].sum())
            recommended[k].update(top[:k].tolist())

    metrics = {name: {k: value / n_users for k, value in per_k.items()} for name, per_k in sums.items()}
    metrics["coverage"] = {k: len(recommended[k]) / n_items for k in KS}
    return metrics, n_users


def show(title: str, metrics: dict, n_users: int, n_items: int) -> None:
    print(f"\n{title}（用户 {n_users:,}，候选池 {n_items:,}）")
    print(f"  {'指标':<10}" + "".join(f"{'@' + str(k):>12}" for k in KS))
    for name, label in (
        ("recall", "recall"),
        ("dcg", "dcg"),
        ("ndcg", "ndcg(正确)"),
        ("hitrate", "命中率(官方ndcg)"),
        ("coverage", "coverage"),
    ):
        print(f"  {label:<10}" + "".join(f"{metrics[name][k]:>12.6f}" for k in KS))
