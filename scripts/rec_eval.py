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
- coverage@k 的**用户集合**官方自己不一致：`Coverage(cut_off_ranked=False)` 用全部用户（popularity / itemknn 的
  `Ranked` 就是训练集全部用户），但官方 sasrec 的 eval 把用户 inner join 到测试集、只剩有目标的用户。
  所以这里两条都报：`coverage(全用户)`（只在调用方给了 `all_tops` 时才有）与 `coverage(有目标)`。
  两者都与评测窗口无关（top-k 只由训练集和模型决定），同一份权重下 val / test 同值。
- **回访 / 新歌两桶**：官方口径不过滤已交互物品，而目标里有很大一部分是“用户训练期已经听过的歌”
  （口径 A test 占 66.6%），不拆开就看不出模型赢在哪条轴上。所以给 `evaluate` 传 `history` 时会额外报
  「回访 recall / 命中率」与「新歌 recall / 命中率」——两桶都用**逐用户**口径（recall 分母 = min(桶内行数, k)，
  只在桶内有目标的用户上平均），与主表可比。回访 = 目标物品在该用户的训练期历史里出现过；新歌 = 没出现过。
- 官方 ndcg@k 是 real_dcg / real_dcg（已知 bug），实际含义 = 命中率@k；这里同时报正确 NDCG 和命中率
"""

import numpy as np

KS = (10, 50, 100)


def split_by_user(uid: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """输入已按 uid 升序排好；返回 (每个分组的 uid, 每组的 values)。"""
    starts = np.concatenate(([0], np.flatnonzero(np.diff(uid)) + 1))
    return uid[starts], np.split(values, starts[1:])


def evaluate(
    tops: list[np.ndarray],
    chunks: list[np.ndarray],
    n_items: int,
    all_tops: np.ndarray | None = None,
    history: list[np.ndarray] | None = None,
) -> tuple[dict, int]:
    """按官方定义算指标；tops 与 chunks（每用户的目标行，含 -1）一一对应。

    all_tops = 全部用户的 top-k（`(n_users, k)` 数组）。给了就多算一份“全用户”的 coverage
    （官方 popularity / itemknn 的口径）；不给就只有“有目标用户”那份（官方 sasrec 的口径）。
    history = 与 tops / chunks 一一对应的「该用户在训练期交互过的物品」。给了就多报「回访 / 新歌」两桶
    （口径见文件头），两桶都是逐用户平均，只在桶内有目标的用户上算。
    """
    discount = 1.0 / np.log2(np.arange(2, max(KS) + 2))
    sums = {name: dict.fromkeys(KS, 0.0) for name in ("recall", "dcg", "hitrate", "ndcg")}
    recommended: dict[int, set] = {k: set() for k in KS}
    buckets = {
        label: {"users": 0, "rows": 0, "recall": dict.fromkeys(KS, 0.0), "hitrate": dict.fromkeys(KS, 0.0)}
        for label in ("回访", "新歌")
    }
    n_users = 0

    for index, (top, chunk) in enumerate(zip(tops, chunks)):
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

        if history is not None:
            seen = np.asarray(history[index])
            in_history = np.isin(chunk, seen) if seen.size else np.zeros(chunk.shape, dtype=bool)
            for label, mask in (("回访", in_history), ("新歌", ~in_history)):
                rows = int(mask.sum())
                if rows == 0:  # 该用户在这个桶里没目标，不进这个桶的分母
                    continue
                bucket_hits = np.flatnonzero(np.isin(top, chunk[mask]))
                bucket = buckets[label]
                bucket["users"] += 1
                bucket["rows"] += rows
                for k in KS:
                    bucket["recall"][k] += bucket_hits[bucket_hits < k].size / min(rows, k)
                    bucket["hitrate"][k] += 1.0 if (bucket_hits < k).size else 0.0

    metrics = {name: {k: value / n_users for k, value in per_k.items()} for name, per_k in sums.items()}
    metrics["coverage"] = {k: len(recommended[k]) / n_items for k in KS}
    if all_tops is not None:
        all_tops = np.asarray(all_tops)
        metrics["coverage_all"] = {k: np.unique(all_tops[:, :k]).size / n_items for k in KS}
    if history is not None:
        metrics["bucket"] = {
            label: {
                "users": bucket["users"],
                "rows": bucket["rows"],
                "recall": {k: bucket["recall"][k] / bucket["users"] for k in KS},
                "hitrate": {k: bucket["hitrate"][k] / bucket["users"] for k in KS},
            }
            for label, bucket in buckets.items()
        }
    return metrics, n_users


def show(title: str, metrics: dict, n_users: int, n_items: int) -> None:
    print(f"\n{title}（用户 {n_users:,}，候选池 {n_items:,}）")
    print(f"  {'指标':<12}" + "".join(f"{'@' + str(k):>12}" for k in KS))
    rows = [
        ("recall", metrics["recall"]),
        ("dcg", metrics["dcg"]),
        ("ndcg(正确)", metrics["ndcg"]),
        ("命中率(官方ndcg)", metrics["hitrate"]),
    ]
    if "coverage_all" in metrics:  # 官方 popularity / itemknn 的口径
        rows.append(("coverage(全用户)", metrics["coverage_all"]))
    rows.append(("coverage(有目标)", metrics["coverage"]))
    if "bucket" in metrics:  # 回访 / 新歌两桶（逐用户口径，口径见文件头）
        for label in ("回访", "新歌"):
            rows.append((f"{label} recall", metrics["bucket"][label]["recall"]))
            rows.append((f"{label} 命中率", metrics["bucket"][label]["hitrate"]))
    for label, values in rows:
        print(f"  {label:<12}" + "".join(f"{values[k]:>12.6f}" for k in KS))
    if "bucket" in metrics:
        back, new = metrics["bucket"]["回访"], metrics["bucket"]["新歌"]
        print(f"  分桶：回访 {back['users']:,} 用户 / {back['rows']:,} 行；新歌 {new['users']:,} 用户 / {new['rows']:,} 行")
