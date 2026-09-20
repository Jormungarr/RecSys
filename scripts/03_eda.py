"""看一眼切分后的数据：分布、长尾、时间结构，给 baseline 实现和评测定口径。

运行：
    uv run python scripts/03_eda.py

只读 artifacts/splits/（口径 = Listen+ + 官方时间切分 + 训练集重编号）。
数字打在终端，图（6 张）写到 artifacts/eda/；结论写进 docs/eda.md。
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "splits"
FIGDIR = ROOT / "artifacts" / "eda"

HOUR = 3600
DAY = 24 * HOUR
WEEK = 7 * DAY
SESSION_GAP = 30 * 60

SOURCE = "来源：artifacts/splits（Listen+ / 官方时间切分，scripts/03_eda.py）"

# 中文字体（macOS 自带；找不到就退回默认，中文会变方框）
CJK_FONT = next(
    (name for name in ("PingFang SC", "Arial Unicode MS", "Hiragino Sans GB", "Heiti TC") if any(f.name == name for f in font_manager.fontManager.ttflist)),
    None,
)
if CJK_FONT:
    plt.rcParams["font.family"] = CJK_FONT
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(SPLITS / f"{name}.parquet")


def save(fig, name: str) -> None:
    """存图到 artifacts/eda/，并在图内标出处。"""
    fig.axes[0].annotate(SOURCE, xy=(1, 0), xycoords="axes fraction", xytext=(-2, -28), textcoords="offset points", ha="right", va="top", fontsize=7, color="0.45")
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[图] {path.relative_to(ROOT)}")


def share(part: int, whole: int) -> str:
    return f"{part:,} ({part / whole:.2%})"


def spread(values: np.ndarray) -> str:
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return f"中位数 {p50:,.0f} / 均值 {values.mean():,.1f} / p10 {p10:,.0f} / p90 {p90:,.0f} / 最大 {values.max():,.0f}"


def main() -> None:
    train = load("train")
    val = load("val")
    test = load("test")

    n_users = int(train.uid.max()) + 1
    n_items = int(train.item_id.max()) + 1
    print(f"[0] 规模       : train {len(train):,} 行 / {n_users:,} 用户 / {n_items:,} 物品；val {len(val):,} 行、test {len(test):,} 行")

    # [1] 目标集规模 —— 决定评测实现（分母=有目标的用户、k 取 10/50/100）
    for name, part in (("val ", val), ("test", test)):
        per_user = np.bincount(part.uid.to_numpy(), minlength=n_users)
        per_user = per_user[per_user > 0]
        print(f"[1] {name} 目标  : {spread(per_user)}；用户 {len(per_user):,}，只有 1 个目标的 {share(int((per_user == 1).sum()), len(per_user))}")

    # [2] 排不到的目标（物品不在训练集，记为 -1）—— 评测口径注意
    for name, part in (("val ", val), ("test", test)):
        item = part.item_id.to_numpy()
        uid = part.uid.to_numpy()
        affected = np.unique(uid[item == -1])
        users_all = int((np.bincount(uid, minlength=n_users) > 0).sum())
        users_rankable = int((np.bincount(uid[item >= 0], minlength=n_users) > 0).sum())
        print(f"[2] {name} 排不到: {share(int((item == -1).sum()), len(part))} 行；涉及用户 {len(affected):,}（占有目标用户的 {len(affected) / users_all:.1%}）；全部目标都排不到的用户 {users_all - users_rankable}")

    # 图：每用户目标数分布
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(0, np.log10(15_000), 40)
    for name, part, color in (("val", val, "tab:blue"), ("test", test, "tab:orange")):
        counts = np.bincount(part.uid.to_numpy(), minlength=n_users)
        counts = counts[counts > 0]
        ax.hist(counts, bins=bins, histtype="step", linewidth=1.8, color=color, label=f"{name}（中位 {np.median(counts):.0f}，最大 {counts.max():,}）")
    ax.set_xscale("log")
    ax.set_xlabel("每用户目标数（含重复行，log 轴）")
    ax.set_ylabel("用户数")
    ax.set_title(f"val / test 目标集规模（val {len(val):,} 行 / test {len(test):,} 行）")
    ax.legend()
    save(fig, "targets")

    # [3][6] 与训练历史的重叠、重复交互
    train_item = train.item_id.to_numpy()
    keys = train.uid.to_numpy().astype(np.int64) * n_items + train_item
    unique_keys = np.unique(keys)
    print(f"[6] train 重复 : 行 {len(keys):,} vs 去重 (uid,item) {len(unique_keys):,} → 重复率 {1 - len(unique_keys) / len(keys):.2%}")
    for name, part in (("val ", val), ("test", test)):
        raw_item = part.item_id.to_numpy()
        keep = raw_item >= 0
        item = raw_item[keep]
        uid = part.uid.to_numpy()[keep]
        pair = uid.astype(np.int64) * n_items + item
        pos = np.searchsorted(unique_keys, pair)
        hit = unique_keys[np.clip(pos, 0, len(unique_keys) - 1)] == pair
        print(f"[3] {name} 重叠  : 目标已在训练历史里 {share(int(hit.sum()), len(pair))}（分母=可排名的目标行）")
        print(f"[6] {name} 重复  : 行 {len(pair):,} vs 去重 {len(np.unique(pair)):,}")

    # [4] 用户历史长度（训练集、按去重物品算）
    history = np.bincount((unique_keys // n_items).astype(np.int64), minlength=n_users)
    history = history[history > 0]
    print(f"[4] 历史长度   : {spread(history)}；<5 个物品的用户 {share(int((history < 5).sum()), len(history))}")

    # 图：用户历史长度
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(history, bins=np.logspace(0, np.log10(history.max()), 40), color="tab:gray")
    ax.axvline(np.median(history), color="black", linewidth=1.4, label=f"中位 {np.median(history):,.0f}")
    ax.axvline(200, color="tab:red", linestyle="--", linewidth=1.4, label="SASRec 常用截断 200")
    ax.set_xscale("log")
    ax.set_xlabel("去重物品数（log 轴）")
    ax.set_ylabel("用户数")
    ax.set_title(f"训练集用户历史长度（{len(history):,} 个用户）")
    ax.legend()
    save(fig, "user_history")

    # [5] 物品热度长尾（训练集）
    popularity = np.bincount(train_item, minlength=n_items)
    popularity = popularity[popularity > 0]
    ranked = np.sort(popularity)[::-1]
    for frac in (0.01, 0.10):
        k = max(1, int(len(ranked) * frac))
        print(f"[5] top {frac:>4.0%} 物品: 占交互 {ranked[:k].sum() / ranked.sum():.1%}")
    print(f"[5] 长尾       : 只被交互 1 次的物品 {share(int((popularity == 1).sum()), len(popularity))}")

    # 图：物品热度长尾（log-log）
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.loglog(np.arange(1, len(ranked) + 1), ranked, linewidth=1.6)
    for frac in (0.01, 0.10):
        k = max(1, int(len(ranked) * frac))
        ax.axvline(k, color="0.6", linestyle="--", linewidth=1)
        ax.annotate(f"top {frac:.0%}：{ranked[:k].sum() / ranked.sum():.1%} 交互", xy=(k, ranked[k - 1]), xytext=(8, 14), textcoords="offset points", fontsize=9)
    ax.set_xlabel("物品热度排名（log 轴）")
    ax.set_ylabel("交互次数（log 轴）")
    ax.set_title(f"训练集物品热度长尾（{len(ranked):,} 个物品，单次交互占 {np.mean(popularity == 1):.1%}）")
    save(fig, "item_longtail")

    # [7] 时间结构（只在 train 上算）
    ts = train.timestamp.to_numpy()
    uid = train.uid.to_numpy()
    hourly = np.bincount(ts // HOUR).astype(float)
    print(f"[7] 时间轴     : 跨度 {ts.max() / DAY:.1f} 天；每小时事件中位数 {np.median(hourly):,.0f}；前 30 天 {hourly[: 30 * 24].mean():,.0f}/小时 vs 后 30 天 {hourly[-30 * 24:].mean():,.0f}/小时")

    prof24 = np.bincount((ts % DAY) // HOUR, minlength=24).astype(float)
    prof24 = prof24 / prof24.mean()
    print("[7] 24h 轮廓   : " + " ".join(f"{v:.2f}" for v in prof24))
    print(f"[7] 24h 峰谷   : 峰在相对第 {prof24.argmax()} 小时、谷在 {prof24.argmin()} 小时，峰谷比 {prof24.max() / prof24.min():.1f}")
    mid = ts.max() / 2
    first = np.bincount((ts[ts < mid] % DAY) // HOUR, minlength=24).astype(float)
    second = np.bincount((ts[ts >= mid] % DAY) // HOUR, minlength=24).astype(float)
    print(f"[7] 24h 稳定   : 前半 vs 后半相关 {np.corrcoef(first, second)[0, 1]:.3f}（峰位 {first.argmax()} vs {second.argmax()}）")

    prof7 = np.bincount((ts % WEEK) // DAY, minlength=7).astype(float)
    prof7 = prof7 / prof7.mean()
    print("[7] 7d 轮廓    : " + " ".join(f"{v:.2f}" for v in prof7) + f"（峰谷比 {prof7.max() / prof7.min():.2f}；300 天不能被 7 整除，分桶天然不均约 2%）")

    gaps = np.diff(ts)[uid[1:] == uid[:-1]]
    print(f"[7] 用户内间隔 : 中位数 {np.median(gaps):,.0f} 秒 / p90 {np.percentile(gaps, 90):,.0f} / =0 的 {np.mean(gaps == 0):.1%} / >30 分钟的 {np.mean(gaps > SESSION_GAP):.1%}")

    # 图：每日活动量（滚动均值只用于看趋势）
    daily = np.bincount(ts // DAY).astype(float)
    smooth = pd.Series(daily).rolling(7, center=True, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(daily, linewidth=0.7, alpha=0.35, label="每日事件数")
    ax.plot(smooth, linewidth=2.0, label="7 日滚动均值")
    ax.annotate(f"前 30 天均值 {daily[:30].mean():,.0f}/天", xy=(30, daily[:30].mean()), xytext=(12, -26), textcoords="offset points", fontsize=9)
    ax.annotate(f"后 30 天均值 {daily[-30:].mean():,.0f}/天", xy=(len(daily) - 30, daily[-30:].mean()), xytext=(-12, 10), textcoords="offset points", fontsize=9, ha="right")
    ax.set_xlabel("相对第几天（起点被官方隐去）")
    ax.set_ylabel("事件数 / 天")
    ax.set_title(f"活动量趋势：后 30 天是前 30 天的 {daily[-30:].mean() / daily[:30].mean():.2f} 倍")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "activity_daily")

    # 图：24h 轮廓（相位未知）
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(24), prof24, marker="o", markersize=3.5, linewidth=1.8, label=f"全部（峰谷比 {prof24.max() / prof24.min():.1f}）")
    ax.plot(range(24), first / first.mean(), linestyle="--", linewidth=1.2, alpha=0.8, label=f"前半（峰位 {first.argmax()}）")
    ax.plot(range(24), second / second.mean(), linestyle="--", linewidth=1.2, alpha=0.8, label=f"后半（峰位 {second.argmax()}）")
    ax.axhline(1.0, color="0.6", linewidth=0.8)
    ax.set_xlabel("相对天内第几小时（相位未知：不是真实钟点）")
    ax.set_ylabel("事件数 / 全天均值")
    ax.set_title(f"24h 轮廓：谷在相对第 {prof24.argmin()} 小时，前后半相关 {np.corrcoef(first, second)[0, 1]:.3f}")
    ax.legend(fontsize=9)
    save(fig, "hourly_profile")

    # 图：用户内相邻事件间隔（=0 画不进 log 轴，标在轴上）
    nonzero = gaps[gaps > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(nonzero, bins=np.logspace(0, np.log10(nonzero.max()), 60), color="tab:purple")
    ax.axvline(np.median(nonzero), color="black", linewidth=1.4, label=f"中位 {np.median(nonzero):,.0f} 秒")
    ax.axvline(SESSION_GAP, color="tab:red", linestyle="--", linewidth=1.4, label="30 分钟（会话阈值）")
    ax.set_xscale("log")
    ax.set_xlabel(f"同一用户相邻事件间隔（秒，log 轴；=0 的 {np.mean(gaps == 0):.1%} 未画入）")
    ax.set_ylabel("次数")
    ax.set_title(f"用户内相邻事件间隔：>30 分钟的占 {np.mean(gaps > SESSION_GAP):.1%}")
    ax.legend(fontsize=9)
    save(fig, "gaps")


if __name__ == "__main__":
    main()
