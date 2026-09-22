# Yambda 数据集笔记（官方结论 + 实测核对）

> 本文只记录**有出处**的事实。官方没说的，明确标注为"未解释"。
> 所有实测数字可用文末命令复现。

## 出处

| 编号 | 来源 | 用途 |
|---|---|---|
| S1 | [HF 数据集卡片](https://huggingface.co/datasets/yandex/yambda)（README.md） | 字段类型与含义、排序说明、FAQ |
| S2 | 论文 [arXiv:2505.22238](https://arxiv.org/abs/2505.22238)（Yambda-5B，v2） | 数据集构成、统计表、评测协议 |
| S3 | [讨论 #15 Start Time for Timestamps](https://huggingface.co/datasets/yandex/yambda/discussions/15) | 作者回复：时间戳起点不公开 |
| S4 | [讨论 #14 Unlike vs Dislike](https://huggingface.co/datasets/yandex/yambda/discussions/14) | 作者回复：四个显式反馈按钮的真实语义 |
| S5 | [讨论 #17 timestap](https://huggingface.co/datasets/yandex/yambda/discussions/17) | 同样的时间戳问题，至今无人回复 |
| S6 | `benchmarks/` 目录（官方 benchmark 代码） | 评测协议常量、指标实现 |

## 字段定义

来自 S1 的 *Common Event Structure* 与 *Unified Event Structure* 两节（原文摘录）。

| 字段 | 类型 | 官方原话 | 含义 |
|---|---|---|---|
| `uid` | uint32 | *Unique user identifier* | 用户 id，**不连续**（实测 9,238 个用户散布在 100 ~ 1,000,000） |
| `item_id` | uint32 | *Unique track identifier* | **曲目** id（不是专辑/艺人），**不连续**（22 ~ 9,390,623） |
| `timestamp` | uint32 | *Delta times, binned into 5s units.* | 事件时间，**单位 = 1 秒**（数值量化到 5 秒的倍数），相对起点（起点不公开，见下） |
| `is_organic` | uint8 | *Boolean flag (0/1) indicating if the interaction was algorithmic (0) or organic (1)* | 0 = 推荐/算法驱动，1 = 用户主动发现 |
| `played_ratio_pct` | Optional[uint16] | *Percentage of track played (1-100), null for non-listen events* | 播放百分比（实测 0 ~ 159，见"已知不一致"） |
| `track_length_seconds` | Optional[uint32] | *Total track duration in seconds, null for non-listen events* | 曲目总时长（秒） |

排序（S1）：*All files are sorted by (uid, timestamp) in ascending order.* → 实测符合，文件顺序即可当行为序列。

正样本定义（S1 FAQ）：*A track is considered "listened" if over 50% of its duration is played.* → 即论文中的 **Listen+**。

## 五个事件文件的实测统计（`flat/50m`）

| 文件 | 行数 | 用户数 | 曲目数 | 时间戳范围 |
|---|---|---|---|---|
| `listens.parquet` | 46,467,212 | 9,238 | 877,168 | 0 ~ 26,000,000 |
| `likes.parquet` | 881,456 | 8,283 | 181,304 | 80 ~ 25,999,970 |
| `dislikes.parquet` | 107,776 | 5,951 | 53,413 | 75 ~ 25,999,945 |
| `unlikes.parquet` | 312,972 | 6,406 | 117,953 | 250 ~ 25,999,945 |
| `undislikes.parquet` | 21,033 | 2,911 | 15,399 | 15,950 ~ 25,993,175 |

与论文 Table 3（Yambda-50M：10,000 用户 / 934,057 曲目 / listens 46,467,212 / likes 881,456 / dislikes 107,776）**逐项吻合**。

### 并集核对：为什么 listens 单独数不出 10,000 用户

```
listens 里的用户/曲目            :   9,238 / 877,168
只出现在显式反馈里的             :     762 /  56,889
五个文件并集                     :  10,000 / 934,057
官方公布的数字                   :  10,000 / 934,057
差额                            :      +0 /      +0
```

这 56,889 首"零播放"曲目是天然的**冷启动测试集**；762 个"零播放"用户在做时间切分时会因为"只保留训练集出现过的用户"被剔除——这是协议选择，不是数据缺陷。

## 时间戳专题（最容易踩坑的地方）

**官方说法**（S1）：*Delta times, binned into 5s units.*
**论文说法**（S2 §3.3）：事件时间做了 `T' = [(T_event - T_start) / 5] × 5` 变换，其中 `T_start` 是数据集中第一个事件的时间戳，保留 5 秒精度（"5-second precision"）。

**单位判定：数值单位 = 1 秒，量化到 5 秒的倍数**。论文公式 `[(T_event − T_start)/5] × 5` 的输出仍是"秒"，只是全部落在 5 秒的整数倍上（"preserves temporal ordering with 5-second precision"）。

**最简判据：先看官方常量怎么用**。官方 benchmark 里 `GAP_SIZE = 1800` 对应论文的 *"Gap: 30 minutes"*，`VAL_SIZE = 86400` 对应 *"Test: 1 day"*；1800 = 30×60 秒、86400 = 24×3600 秒 → **常量本身就是秒**。判量纲应先看官方常量，不用绕圈子。

四条判据（决定性的是第一条）：

| 判据 | 若 1 单位 = 1 秒 | 若 1 单位 = 5 秒 | 结论 |
|---|---|---|---|
| 最大时间戳 26,000,000 的跨度 | **300.9 天 ≈ 9.9 个月** | 1504.6 天 ≈ 4.1 年 | 论文："approximately 11-month observation period"、"Train: 300 days / Test: 1 day" → 只有"秒"吻合 |
| 用户活跃跨度中位数 | 283 天（在 301 天窗口内） | 1416 天（超出窗口） | 只有"秒"自洽 |
| 人均每日 listens | 16.7 条/天 | 3.3 条/天 | 前者符合音乐 App 量级 |
| `track_length_seconds` 是否都是 5 的倍数 | 是（论文：时长也量化到 5 秒） | — | 支持"秒 + 5 秒量化" |

**其他实测**：
- 相邻事件时间戳差值的最小值 = 5，且全部是 5 的倍数 → 量化步长确认为 5 秒
- 所有用户共享同一条时间轴（各用户最晚时间戳中位数 25,936,402，全局最大 26,000,000）→ 是**全局时间**，不是每个用户各自的相对时间
- 文件按 `(uid, timestamp)` 升序，用户内时间戳无回退

**换算**：天数 = `timestamp / 86400`。官方测试集从 `25,913,600` 开始 = 第 299.9 天；训练集截止 `25,823,600` = 第 298.9 天（约 300 天，与论文一致）。

**起点不可得**（S3，作者 tytskiy 2025-11-20 原话）：

> *"Unfortunately, we're unable to share the exact start time or any other timestamp-related information. This restriction is in place to comply with our privacy policy, which prevents us from releasing details that could make user activity traceable."*

→ **绝对日期永久不可得**。任何"第几天"都只能是相对值。同样的提问在 S5 中无人回复。

**未解释的局限**：实测存在大量"同一 `(uid, timestamp)` 里有多次播放"的组（最大 163 条事件共用一个时间戳；3.54% 的组含 ≥2 次完整播放，涉及 13.17% 的事件）。例：170 秒 + 105 秒两首完整听完，时间戳完全相同。官方文档与论文均无解释。

**可操作结论**：

| 可以用 | 不可以用 |
|---|---|
| 排序（文件已排好，用户内单调） | 用相邻时间戳相减推算单曲收听时长 |
| 按时间切训练/验证/测试（5 秒精度足够） | 判定同一时间戳内多首的先后顺序 |
| 粗粒度会话划分（如 >30 分钟视为新会话） | 精确对齐每首歌的起止时刻 |
| 按天/小时聚合看活跃度 | 去重 |

要算"听了多久"，用显式字段：`track_length_seconds * played_ratio_pct / 100`。

## 四个显式反馈的真实语义（作者原话）

提问者以为 unlike 是"先喜欢后不喜欢"，作者 ploshkin 纠正（S4）：

> *"There are 2 different buttons in Yandex.Music: **like** means "add to favourites", hence **unlike** is "remove from favourites"; **dislike** stands for "do not recommend me this track again", so **undislike** cancels this intent."*

| 事件 | 含义 | 能否当负样本 |
|---|---|---|
| `like` | 加入收藏 | 正样本 |
| `unlike` | **取消收藏**（移出收藏夹） | ❌ 不能 |
| `dislike` | 别再推荐这首歌（屏蔽） | ✅ 明确负反馈 |
| `undislike` | 撤销屏蔽 | — |

**数据侧验证**：dislike 中 81.3% 之前播放过该曲目、其中 98.2% 的 dislike 发生在播放之后；unlike 中只有 28.1% 能在窗口内找到对应 like（其余是窗口开始前就已收藏），其中有 like 的 84.7% 时序正确（unlike 晚于 like）。

## 未进管线的字段与文件（2026-09-22 盘点）

原始文件全都下过、sha256 校验过，但**进 `artifacts/splits/` 的只有 `uid/item_id/timestamp` 三列**。以下都没被任何模型看到：

| 文件 | 规模 | 列 | 现状 |
|---|---|---|---|
| `flat/50m/listens.parquet` | 46,467,212 行 | uid, timestamp, item_id, **is_organic**, **played_ratio_pct**, **track_length_seconds** | 只用了前三列 |
| `flat/50m/likes.parquet` | 881,456 行 | uid, timestamp, item_id, is_organic | 完全没用 |
| `flat/50m/dislikes.parquet` | 107,776 行 | 同上 | 完全没用 |
| `flat/50m/unlikes.parquet` | 312,972 行 | 同上 | 完全没用 |
| `flat/50m/undislikes.parquet` | 21,033 行 | 同上 | 完全没用 |
| `artist_item_mapping.parquet` | 9,271,906 行 / 1,293,394 艺人 / 9,270,506 物品 | artist_id, item_id | 完全没用 |
| `album_item_mapping.parquet` | 9,651,644 行 / 3,367,691 专辑 / 8,653,783 物品 | album_id, item_id | 完全没用 |
| `sequential/50m/listens.parquet` | 9,238 行（每用户一行、嵌套列表） | 同 listens 六列 | 不用（序列自己从 flat 建） |
| `embeddings.parquet` | 13.8 GB | 曲目内容嵌入 | **未下载** |

各字段实测：

| 字段 | 实测 |
|---|---|
| `listens.is_organic` | 推荐驱动（0）占 **48.3%**（论文报 48.74%）；按正负样本拆开是 **Listen+ 52.03% / Listen− 41.97%**（train 里 52.01%）——推荐推的歌反而更容易被“听完” |
| 四个反馈文件的 `is_organic` | 推荐驱动占 43.0%（likes）/ 48.1%（dislikes）/ **5.3%**（unlikes）/ 11.0%（undislikes） |
| `played_ratio_pct` | `=0` 占 7.4%、`<50` 占 **36.6%**（现在被整段丢弃）、`≥50` 占 63.4%；`>100` 占 0.47%；中位/p90/p99 都是 100 |
| `track_length_seconds` | min 5 / 中位 200 / p90 275 / max 2495 秒；无 0 |

**时间戳是 5 秒分箱**：卡片那句 *"Delta times, binned into 5s units"* 实测成立——`listens.timestamp % 5 == 0` 的比例是 **1.0000**。
这解释了建模时看到的“**13.1% 的相邻两首时间戳完全相同**”（2026-09-22 的只读检查；口径是**模型输入的最后 512 条窗口**，全历史口径是 12.1%，见 `docs/eda.md`）：同一 5 秒桶内的事件无法用时间区分，所以任何“只用 Δ”的时间编码在那部分位置上也必然重合。

**可以直接用的三件事**（都还没接进模型）：① `played_ratio_pct` 从“阈值”升级成“参与度”（分级加权 / 实际收听秒数 = ratio × length）；
② 那 36.6% 的 `<50` 行是“点开就划走”的弱负反馈，现在被丢了；③ 艺人/专辑映射让模型知道“这首歌的艺人/专辑你听过”——冷启动（新歌轴）最可能的杠杆。

**2026-09-22 阶段 1：以上字段与文件已落进 artifacts**（见 `memory.md`「已完成」27）。口径 A / B 各一套，与 splits 同一套 id 空间与 `-1` 约定：`train/val/test.parquet` 补了 `is_organic` / `played_ratio_pct` / `track_length_seconds` 三列（正样本定义与行集合不变）；另存 `feedback.parquet`（四个反馈 + `event_type`）、`weak_negative.parquet`（**训练窗口**内 `<50` 的行，A 1,650 万 / B 1,658 万行）、`item_artist.parquet` / `item_album.parquet`（只覆盖该口径训练集的物品，覆盖 99.57% / 99.96%）。**模型还没用到它们。**

## 已知的文档与数据不一致

1. `played_ratio_pct` 卡片写 *1-100*，实测范围 **0 ~ 159**：存在 7.45% 的 0（一次都没播到）、0.469% 的 >100（S1 FAQ 解释为回放/拖动，属正常）。
2. "Delta times" 措辞含糊（是事件间隔还是距起点的时间？）——论文的公式表明是**距起点的时间**。
3. 官方 benchmark 的指标实现有已知 bug：`benchmarks/yambda/evaluation/metrics.py` 中 NDCG 用真实 DCG 除以真实 DCG 当理想值，recall 的分母注释也提到曾用错。作者已开 PR 修正（#20/#21/#22）。**复现基准或对比论文表格时必须注意**。

## 对建模的结论

| 事实 | 影响 |
|---|---|
| Listen+（`played_ratio_pct >= 50`）保留 63.35% 的事件 | 这是默认的正样本定义 |
| `is_organic=0`（推荐驱动）占 listens 的 48.3%（论文报 48.74%） | 可做"推荐 vs 主动发现"的对照实验；Unlike 只有 5.3% 是推荐驱动 |
| 显式反馈极稀疏（likes ≈ listens 的 1.9%） | 单独用 like 训练样本不足；论文因此分 Listen+ / Like 两套实验 |
| 稀疏度 = 0.5734%；Top 1% 曲目覆盖 49.2% 的事件；35.7% 曲目只被听 1 次 | 长尾极重，必须走"召回 → 排序"两级结构 |
| 官方 GTS 协议：训练 300 天 / gap 30 分钟 / 测试 1 天，测试集只占 0.5% | 评测按时间切，不能随机划分 |

## 复现本文数字

```bash
# 环境
uv sync

# 数据样式与字段含义
uv run python scripts/01_inspect_listens.py
```

跨文件的一次性核对（并集 10,000 / 934,057、时间戳单位判据、显式反馈语义）结论已固化在上文，不再单独保留脚本；其中"文件已按 (uid, timestamp) 排序"与"uid/item_id 不连续必须重编号"两条会作为数据管线的输入断言长期保留。

