# RecSys

用 Yandex **Yambda-5B** 数据集搭建一个推荐系统。做两件事：

1. **基本架构**：把一条最小可用的管线立起来——数据下载与校验 → 官方口径的时间切分与 id 重编号 → EDA → 统一评测口径 → 模型按同一口径接入。
2. **论文 / 官方 benchmark 复现**：`popularity`、`itemknn`、`sasrec` 三个基线各自己实现一份，并与官方实现逐项对照。

## 三个基线复现到什么程度

| 模型 | 我们的实现 | 与官方实现的一致性证据 | val recall@100 |
|---|---|---|---|
| popularity | `scripts/04_popularity.py` | val 数字与官方**逐项一致到 6 位小数**（含 7 档 `hour` 扫描、最优档也相同） | 0.047727 |
| itemknn | `scripts/05_itemknn.py` | 13 档 `hour` 网格跑满、最优档都是 0.5；逐用户 top-100 平均重合 **99.88/100**（4,627 人中 4,587 人完全相同） | 0.103341 |
| sasrec | `scripts/06_sasrec.py` | 把官方 checkpoint 的权重导进来喂同一批输入，前向输出**逐位相同（最大差 0；那条对照是在 `MAX_SEQ_LEN=200` / `emb 64` 下做的）**；训练侧有随机性，只能对量级。2026-09-21 重写注意力 block（为了能加加性偏置）后，用同权重逐层对照验证与原实现等价（完整 block 最大差 4.8e-7） | 0.121823 |

上表的 `val recall@100` 只作复现程度的摘要；val / test 的完整指标见下一节。

## 各方法的效果（val / test）

同一份第一遍切分、同一份评测口径（候选池 627,648，不过滤已交互物品、目标里的 -1 保留为不可排名行）。

**val**（4,627 个有目标的用户）

| 方法 | recall@10 | recall@50 | recall@100 | 命中率@100 | ndcg@100 | coverage@100(有目标) |
|---|---|---|---|---|---|---|
| popularity（hour=1.0） | 0.022164 | 0.032730 | 0.047727 | 0.408472 | 0.034326 | 0.000159 |
| sasrec（50 epoch，seq 512 / emb 128 / 4 层） | 0.069653 | 0.088247 | 0.121823 | 0.685974 | 0.104075 | 0.032110 |
| itemknn（hour=0.5） | 0.057684 | 0.077010 | 0.103341 | 0.603847 | 0.086693 | 0.037583 |

**test**（4,599 个有目标的用户）

| 方法 | recall@10 | recall@50 | recall@100 | 命中率@100 | ndcg@100 | coverage@100(有目标) |
|---|---|---|---|---|---|---|
| popularity（hour=1.0） | 0.021764 | 0.033031 | 0.046988 | 0.388128 | 0.033241 | 0.000159 |
| sasrec（50 epoch，seq 512 / emb 128 / 4 层） | 0.060455 | 0.077638 | 0.110816 | 0.650359 | 0.090606 | 0.032158 |
| itemknn（hour=0.5） | 0.055089 | 0.071640 | 0.098117 | 0.579691 | 0.078911 | 0.037859 |

- **命中率@100** 就是官方那个 `ndcg@100`（官方实现有 bug，实际等于命中率）；**ndcg@100** 是正确实现。两个都报，不被官方 bug 带偏，同时保留对表能力。
- **序在两个窗口上一致**：**sasrec > itemknn > popularity**（recall@100 与命中率@100 都是）。这跟**官方表**的相对序不同了（官方是 itemknn 0.1085 > sasrec 0.0828）：我们这份 sasrec 已经不是官方配置（seq 512 / emb 128 / 4 层 vs 官方 200 / 64 / 2），而另两个基线仍是复现配置。
- **val → test 略降**（recall@100）：popularity −1.5%、itemknn −5.1%、sasrec −9.0%。test 是更靠后的窗口，而模型只用 train 训练，降幅符合预期。
- **超参只在 val 上选**：popularity hour=1.0、itemknn hour=0.5（13 档网格里选出的），test 直接用同一档，不在 test 上选参。sasrec 在 test 上复用 val 那次训练的 checkpoint，两列出自**同一个模型**。
- **重跑**：`uv run python scripts/04_popularity.py test 1.0` / `05_itemknn.py test 0.5` / `06_sasrec.py test`。三个基线在 test 上的排序只由训练集决定（用户向量与 top-100 与评测窗口无关），所以换 split 只换评测目标。
- 逐档 dcg、两个对照口径（过滤已交互 / 只看可排名目标）、`hour` 网格，见 `docs/baselines.md`。
- **coverage 有两种用户集合**（官方自己就不一致）：表里那一列是 `coverage@100(有目标)`；另有 `coverage@100(全用户)`——口径 A 下 popularity 0.000159、sasrec **0.041816**、itemknn **0.064442**（口径 B 见 `docs/baselines.md`）。它只由训练集和模型决定、**与评测窗口无关**，所以 val / test 同值。官方 popularity / itemknn 用全用户口径、官方 sasrec 用有目标用户口径，popularity / itemknn 两条都报、都能对表（sasrec 那格口径相同但数值差 58%，因为它不是官方配置）。
- **回访 vs 新歌**：官方指标里 **66.6% 的目标是“回访”**（训练期已听过的歌）。拆开看，**容量包之后两条轴都是 sasrec 第一**（口径 A test，recall@100）：回访 sasrec 0.1399 > itemknn 0.1272 > popularity 0.0484，新歌 sasrec 0.0450 > popularity 0.0448 > itemknn 0.0246（新歌轴只领先 0.4%，很薄）。之前那条“两条轴排序完全相反、新歌轴最好是 popularity”是 d256 时代的结论，已作废。`04/05/06` 每次运行都会打印这两桶，细节见 `docs/baselines.md`。
- **排名融合已作废、待重做**：之前记的“itemknn + sasrec 的 RRF 融合 0.1081（超 itemknn +10.2%）”用的是 d256 那版的 top-100，而那份 sasrec 数字是在一个评测缺陷（padding 行 NaN，使 25% 用户失效）下算出来的——融合的输入本身就是错的。现在 sasrec 单模型已经是 0.1108（test）且两条轴都第一，这个问题要重新问。缺陷的机理与修正后的数字见 `docs/baselines.md`「评测实现里一个缺陷：padding 行的 NaN」。

官方 benchmark 的 test 表（50m / listens，recall@100）：random 0.00006 → **popularity 0.0479** → **sasrec 0.0828** → **itemknn 0.1085**。它用的是**第二遍训练集**（`val_size=0`，train 延伸到 test 前 30 分钟），与上表的口径不同、**不能并排比**。这套协议已经走通：同一口径下 popularity 与官方**六位小数全中**（含 dcg 与 coverage）、itemknn 差 ≤5.3e-5（这两格的序与官方一致）；sasrec 那一格**比官方高 14.1%**（口径 B：我们 0.094515 vs 官方 0.082849），但两边配置不同（我们 seq 512、官方 200），所以只能说明“同一套切分 / id 空间 / 候选池 / 评测代码下能跑赢官方已发表的数”，**不算复现一致**。见 `docs/baselines.md` 的“版本 B 口径”一节。

## 口径（为什么这么做）

- **正样本 = Listen+**（`played_ratio_pct >= 50`）：官方 FAQ 与各模型预处理都用它。
- **按官方 GTS 时间切分**，不随机切：随机切会把未来行为混进训练，离线指标虚高。
- **id 空间按训练集重编号**（uid `0..9,206`、item `0..627,647`）：评测候选池天然等于 id 空间，不用额外维护候选 mask；val / test 里训练集没出现过的物品记 `-1`，评测时算"必然未命中"。
- **评测自己实现**（`scripts/rec_eval.py`，三个模型共用）：官方包依赖 torch + polars 且自带另一套切分。官方那个 `ndcg` 有已知 bug（实际等于命中率），我们并行报"正确 NDCG"和"命中率"。
- 口径的代码级出处写在各脚本 docstring 里；数据流与设计取舍见 `architecture.md`。

## 目录

```
scripts/   01 看数据 / 02 切分 / 03 EDA / 04 popularity / 05 itemknn / 06 sasrec
           rec_eval.py      评测库（三个模型共用）
docs/      dataset_notes.md  数据集事实与已知坑
           benchmark_repro.md 官方 benchmark 在本机的落地与官方数字
           eda.md            切分后数据的分布 / 长尾 / 时间结构
           baselines.md      各基线的原理、数字与对照
AGENTS.md        协作约定与索引
architecture.md  数据流、口径、目录约定、为什么这么设计
memory.md        进度、关键决策与理由、数据与产物状态
```

## 跑起来

```bash
uv sync
uv run python scripts/02_build_splits.py   # 需要先下好 data/raw/flat/50m（见 docs/dataset_notes.md）
uv run python scripts/04_popularity.py     # 约 3 秒
uv run python scripts/05_itemknn.py        # 13 档 hour 网格约 16 分钟
uv run python scripts/06_sasrec.py         # 50 epoch；按脚本**当前常量**（emb 64 / 2 层 / seq 512）约 13 分钟（MPS 训练、CPU 推理；墙钟可能因机器 Idle Sleep 拉长）
```

**sasrec 的两条注意**（另外三个脚本没有这个问题）：

- **脚本顶部的常量是滚动的**：`EMB` / `LAYERS` / `POS_MODE` / `USE_DECAY` / `USE_TIME_FEATURE` / `USE_TIME_BIAS` 会随实验换，所以"直接跑"得到的是**当前那一组常量**，不等于本 README 表里的那行。表里那行（seq 512 / emb 128 / 4 层，51 秒/epoch）的权重是 `artifacts/sasrec/state_l512_d128_l4.pt`；要重跑同一配置，得先把 `EMB` 改回 128、`LAYERS` 改回 4（50 epoch 约 45 分钟）。
- **`state.pt` 每次训练都被覆写**，而且它现在装的是 2026-09-22 那批实验的权重（`POS_MODE=time_t2v`、emb 64 / 2 层）。所以 `test` 那条命令只在 checkpoint 与当前常量同配置时才成立；不匹配时守卫会直接给可读提示，不会静默算错（见 `AGENTS.md`）。

在 test 上复现上面那张表（都用 val 选出的档）：

```bash
uv run python scripts/04_popularity.py test 1.0   # 约 3 秒
uv run python scripts/05_itemknn.py test 0.5      # 约 1.5 分钟
uv run python scripts/06_sasrec.py test           # 复用 checkpoint、不重训；sasrec 那格要先用上表的配置训一次（见上面两条注意）
```

版本 B 口径（官方表口径，`val_size=0`、没有 val；见 `architecture.md`「实验流程与两套口径（A / B）」）：

```bash
uv run python scripts/02_build_splits.py b        # → artifacts/splits_b/，约 1 分钟
uv run python scripts/04_popularity.py test 1.0 b # 约 3 秒
uv run python scripts/05_itemknn.py test 0.5 b    # 约 1.5 分钟
uv run python scripts/06_sasrec.py train b        # 按当前常量（emb 64 / 2 层 / seq 512）重训 B；之后 06_sasrec.py test b 复用 state_b.pt（那份恰好就是 d64/2，能直接对上）
```

`data/`、`artifacts/`、`vendor/` 都不入库：数据可重新下载、产物可重新生成、上游代码按 clone 方式获取。

## 来源与许可

- 数据集：[Yandex Yambda](https://huggingface.co/datasets/yandex/yambda)（本仓库只用 50m 子集，不入库）。
- 官方 benchmark 代码：`vendor/yambda-benchmarks/`（上游 clone，Apache-2.0，不入库；只当参照，不接成项目依赖）。
- 仓库外的参考读物（`fun-rec` 等）只当读物，代码不进本仓库。

## 当前状态

**读表先看这条**：训练在 MPS 上**跨 session 不可复现**（同配置同 seed 差 9~11%，机制未解释；同 session 三次重复的极差是 0.85%）。所以上面表里的绝对值只能当"某一次 session 的抽样"，<1% 的差异判不了；**所有对照必须同 session 背靠背跑**。细节见 `docs/baselines.md`「跨 session 不可复现」。

已完成：数据与切分、EDA、三个基线的独立实现与官方对照、A / B 两套口径与对官方表的逐项对照、sasrec 序列长度 200→512、自写注意力 block（加性偏置；顺带修掉一个让 25% 用户在评测里失效的 NaN 缺陷）、Δ 时间特征与相对时间偏置 b[Δt桶]（都试过、都变差，开关默认关）、容量包 emb 64→128 + 2 层→4 层（val recall@100 +21.5% / test +22.4%，两个窗口都超过 itemknn）、提速三杠杆全部验伪（bf16 无收益、`torch.compile` 拒用、其余无入口）、用时间替换位置 `POS_MODE`（连续形态 val +2.7~5.1%，但只在 d64/2 上做过、且不全是严格配对）、**目标侧字段进管线**（splits 补 `is_organic` / `played_ratio_pct` / `track_length_seconds` + `feedback` / `weak_negative` / `item_artist` / `item_album` 四张表，不训练、不改数字）、**评测分层轴**（目标来源 / 参与度 / 回访×来源；结论见 `docs/baselines.md`「分层轴」）。

待办（一次只做一件，完整顺序见 `memory.md` 的快照）：① 阶段 3（模型层，一次一臂：难负样本 / 正样本加权 / `is_organic` 事件特征 / 艺人-专辑；**都在 d64/2 原始架构上做**，才能与现有 baseline 直接比）→ ② 位置通道收尾（重复 mlp / t2v、上 d128/4 验证）→ ③ 拆开 EMB 与层数归因 → ④ dropout / 早停 → ⑤ 口径 B 的容量包。另：排名融合重做（旧结论已作废）、多 seed 定量噪声、BPR 基线、likes / dislikes 交互、服务化。
