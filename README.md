# RecSys

用 Yandex **Yambda-5B** 数据集搭建一个推荐系统。做两件事：

1. **基本架构**：把一条最小可用的管线立起来——数据下载与校验 → 官方口径的时间切分与 id 重编号 → EDA → 统一评测口径 → 模型按同一口径接入。
2. **论文 / 官方 benchmark 复现**：`popularity`、`itemknn`、`sasrec` 三个基线各自己实现一份，并与官方实现逐项对照。

## 三个基线复现到什么程度

| 模型 | 我们的实现 | 与官方实现的一致性证据 | val recall@100 |
|---|---|---|---|
| popularity | `scripts/04_popularity.py` | val 数字与官方**逐项一致到 6 位小数**（含 7 档 `hour` 扫描、最优档也相同） | 0.047727 |
| itemknn | `scripts/05_itemknn.py` | 13 档 `hour` 网格跑满、最优档都是 0.5；逐用户 top-100 平均重合 **99.88/100**（4,627 人中 4,587 人完全相同） | 0.103341 |
| sasrec | `scripts/06_sasrec.py` | 把官方 checkpoint 的权重导进来喂同一批输入，前向输出**逐位相同（最大差 0；那条对照是在 `MAX_SEQ_LEN=200` / `emb 64` 下做的）**；训练侧有随机性，只能对量级 | 0.087210 |

上表的 `val recall@100` 只作复现程度的摘要；val / test 的完整指标见下一节。

## 各方法的效果（val / test）

同一份第一遍切分、同一份评测口径（候选池 627,648，不过滤已交互物品、目标里的 -1 保留为不可排名行）。

**val**（4,627 个有目标的用户）

| 方法 | recall@10 | recall@50 | recall@100 | 命中率@100 | ndcg@100 | coverage@100(有目标) |
|---|---|---|---|---|---|---|
| popularity（hour=1.0） | 0.022164 | 0.032730 | 0.047727 | 0.408472 | 0.034326 | 0.000159 |
| sasrec（50 epoch，seq 512 / emb 256） | 0.051326 | 0.062061 | 0.087210 | 0.591744 | 0.077855 | 0.034166 |
| itemknn（hour=0.5） | 0.057684 | 0.077010 | 0.103341 | 0.603847 | 0.086693 | 0.037583 |

**test**（4,599 个有目标的用户）

| 方法 | recall@10 | recall@50 | recall@100 | 命中率@100 | ndcg@100 | coverage@100(有目标) |
|---|---|---|---|---|---|---|
| popularity（hour=1.0） | 0.021764 | 0.033031 | 0.046988 | 0.388128 | 0.033241 | 0.000159 |
| sasrec（50 epoch，seq 512 / emb 256） | 0.045371 | 0.057168 | 0.079925 | 0.562296 | 0.069162 | 0.034315 |
| itemknn（hour=0.5） | 0.055089 | 0.071640 | 0.098117 | 0.579691 | 0.078911 | 0.037859 |

- **命中率@100** 就是官方那个 `ndcg@100`（官方实现有 bug，实际等于命中率）；**ndcg@100** 是正确实现。两个都报，不被官方 bug 带偏，同时保留对表能力。
- **序在两个窗口上一致**：itemknn > sasrec > popularity（recall@100 与命中率@100 都是），与官方表的相对序相同。
- **val → test 略降**（recall@100）：popularity −1.5%、itemknn −5.1%、sasrec −8.4%。test 是更靠后的窗口，而模型只用 train 训练，降幅符合预期。
- **超参只在 val 上选**：popularity hour=1.0、itemknn hour=0.5（13 档网格里选出的），test 直接用同一档，不在 test 上选参。sasrec 在 test 上复用 val 那次训练的 checkpoint，两列出自**同一个模型**。
- **重跑**：`uv run python scripts/04_popularity.py test 1.0` / `05_itemknn.py test 0.5` / `06_sasrec.py test`。三个基线在 test 上的排序只由训练集决定（用户向量与 top-100 与评测窗口无关），所以换 split 只换评测目标。
- 逐档 dcg、两个对照口径（过滤已交互 / 只看可排名目标）、`hour` 网格，见 `docs/baselines.md`。
- **coverage 有两种用户集合**（官方自己就不一致）：表里那一列是 `coverage@100(有目标)`；另有 `coverage@100(全用户)`——口径 A 下 popularity 0.000159、sasrec **0.040937**、itemknn **0.064442**（口径 B 见 `docs/baselines.md`）。它只由训练集和模型决定、**与评测窗口无关**，所以 val / test 同值。官方 popularity / itemknn 用全用户口径、官方 sasrec 用有目标用户口径，我们两条都报，两边都能对表。
- **回访 vs 新歌（与排名融合）**：官方指标里 **66.6% 的目标是“回访”**（训练期已听过的歌），拆开看**两条轴的排序正好相反**——回访 itemknn > sasrec > popularity，新歌 popularity > sasrec > itemknn（口径 A test，recall@100：回访 0.1272 / 0.1022 / 0.0484，新歌 0.0246 / 0.0308 / **0.0448**）。把 itemknn 与 sasrec 做 RRF 融合能明显超过 itemknn（**0.1081 vs 0.0981，+10.2%**）；三路融合（加 popularity）把新歌拉到 **0.0444**，代价是“全部”掉到 0.1018。`04/05/06` 每次运行都会打印这两桶，细节见 `docs/baselines.md`。

官方 benchmark 的 test 表（50m / listens，recall@100）：random 0.00006 → **popularity 0.0479** → **sasrec 0.0828** → **itemknn 0.1085** —— 相对序与我们一致。它用的是**第二遍训练集**（`val_size=0`，train 延伸到 test 前 30 分钟），与上表的口径不同、**不能并排比**。这套协议已经走通：同一口径下 popularity 与官方**六位小数全中**（含 dcg 与 coverage）、itemknn 差 ≤5.3e-5、sasrec 低 7.2%（口径 B、seq 512；单 seed 的训练随机性 + 两处已记差异），见 `docs/baselines.md` 的“版本 B 口径”一节。

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
uv run python scripts/06_sasrec.py         # 50 epoch：seq 512 / emb 256 自报约 47 分钟（墙钟可能因机器 Idle Sleep 拉长；MPS 训练、CPU 推理）
```

在 test 上复现上面那张表（都用 val 选出的档）：

```bash
uv run python scripts/04_popularity.py test 1.0   # 约 3 秒
uv run python scripts/05_itemknn.py test 0.5      # 约 1.5 分钟
uv run python scripts/06_sasrec.py test           # 约 30 秒，复用 val 那次存的 checkpoint
```

版本 B 口径（官方表口径，`val_size=0`、没有 val；见 `architecture.md`「实验流程与两套口径（A / B）」）：

```bash
uv run python scripts/02_build_splits.py b        # → artifacts/splits_b/，约 1 分钟
uv run python scripts/04_popularity.py test 1.0 b # 约 3 秒
uv run python scripts/05_itemknn.py test 0.5 b    # 约 1.5 分钟
uv run python scripts/06_sasrec.py train b        # 按当前 MAX_SEQ_LEN（512）重训 B；之后 06_sasrec.py test b 复用 state_b.pt
```

`data/`、`artifacts/`、`vendor/` 都不入库：数据可重新下载、产物可重新生成、上游代码按 clone 方式获取。

## 来源与许可

- 数据集：[Yandex Yambda](https://huggingface.co/datasets/yandex/yambda)（本仓库只用 50m 子集，不入库）。
- 官方 benchmark 代码：`vendor/yambda-benchmarks/`（上游 clone，Apache-2.0，不入库；只当参照，不接成项目依赖）。
- 仓库外的参考读物（`fun-rec` 等）只当读物，代码不进本仓库。

## 当前状态

已完成：数据与切分、EDA、三个基线的独立实现与官方对照、A / B 两套口径与对官方表的逐项对照、sasrec 序列长度 200→512（dcg 与 coverage 一致变好、recall@100 不稳）、回访/新歌两桶与排名融合对照（d256 下融合超 itemknn +10.2%）、sasrec 扩容 emb 64→256（val +5.7% / test +7.4% recall@100）。

待办：BPR 基线、likes / dislikes 交互、服务化。见 `memory.md`。
