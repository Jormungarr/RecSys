# RecSys

用 Yandex **Yambda-5B** 数据集搭建一个推荐系统。做两件事：

1. **基本架构**：把一条最小可用的管线立起来——数据下载与校验 → 官方口径的时间切分与 id 重编号 → EDA → 统一评测口径 → 模型按同一口径接入。
2. **论文 / 官方 benchmark 复现**：`popularity`、`itemknn`、`sasrec` 三个基线各自己实现一份，并与官方实现逐项对照。

## 三个基线复现到什么程度

| 模型 | 我们的实现 | 与官方实现的一致性证据 | val recall@100 |
|---|---|---|---|
| popularity | `scripts/04_popularity.py` | val 数字与官方**逐项一致到 6 位小数**（含 7 档 `hour` 扫描、最优档也相同） | 0.047727 |
| itemknn | `scripts/05_itemknn.py` | 13 档 `hour` 网格跑满、最优档都是 0.5；逐用户 top-100 平均重合 **99.88/100**（4,627 人中 4,587 人完全相同） | 0.103341 |
| sasrec | `scripts/06_sasrec.py` | 把官方 checkpoint 的权重导进来喂同一批输入，前向输出**逐位相同（最大差 0）**；训练侧有随机性，只能对量级 | 0.074572 |

上表的数字都在第一遍切分的 **val** 上（候选池 627,648，4,627 个有目标的用户）；每个基线的 `hour` 网格、dcg / 命中率 / coverage、与官方 test 表的量级对照，见 `docs/baselines.md`。

官方 test 表（50m / listens，recall@100）：random 0.00006 → popularity 0.0479 → sasrec 0.0828 → itemknn 0.1085 —— **相对序与我们一致**，绝对值各自略高（官方用第二遍训练集、评测集是 test，不能直接比）。

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
uv run python scripts/06_sasrec.py         # 50 epoch 约 5.5 分钟（MPS 训练、CPU 推理）
```

`data/`、`artifacts/`、`vendor/` 都不入库：数据可重新下载、产物可重新生成、上游代码按 clone 方式获取。

## 来源与许可

- 数据集：[Yandex Yambda](https://huggingface.co/datasets/yandex/yambda)（本仓库只用 50m 子集，不入库）。
- 官方 benchmark 代码：`vendor/yambda-benchmarks/`（上游 clone，Apache-2.0，不入库；只当参照，不接成项目依赖）。
- 仓库外的参考读物（`fun-rec` 等）只当读物，代码不进本仓库。

## 当前状态

已完成：数据与切分、EDA、三个基线的独立实现与官方对照。

待办：官方"第二遍训练集"口径（才能产出与官方表格并排的 test 数字）、BPR 基线、likes / dislikes 交互、服务化。见 `memory.md`。
