# 复现官方 benchmark

> 上游 `yandex/yambda`（Apache-2.0）的 `benchmarks/` 代码在本机落地、各模块用途与已验证结果。最后更新：2026-09-19。

## 上游版本与获取方式

- commit：`dd6f3a19eef5866e346c3270e098baa641a44948`（HF 上 lastModified 2026-04-06）
- 位置：`vendor/yambda-benchmarks/`（已在 `.gitignore`）

```bash
cd /Users/yukuanzou/workspace/RecSys
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://huggingface.co/datasets/yandex/yambda vendor/yambda-benchmarks
cd vendor/yambda-benchmarks && git sparse-checkout set benchmarks
```

- `GIT_LFS_SKIP_SMUDGE=1`：parquet 都是 LFS 对象，跳过 smudge 只留下指针，376 KB 拿到代码。
- **不要加 `--filter=blob:none`**：HF 的 promisor fetch 不支持，`git sparse-checkout set` 会报 `fatal: expected 'packfile'`。
- 顶层残留的 `embeddings.parquet` / `album_item_mapping.parquet` / `artist_item_mapping.parquet` 是 LFS 指针文件，不是数据。

## 环境

独立 env，不碰项目根的 `.venv`（项目 env 是 pandas/pyarrow/numpy，无 torch）：

```bash
cd vendor/yambda-benchmarks/benchmarks
uv sync
```

得到 Python 3.12.14（沿用根目录 `.python-version`）+ torch 2.14.0 + polars-u64-idx 1.28.1 + optuna 5.0.0 + scipy + matplotlib，`yambda` 包以 editable 装进该 env。

## 运行方式

官方脚本假定数据在 `<data_dir>/<size>/<interaction>.parquet`，我们的是 `data/raw/flat/50m/`，所以传绝对路径；所有脚本默认 `--device cuda:0`，需显式指定：

```bash
DATA=/Users/yukuanzou/workspace/RecSys/data/raw/flat
cd vendor/yambda-benchmarks/benchmarks/models/<model>
uv run python main.py --data_dir $DATA --size 50m --interaction listens --device cpu
```

SASRec 的数据口径不同（需要 sequential 数据），并且分成 train / eval 两个脚本，见下。

## 数据准备

- `flat/50m/listens.parquet`（已有）
- `sequential/50m/listens.parquet`：只有 sasrec 需要，用官方脚本从 flat 生成（1.1 秒，426 MB）：

```bash
uv run python scripts/transform2sequential.py \
  --src_dir /Users/yukuanzou/workspace/RecSys/data/raw/flat/50m \
  --dst_dir /Users/yukuanzou/workspace/RecSys/data/raw/sequential/50m \
  --files listens.parquet --aggregation columns
```

用 `--aggregation columns` 才会得到 `item_id` / `timestamp` 单数列名，正是 `models/sasrec/data.py` 期望的（HF 卡片上写的 `item_ids` / `timestamps` 复数名与代码不符）。

## 本机（Apple Silicon arm64 / 24 GB / 无 CUDA，MPS 可用）可行性

| 模型 | 状态 | 说明 |
|---|---|---|
| random_rec | 已跑通 | 秒级 |
| popularity | 已跑通 | 秒级 |
| itemknn | 已跑通 | 官方 13 档小时网格耗时 35 分 45 秒 |
| sasrec | 已跑通 | MPS 训练；**eval 必须用 CPU**（见偏差） |
| bpr_als | 跑不了 | 代码 import `implicit.gpu.als` / `implicit.gpu.bpr`，GPU 版要 CUDA |
| sansa | 跑不了 | 官方自述 50m+listens 需 ≥100 GB RAM，另需 libsuitesparse + 编译 glami/sansa |

## 基线结果（50m / listens / test split）

| 模型 | recall@10 | recall@50 | recall@100 | ndcg@10 | ndcg@50 | ndcg@100 | coverage@100 |
|---|---|---|---|---|---|---|---|
| random_rec（重复 2 次平均） | 0.000033 | 0.000034 | 0.000062 | 0.000326 | 0.001522 | 0.002827 | 0.76852 |
| popularity（hours 网格选 `best_hour=1.0`） | 0.022555 | 0.033143 | 0.047866 | 0.134377 | 0.299848 | 0.390302 | 0.000159 |
| sasrec（50 epoch，MPS） | 0.046308 | 0.060697 | 0.082849 | 0.264188 | 0.475103 | 0.571646 | 0.015887 |
| itemknn（hours 网格选 `best_hour=0.5`） | 0.063051 | 0.079791 | 0.108476 | 0.282888 | 0.502935 | 0.613394 | 0.064990 |

dcg（sasrec 的 eval.py 不上报 dcg）：

| 模型 | dcg@10 | dcg@50 | dcg@100 |
|---|---|---|---|
| random_rec | 0.000111 | 0.000364 | 0.000575 |
| popularity | 0.101055 | 0.192947 | 0.256216 |
| itemknn | 0.277178 | 0.491950 | 0.629622 |

来源：`vendor/yambda-benchmarks/benchmarks/models/{random_rec,popularity,itemknn,sasrec}`。

**读表注意**：`ndcg@k` 在这份实现里等于「top-k 至少命中一次的用户占比」（见下文 NDCG 实现事实），不是通常意义上的 NDCG，不可与论文数字直接比较。

## 一致性核对（已通过）

1. **切分测试**：官方 `tests/test_timesplit.py` 通过（flat 与 sequential 切分等价）。
2. **评测用户数**：官方 popularity 的 target mask 显示 4,627（val）/ 4,599（test），与独立统计完全一致——val/test 各有恰好 1 个用户不在训练集里，被官方切分滤掉。
3. **候选池大小**：random_rec 打印 `NUM_USERS 9208, NUM_ITEMS 629298`；从 popularity 的 `coverage@100 = 100 / num_item_ids` 反推同样是 629,298。
4. **切分行数守恒**：Listen+ 共 29,439,278 行 = train 29,135,186 + val 143,562 + test 157,696 + 两个 gap 窗口 2,834，差额为 0。
5. **本机数据基线**：raw listens 46,467,212 行；Listen+ 占比 63.36%；训练窗口截止 25,823,600。

## 模块用途

**库 `yambda/`**

| 模块 | 用途 |
|---|---|
| `constants.py` | 全局口径：`GAP_SIZE=1800s`、`VAL_SIZE=86400s`、`TEST_SIZE=86400s`、`LAST_TIMESTAMP=26,000,000` → `TEST_TIMESTAMP=25,913,600`；`TRACK_LISTEN_THRESHOLD=50`（Listen+ 阈值）；`NUM_RANKED_ITEMS=100`；`METRICS`（默认上报的 12 个指标）。所有模型都从这里取口径，改口径只改这一处 |
| `processing/timesplit.py` | 官方时间切分。`flat_*` 处理逐行表，`sequential_*` 处理按 uid 聚合的列表表，二者语义等价。规则：train `[0, test−gap−val−gap)`、val `[test−val−gap, test−gap)`、test `[test, ∞)`；只保留训练集出现过的 uid；`drop_non_train_items` 默认 False（test 里可能有训练集没出现过的物品） |
| `processing/chunk_read.py` | 把 polars LazyFrame 包成 torch IterableDataset（分片、多 worker 划分、默认 DataLoader 配置）。6 个基线都没用它，是面向大规模训练的通用件 |
| `evaluation/ranking.py` | 评测的「打分」半边：`Embeddings`（id + 向量表）、`Targets`（每个用户的 val/test 物品集合）、`Ranked`（排序结果）、`rank_items`（按 batch 全量内积 + topk）。注意 `rank_items` 的 `num_items` 参数是 top-k 的 k，不是物品总数 |
| `evaluation/metrics.py` | 评测的「算指标」半边：`create_target_mask` 命中矩阵、recall / precision / mrr / dcg / ndcg / coverage、`cut_off_ranked`（把 ranked 裁到有 target 的用户上，所以分母是 4,599 而不是 9,208） |
| `utils.py` | 指标字典的加 / 除 / 平均（多随机种子、多次重复聚合）与 `argmax`（在超参候选里挑最优） |

**脚本 `scripts/`**

| 脚本 | 用途 |
|---|---|
| `transform2sequential.py` | flat → sequential，按 uid 聚合成长列表。`--aggregation columns` 保留原列名（sasrec 需要），`structs` 把整行塞进一个 events 结构体 |
| `make_multievent.py` | 把 5 个事件文件合并成 `multi_event.parquet`（HF 上默认发布的那个文件） |
| `get_dataset_stats.py` | 数据集统计 + 三张分布图（用户历史长度、对数版、物品交互数），数据集卡片里的图就是它画的 |

**模型 `models/`**

| 模型 | 用途 |
|---|---|
| `random_rec` | 下界：随机抽 100 个物品当推荐，默认重复 2 次取平均，标定「完全不学」的水平 |
| `popularity` | 按物品加权交互次数排序。`hour` 是衰减参数（`tau = 0.9 ** (1/86400/(hour/24))`，权重 `tau ** (最后时间−交互时间)`），`hour=0` 表示不衰减；官方在 val 上扫 7 档选最优 |
| `itemknn` | 物品表示成「用户频率向量」，用户表示成历史物品向量的时间衰减加权和，余弦相似度排序；`hour` 同为衰减参数，官方在 val 上扫 13 档。因为要建用户×用户稠密矩阵，官方不支持 5b |
| `sasrec` | 序列模型：item embedding + 位置 embedding（倒序，最新为 0）→ 2 层 Transformer（因果掩码）→ 取最后位置向量；训练目标是「每个位置的下一首歌」对 1 个随机负样本的 BCE；评测时取训练序列末向量与全部物品内积排序。数据用 sequential 格式，train/eval 分成两个脚本 |
| `bpr_als` | 经典矩阵分解（BPR / ALS），底层用 `implicit.gpu`，本机无 CUDA 跑不了 |
| `sansa` | 稀疏线性自编码器（EASE 类），需 SuiteSparse + 编译 glami/sansa，50m 就需 100 GB RAM，本机跑不了 |

## 已确认的上游实现事实（影响解读）

- **NDCG 恒等于「命中率@k」**：`yambda/evaluation/metrics.py:167` 是 `ideal_dcg = calc_dcg(target_mask)`，算好的 `ideal_target_mask` 没被使用 → 真实 DCG 除以真实 DCG。所以表里的 `ndcg@k` 实际含义是「top-k 里至少命中一个 target 的用户占比」。
- **recall 分母是 `min(num_positives, k)`**（`torch.clamp(num_positives, max=k)`），而同行注释写的是 `max(num_positives, k)`，注释与代码不一致。
- **指标只统计有 target 的用户**（`cut_off_ranked`），分母是 4,599 而不是 9,208。
- **两段式流程**：先用 `val_size=1 天` 的短训练集在 val 上选超参，再用 `val_size=0`（训练集向后延伸到 test 前 30 分钟）重训一次在 test 上报告。
- **全量排序，不做已交互过滤**；候选池 = 训练期出现过的物品（sasrec 的口径是全部 Listen+ 物品 + padding 行 0，即 631,004 vs 其他模型的 629,298，差 0.27%）。
- **sasrec 的训练脚本没有验证集、没有早停**，跑满 `num_epochs` 后存最后状态。

## 本机偏差与限制

1. `bpr_als`、`sansa` 未跑（硬件限制，见表）。bpr_als 的折中方案是改用同库 `implicit.cpu`，属于改官方代码，需单独记录。
2. `sasrec` 用 **50 epoch**（官方默认 100）。
3. `sasrec` **训练在 MPS、评测在 CPU**：`eval.py` 在 MPS 上会崩——`nn.TransformerEncoder` 推理路径调用了 MPS 未实现的 `aten::_nested_tensor_from_mask_left_aligned`（训练路径不触发）。
4. `pin_memory_device="cuda"` 在 torch 2.14 只是 deprecation warning，会自动改用当前加速器，**无需改官方代码**。
5. 所有模型显式传 `--device`；数据放 `data/raw/` 而非官方假定的 `data/`，用 `--data_dir` 指过去，不改代码。
6. 中间有过一次机器休眠，SASRec 训练的墙上时间被拉长（纯计算约 18 分钟）。

## 未做

- `likes` 交互：官方六个脚本都支持 `--interaction {likes,listens}`（代码差异只有一行：只有 `listens` 才做 `played_ratio_pct >= 50` 过滤），但**本仓库没有归档任何 likes 数字**。（此处原先写成"popularity 之外的模型没有跑 likes"，读起来像 popularity 跑过，与归档不符；2026-09-20 更正。）
- 500m / 5b 规模未跑。
- 官方 `scripts/make_multievent.py`、`get_dataset_stats.py` 未运行（用不到）。
