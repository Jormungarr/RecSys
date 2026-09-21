# architecture.md — 管线与口径

> 项目档案：数据怎么流、口径怎么定、为什么这么设计。
> 进度与决策见 `memory.md`；协作约定见 `AGENTS.md`；数据集事实见 `docs/dataset_notes.md`。
> 最后更新：2026-09-21。

## 数据流

```
data/raw/flat/50m/listens.parquet        46,467,212 行（原始 uid / item_id，都不连续）
        │  scripts/02_build_splits.py
        │  ① Listen+ 过滤  ② 官方 GTS 时间切分  ③ 只留训练集 uid  ④ 重编号
        ▼
artifacts/splits/{train,val,test}.parquet        + uid_map / item_map
        │  scripts/03_eda.py   →  artifacts/eda/*.png + docs/eda.md
        ▼
scripts/04_popularity.py（打分 → 全局 top-100 → 评测）      → 终端数字 + docs/baselines.md
scripts/05_itemknn.py（物品表示 + 用户表示 → 内积排序 → 评测）→ 同上
scripts/06_sasrec.py（训练 → 序列末向量 × 物品表 → 评测）    → 同上 + artifacts/sasrec/state.pt
        （评测实现在 scripts/rec_eval.py，三个模型共用；之后：BPR 按同一口径接入）
```

## 数据口径（唯一来源：`scripts/02_build_splits.py`）

- 正样本：Listen+ = `played_ratio_pct >= 50`（官方 `TRACK_LISTEN_THRESHOLD`）。
- 时间切分（官方 `processing/timesplit.py`，`val_size = VAL_SIZE`）：
  - train `timestamp < 25,823,600`
  - val `[25,825,400, 25,911,800)`
  - test `timestamp >= 25,913,600`
  - 两段 gap 各 1,800 秒；**只保留训练集出现过的 uid**；物品**不做**过滤。
- 规模：train 29,135,186 行 / val 143,557 行 / test 157,695 行；有目标的用户 4,627 / 4,599。

## id 空间与 `-1` 约定

- 原始 id 稀疏（uid 9,238 个散布在 100 ~ 1,000,000；item 877,168 个散布在 22 ~ 9,390,623）。embedding 表与评分矩阵要稠密索引，**必须重编号**。
- 重编号基准 = **训练集**：uid → `0..9,206`、item → `0..627,647`；候选池就是全部 `0..627,647`。
- val / test 里"训练集没出现过的物品"记为 **`-1`**：评测时算"必然未命中"；官方等价物是 `drop_non_train_items=False`（条目保留，但这些物品本来也排不到）。
- 反向映射：`artifacts/splits/uid_map.parquet`（`raw_uid` ↔ `uid`）、`item_map.parquet`。
- 这两张表**只属于 listens（Listen+）这一套切分**。likes / dislikes 是**另一套实验**：官方给每种交互各自算 id 空间（`timesplit` 只保留该交互自己训练集里出现过的 uid），所以不能直接复用这里的映射——实测 likes 的 8,283 个用户里只有 7,512 个、181,304 个曲目里只有 166,233 个落在 listens 训练集的 id 空间内（85% 的 likes 行两者都在）。

## 评测口径（唯一来源：`scripts/rec_eval.py` 的实现 + docstring）

- 调用约定：`tops` 要按 **uid 行序**给（不是 val 的行序），`chunks` 按 val 分组序 —— 必须 `tops[val_users]` 对齐。popularity 因为人人共享同一份 top 才没暴露这个坑，itemknn 第一次就撞上了。
- 候选 = 全部训练物品（候选池 627,648）；**不做 per-user 已交互过滤**（官方口径），另出一份"过滤已交互"对照。
- 指标定义（对齐官方 `yambda/evaluation/`）：
  - `recall@k = 命中数 / min(目标行数, k)` —— 分母是**行数**，不是去重物品数
  - `dcg@k = Σ(命中位次) 1 / log2(位次 + 2)`
  - `coverage@k = top-k 里出现过的不重复物品数 / 候选池大小` —— **两种用户集合都报**（见下一条）
  - 求平均的用户集合 = 有目标的用户（recall / dcg / ndcg / 命中率）
- `coverage@k` 的**用户集合**在官方内部就不一致：`Coverage(cut_off_ranked=False)` 用全部用户（`popularity` / `itemknn` 的 `Ranked` 就是训练集全部用户），官方 `sasrec` 的 eval 却把用户 inner join 到测试集、只剩有目标的用户。所以我们两条都报——`coverage(全用户)` 对 popularity / itemknn、`coverage(有目标)` 对 sasrec，两边都能对表。`coverage(全用户)` 与评测窗口无关（top-k 只由训练集和模型决定），同一份权重下 val / test 同值。
- **回访 / 新歌两桶**：官方口径不过滤已交互物品，而目标里大部分是“用户训练期已经听过的歌”（口径 A test 占 **66.6%**：回访 105,069 行 / 新歌 52,626 行），不拆开就看不出模型赢在哪条轴上。`rec_eval.evaluate` 接受 `history`（每用户在训练期的物品集合）时会额外报 `回访 recall / 命中率` 与 `新歌 recall / 命中率`（回访 = 目标物品在该用户训练期历史里出现过）。两桶都用**逐用户**口径（分母 `min(桶内行数, k)`，只在桶内有目标的用户上平均），与主表可比；`04/05/06` 的**主表**都会传它。
- 官方 `ndcg@k` 是 `real_dcg / real_dcg`（已知 bug）→ 实际含义 = **命中率@k**。我们同时报"正确 NDCG"和"命中率"：对官方表格用后者，比较模型用前者。
- **两遍口径**：官方协议本身就是三段切分（`constants.py` 的 `VAL_SIZE = 1 天`）——第一遍用短训练集在 val 上选超参，第二遍用 `val_size=0` 的长训练集（val 窗口并回训练集，延伸到 test 前 30 分钟，候选池 629,298）报告 test。**两遍都已产出**：`04/05` 第 3 个参数、`06` 第 2 个参数就是口径（`a` / `b`，默认 `a`）。口径 A 的数字在各模型一节，可对官方表的口径 B 见「版本 B 口径」一节。

## 实验流程与两套口径（A / B）

所有模型按同一套流程接入。口径一旦发布就不再改：新模型只能往表里加行，中途改口径则前面所有的行全部作废。

| 步 | 做什么 | 输入 | 输出 | 约束 |
|---|---|---|---|---|
| 1 | 固定公共 val（官方第一遍窗口，4,627 个有目标的用户） | train 窗口 | val | 所有模型共用，不各自另切 |
| 2 | 扫超参 / 选配置 / 定 seed 规则 | train → val | 每个模型一份冻结的配置 | 只看 val 指标 |
| 3 | 在 val 上排名，决定保留哪几个模型（可选） | val | 候选名单 | 只在 val 上 |
| 4 | 用冻结的配置跑 test（版本 A 或 B） | train / train ∪ val | 两张表 | test **只读一次** |
| 5 | 总结归档 | — | `docs/baselines.md` + `memory.md` | 数字带口径标注 |

- **公共 val**：val 是 test 前 1 天、两侧各 30 分钟 gap 的**整段时间窗**，不允许随机切，也不允许在 train 里随机挖一块（时间结构决定了随机切会泄漏未来，见 `docs/eda.md`）。某个模型有硬需求要换 val 时单独记录，且该行的 val 数字不与其它行并排比。
- **val 上选超参 / 选模型 / 选 seed 都算调参**，只允许发生在 val；test 上不做任何决策，也不「多跑几次挑最好的一次」报出来。
- **带随机性的模型**按 val 上定好的 seed 规则跑 test（官方 `random_rec` 就是重复 2 次取平均，见 `docs/benchmark_repro.md`）。

| 口径 | 训练集 | 评测窗口 | 用途 | 能否对官方表 |
|---|---|---|---|---|
| 版本 A | 第一遍 train（截止 25,823,600，候选池 627,648） | 第一遍的 test 窗口 | 日常迭代与选型（sasrec 一轮 5.5 分钟） | 不能 |
| 版本 B | `val_size=0` 的长训练集（延伸到 25,911,800，候选池 629,298） | 官方 test 窗口 | 定型后对外可比 | 能 |

- 版本 B 没有 val，超参**沿用版本 A 在 val 上选出的**；B 的 id 空间与 A 不同（物品 627,648 → 629,298、用户 9,207 → 9,208），**另存一套 artifacts**，不覆盖第一遍。
- 官方 SASRec 脚本（`models/sasrec/train.py` / `eval.py`）两遍都用 `val_size=0`：它没有超参、也没有早停，所以不产出 val。我们的 `06_sasrec.py` 给它配了 val，是为了让三个基线共用同一套选参流程。

## 目录与命名

| 路径 | 说明 |
|---|---|
| `scripts/NN_*.py` | 按序号的管线脚本（`01` 看数据 / `02` 切分 / `03` EDA / `04` popularity / `05` itemknn / `06` sasrec）；朴素 `print`，可反复重跑 |
| `scripts/rec_eval.py` | 评测库（各模型共用）；不编序号，因为它不是可执行脚本 |
| `docs/*.md` | 专题结论：`dataset_notes.md`、`benchmark_repro.md`、`eda.md`、`baselines.md` |
| `AGENTS.md` | 只放 agent 相关：协作方式、工作要求、环境、约定、索引 |
| `memory.md` | 进度、关键决策与理由、数据与产物状态、杂项 |
| `architecture.md` | 本文档 |
| `data/`、`artifacts/`、`vendor/` | 一律 gitignore：原始数据 / 产物 / 上游代码 |

## 为什么这么设计

- **重编号跟训练集走**：既满足模型对稠密 id 的需求，又让"评测候选池 = id 空间"，评测代码不用额外维护候选 mask。
- **按时间切分**：随机切会把未来行为混进训练，离线指标虚高；用官方 GTS 才能和官方数字对表。
- **两遍口径**：官方表格就是这么算出来的（选超参用短训练集、最终报告用长训练集），要并排比较就得复刻。
- **评测自己实现**：官方包依赖 torch + polars、且读 raw 数据自带切分；我们的评测实现只用 numpy / pandas，模型实现按需引入依赖（itemknn 用 scipy、sasrec 用 torch），锁住自己的 artifacts 与口径，并在 val 上与官方逐项核对过（popularity 对到 6 位小数）——既有单一来源，又不失可比性。
- **抽公共模块的时机**：评测一直内联在 `04_popularity.py` 里，等到第二个模型（itemknn）真的要复用时才抽成 `rec_eval.py`，抽完拿 popularity 当回归测试（数字必须一字不变）。
- **自己实现就要能和官方逐项对照**：popularity 对到 6 位小数（验口径）；itemknn 进一步比逐用户 top-100（平均重合 99.88/100）——后者能定位到实现里具体哪一步写错了，itemknn 的两个 bug 都是这么抓出来的。
- **训练型模型的验证边界**：sasrec 带随机性（初始化 / 负采样 / shuffle），数字不可能逐位复现；能钉死的是模型定义——把官方 checkpoint 的权重导进来、喂同一批输入比前向，实测差为 0。
- **口径集中在代码里**（脚本 docstring + 实现），文档只写结论与指引，避免多处复制后漂移。
