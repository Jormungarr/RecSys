# memory.md — 项目进度与决策

> 项目档案：进度、关键决策与理由、数据与产物状态、杂项。
> 管线与口径见 `architecture.md`；数据集事实见 `docs/dataset_notes.md`；协作约定见 `AGENTS.md`。
> 逐字对话不存仓库（Kun 的线程存档在 `~/.kun/data/threads/`），这里只留提炼后的要点。
> 最后更新：2026-09-20。

## 进度

**已完成**

1. 数据下载 + sha256 校验：`data/raw/`（7 个文件与 HF 的 LFS 哈希逐一核对通过；**未下载** `embeddings.parquet`，13.8 GB，等做内容特征再议）
2. 字段含义、评测协议、显式反馈语义从官方源头确认并归档 → `docs/dataset_notes.md`
3. `scripts/01_inspect_listens.py`：数据样式与字段含义
4. 官方 benchmark 落地：`vendor/yambda-benchmarks/`（commit `dd6f3a19…`，独立 env，已跑通 random_rec / popularity / itemknn / sasrec）→ `docs/benchmark_repro.md`
5. 数据切分：重编号 + Listen+ + 官方时间切分 → `artifacts/splits/`（`scripts/02_build_splits.py`）
6. EDA：分布 / 长尾 / 时间结构，6 张图 → `scripts/03_eda.py`、`docs/eda.md`、`artifacts/eda/`
7. popularity 基线 + 评测口径 → `scripts/04_popularity.py`、`docs/baselines.md`
8. 评测抽公共模块 → `scripts/rec_eval.py`（`04_popularity.py` 改用后重跑，val 数字一字未变）
9. itemknn 基线：官方 13 档 hour 网格跑满，最优 hour=0.5 → `scripts/05_itemknn.py`、`docs/baselines.md`
10. sasrec 基线：自己实现（torch 2.14.0），50 epoch / 329.8 秒，前向与官方 checkpoint 逐位一致 → `scripts/06_sasrec.py`、`docs/baselines.md`

**待办**

- 官方"第二遍训练集"口径（`val_size=0`，train 延伸到 test 前 30 分钟，候选池 629,298）→ 才能产出与官方表格并排的 test 数字
- 基线：BPR（官方的 `bpr_als` 在本机跑不了——`implicit.gpu` 要 CUDA；要做得换 CPU 后端，属于改官方代码）
- likes / dislikes 还没进管线（各自语义与时间轴不同，见 `docs/dataset_notes.md`）
- 服务化（最后一步）

## 关键决策与理由

| 决策 | 理由 / 出处 |
|---|---|
| 正样本 = Listen+（`played_ratio_pct >= 50`） | 官方 FAQ 与各模型 `preprocess` 都用它；不筛会把"点开就划走"当正反馈 → `docs/dataset_notes.md` |
| 按官方 GTS 时间切分，不随机切 | 随机切会泄漏未来、指标虚高；边界见 `architecture.md` → `vendor/.../processing/timesplit.py` |
| 只保留训练集出现过的 uid | 官方口径；val / test 各掉 1 个用户（5 行 / 1 行） |
| 重编号 = **训练集口径**（uid 0..9,206、item 0..627,647） | 让"评测候选池"天然等于 id 空间，评测不用额外维护 mask；与官方"候选池 = 训练期物品"一致。代价：val/test 里 1,928 / 2,332 行目标记为 `-1` |
| `-1` = 训练集没出现过的物品，评测时算"必然未命中" | 等价于官方 `drop_non_train_items=False`；影响已量化：recall@100 相对 +1.1% |
| 评测默认**不过滤已交互物品**，另出一份过滤对照 | 与官方可比；过滤后 val 只剩 3,453 / 4,627 个用户还有"新歌"目标 → 官方指标的"回访"成分很重 |
| 官方 `ndcg` 有 bug（等于命中率），我们并行报"正确 NDCG"和"命中率" | 不被官方 bug 带偏，同时保留对表能力；用户要求"做评测时用正确实现" |
| popularity 的 `hour` 在 val 上扫 7 档（官方网格） | 数据非平稳（活动量 300 天近翻倍），"往前看多久"是真超参；最优 `hour=1.0` |
| 打分与评测自己实现（numpy/pandas），官方包只当参照 | 官方包要 torch + polars、自带另一套切分；已在 val 上与官方逐项核对（6 位小数一致） |
| 不把 `vendor/` 或参考读物接成项目依赖 | `vendor/` 已 gitignore；参考读物明确"只当读物，不抄代码" |
| itemknn 自己实现（numpy + scipy），官方包只当参照 | 与 popularity 同思路；新引入 scipy 是为了 `A = W @ Cᵀ` 的稀疏乘与按物品累加外积（纯 numpy 要么写 2900 万次循环、要么 5e10 次无序累加）→ `scripts/05_itemknn.py` |
| 省掉官方对 `A` 的按行 L2 归一 | 逐行除以一个正常数不改变该行排序，而指标只看 top-k |
| 与官方的一致性靠"同数据逐用户 top-100 对照"，不靠对表 | 官方脚本只打印 `best_hour`、不打印逐档 val 指标；逐用户对照更强，且一档只要 70 秒 |
| 评测函数抽成 `scripts/rec_eval.py`（不参与序号） | 三个模型共用同一份实现；无序号是因为它是库不是可执行脚本 |
| sasrec 自己实现（torch 2.14.0，与 vendor 同版本） | 与前两个基线同思路；torch 到这一步才引入 → `scripts/06_sasrec.py` |
| sasrec 的 padding 用 id 627,648、排序时排除 | 官方的 0 在我们的 id 空间里是真实物品；排除后候选池 627,648，三个基线可比 |
| sasrec 的位置编号固定按左填充到 200 | 官方随 batch 内最大长度平移（eval 还 `shuffle=True`）→ 同一用户换 batch 结果会变；满长度时我们与官方逐位一致 |
| 训练型模型的验证用"导官方权重喂同一批输入比前向" | 训练有随机性（初始化/负采样/shuffle），逐位对不了；但模型定义这一层可以钉死（实测差 0） |

## 本轮（2026-09-20）要点

- itemknn 基线跑通：val 上 hour=0.5，命中率@100 = 0.603847、recall@100 = 0.103341、coverage@100 = 0.037583（popularity 的 236 倍）；13 档网格全部落在 0.585~0.604，**最优 hour 与官方在 val 上选出的 0.5 一致**。
- 与官方实现逐用户对照（同一份第一遍切分、hour=0.5）：top-100 平均重合 99.88/100，4,627 人中 4,587 人完全相同；`A` 的元素和相对差 7e-8。
- 修掉两个自己写错的点（都靠逐项对照才发现）：① 物品表示的用户范数按 CSR 的列号取了，而 CSC 的 `indices` 是行号（uid）；② `A = W @ Cᵀ` 误把 C 也限制到 W 的支撑上（应对 C 的**整列**求和）。
- 一个新观察：官方的选参指标（坏 `ndcg` = 命中率）选出 hour=0.5，而 `recall@100` 的最大值在 hour=0.064（高 1.1%）——选参指标会影响结论（原因未解释）。
- 新依赖：scipy（`uv add scipy`）。
- sasrec 基线跑通：`uv add torch==2.14.0`（与 vendor 同版本）；50 epoch 用 329.8 秒（每 epoch 4.9 → 后段 ~7.0 秒，变慢原因未解释），val recall@100 = 0.074572 —— popularity 0.0477 < sasrec 0.0746 < itemknn 0.1033，**相对序与官方 test 表一致**。
- 前向对照：用官方 checkpoint 喂同一批 200 条满长度序列，两边输出**逐位相同（最大绝对差 0.000e+00）**；评测那一半复用 `rec_eval.py`，没重复验证。
- 成本修正：`docs/benchmark_repro.md` 记的"官方 50 epoch 约 18 分钟"，我们实测 5.5 分钟；差异原因**未解释**（官方那份用了 `num_workers=3` + prefetch + pin_memory，且当时机器休眠过）。

## 本轮（2026-09-19）要点

- 项目从"裸 `requirements.txt`"改成 **uv 项目**：新增 `pyproject.toml` / `uv.lock`（`uv add -r requirements.txt`），`uv add` 现在可用；`requirements.txt` 保留未删。
- 本机装了 `codebase-memory-mcp`（代码记忆，用法见 `AGENTS.md`）；给本仓库建了索引（`scripts/`、`docs/` 靠 `.cbmignore` 的 `!` 规则放回默认排除）。
- 数据切分与官方数字逐项对上：Listen+ 29,439,278 → train 29,135,186 / val 143,557 / test 157,695；有目标的用户 4,627 / 4,599。
- EDA 的时间三层结论：日周期强且稳定（前半 / 后半相关 0.999）、周节律疑似 ±10%（标"未解释"）、活动量 300 天近翻倍（非平稳）。
- popularity 的 val 数字与官方实现**逐项一致到 6 位小数**（含 7 档 `hour` 扫描、最优 `hour=1.0`、recall / dcg / 命中率 / coverage）。
- 仓库根定下"档案分层"：`AGENTS.md` 只放 agent 相关 + 索引，进度与决策放本文件，管线与口径放 `architecture.md`。

## 数据与产物状态

- `data/raw/`（gitignore，454 MB）：`flat/50m/{listens,likes,dislikes,unlikes,undislikes}.parquet` + `album_item_mapping` / `artist_item_mapping`；sha256 已核对。
- `artifacts/splits/`（gitignore）：`train.parquet` 199.7 MB、`val.parquet` 0.8 MB、`test.parquet` 0.9 MB、`uid_map.parquet`、`item_map.parquet`。
- `artifacts/eda/`（gitignore）：6 张 PNG。
- `scripts/05_itemknn.py`、`scripts/rec_eval.py`：只出终端数字，不落产物（itemknn 一档约 1 分 20 秒，13 档约 16 分钟）。
- `artifacts/sasrec/state.pt`（gitignore，155 MB）：`06_sasrec.py` 的 50-epoch 权重；重跑脚本会覆盖它。
- `vendor/yambda-benchmarks/`（gitignore）：上游 clone，含自己那套 `.venv`。
- 仓库**目前一个 commit 都没有**。

## 杂项

- pip 缓存 `~/Library/Caches/pip` 168 MB、uv 缓存 `~/.cache/uv` 189 MB（用户级共享，未清理）。
- uv 安装脚本改过 `~/.zshrc`（加入 `~/.local/bin` 到 PATH）。
