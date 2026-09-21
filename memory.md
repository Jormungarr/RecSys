# memory.md — 项目进度与决策

> 项目档案：进度、关键决策与理由、数据与产物状态、杂项。
> 管线与口径见 `architecture.md`；数据集事实见 `docs/dataset_notes.md`；协作约定见 `AGENTS.md`。
> 逐字对话不存仓库（Kun 的线程存档在 `~/.kun/data/threads/`），这里只留提炼后的要点。
> 最后更新：2026-09-21。

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
11. val / test 两个窗口的指标跑齐：三个脚本加 `split` 参数（`04/05` 还可再带一个 hour），test 上用 val 选出的档、不重跑网格，sasrec 复用 val 的 checkpoint → `README.md`、`docs/baselines.md`
12. 版本 B 口径（`val_size=0`，能对官方表）：`02/04/05/06` 支持口径参数、产出 `artifacts/splits_b/`，三个基线跑齐并与官方 test 表逐项对照；并发掘出 coverage 在官方内部就不一致（已定：两条都报）→ `docs/baselines.md`、`architecture.md`、`README.md`
13. **sasrec 重写注意力 block（加性浮点掩码），顺带查出一个评测缺陷**（2026-09-21）：`nn.TransformerEncoder` → 自写 `AttentionBlock` / `Encoder`（子模块同名，所以旧 checkpoint 两个实现都能装），掩码折成一个**加性浮点矩阵**。查出旧写法在 **eval 的 MHA fused fast path** 下给“整行被 mask 掉的 padding 行”返回 **NaN**，而那些 NaN 在第二层当 K/V 把**有效行**也污染 → 9,207 用户里 **2,321 个（历史 < 512，占 25.2%）** 用户向量为 NaN、评测被压低约 21%；`train` 模式不走那条路，所以训练一直正常（这正好解释了“为什么只有评测被压低”）。**修正后 d64/512：val recall@100 0.082480 → 0.100252、test 0.074404 → 0.090524、口径 B 0.076916 → 0.094515。**
14. **相对时间偏置 `b[Δt桶]`**（参考书里 HSTU / GenRank 那一支）：可学、按 head 分开、加到注意力 logits 上（不取代点积），Δt 参照查询位自身。试过一次、**全指标变差**（val recall@100 0.100252 → 0.095807，回访 −4.9%、新歌 −2.8%，loss 持平）→ 开关 `USE_TIME_BIAS` 默认关。学出的 b 表本身是干净的单调衰减 → `docs/baselines.md`「相对时间偏置」。
15. **容量包：emb 64→128 + 2 层→4 层**（两个旋钮一起改，经确认）：val recall@100 **0.100252 → 0.121823（+21.5%）**、test 0.090524 → **0.110816（+22.4%）**，全部指标 +19~30%；**sasrec 第一次在两个窗口都超过 itemknn**，新歌轴也不再输给 popularity（但新歌轴只领先 0.4%，很薄）。→ `scripts/06_sasrec.py` docstring、`docs/baselines.md`「容量包」。
16. 归档：`docs/baselines.md` 逐处改正受缺陷影响的数字（seq 512 / 扩容两节加注意框、主表换成当前配置、口径 B 与官方对照重写、“低于官方 7.2%”改成“高 14.1%”）；README 的表与结论同步；排名融合**标为已作废、待重做**。

**待办**

- ~~sasrec 加时间特征~~ → **两项都做完了，结论都是“变差”**（2026-09-21，细节见下面两条“已完成”）：① Δ 嵌入（进 token 表示）−5.6%、③ 的相对时间偏置（进注意力 logits）−4.4%，两个开关都留着、**默认关**；③ 会话边界经确认**不做**。下面那三条候选的原设计只作历史：
  ① **Δ 嵌入**（推荐先做）：每个位置带上“距序列最后一次事件多久”，按对数分桶（5s / 1min / 1h / 1d / 1w…）查一张 `(桶数 × EMB)` 表加到 item embedding 上——最接近 itemknn 的 `tau^Δ`（回访轴上最强的对手）。前提是把 timestamp 打通到 `TrainDataset` / `collate`（现在只传 item / positive / negative / mask）。
  ② **相对时间偏置**（到位版，参考书里 HSTU 那一支）：手写注意力，在 logits 上加 `b[Δ 桶]`，位置偏移与时间偏移**分开**建模（`nn.TransformerEncoder` 不支持加性偏置，要自己写 ~50 行）。
  ③ **会话边界**（便宜补充）：相邻间隔 > 30 分钟（`docs/eda.md` 的会话阈值）给一个分段 embedding——接近 DSIN 的会话内/会话间拆分。
  注意：`is_organic` / `played_ratio_pct` / `track_length_seconds` **不在**现有 artifacts 里（`02_build_splits.py` 只写了 uid/item_id/timestamp）——要用得重跑 02（约 1 分钟，A 与 B 都要）；这几个与时间无关，可以分开做。
- **排名融合重做**：原来的 0.1081 用的是 d256 那版、在 NaN 缺陷下算出来的 top-100，**已作废**。现在 sasrec 单模型 test 已经 0.1108 且两条轴都第一，“融合还能不能再赚”要重新问（顺带：这次要不要把融合固化成脚本——上次它只是一次性核对）。
- **dropout / 早停**：官方实现就是 dropout 0.0、无早停。容量包把 loss 从 0.0471 拉到 0.0252（−46.5%），指标也大涨，但这两件事靠“loss 降得比指标快”已经不能判断了（旧论据受缺陷污染，已作废）——要判断就得真加 dropout 跑一次对照。
- **口径 B 的容量包**：目前 B 口径的 sasrec 还是 d64 / 2 层（`state_b.pt`，修正后 0.094515），最好那版没进 B。
- **拆开 EMB 与层数**：容量包是一次改两个旋钮（经确认），要归因得再跑两次。
- **多 seed**：现在所有 sasrec 数字都是单 seed 42；要谈“±几 % 算噪声”得重跑。
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
| 评测跑 val + test 两个窗口，超参只在 val 上选 | test 只用来报告；在 test 上扫网格等于拿评测集选参。三个基线的 top-100 只由训练集决定（与评测窗口无关），所以换 split 只换评测目标 |
| sasrec 在 test 上不重训、复用同一份 checkpoint | 重训会引入新的随机性，两列就不是同一个模型了；复用后 test 跑一次只要 30 秒 |
| 实验流程：公共 val 选参 → 冻结配置 → test 只读一次；A / B 两套口径并存 | 见 `architecture.md`「实验流程与两套口径（A / B）」；A（第一遍切分）迭代便宜、用于选型（sasrec 一轮 5.5 分钟），B（`val_size=0` 长训练集）只在定型后跑一次、用于对官方表；B 另存一套 artifacts，不覆盖第一遍 |
| coverage 两条都报（全用户 + 有目标用户） | 官方自己在模型之间就不一致（popularity / itemknn 用全用户、sasrec 用有目标用户）；两条都报才能对每一张官方数字。`coverage(全用户)` 还与评测窗口无关 → `architecture.md`、`scripts/rec_eval.py` |
| 长时训练用 `bg-train` skill 起，不用 `sleep + tail` 轮询 | 轮询每轮是一次完整 LLM 往返；skill 把渲染交给确定性进程，agent 只需“启动一次 + 长 poll 一次” → `~/.kun/skills/bg-train/SKILL.md` |
| Δ 时间特征留成开关 `USE_TIME_FEATURE`（默认关） | 实测 val 全指标变差（`docs/baselines.md`），默认回到已知更好的配置；留开关是为了能复现那版 |
| 自写 attention block（与 `nn.TransformerEncoderLayer` 同构，掩码用**加性浮点矩阵**） | 官方那头只吃布尔 mask、加不了加性偏置；顺带修掉 padding 行 NaN 污染评测的缺陷。子模块同名，所以旧 checkpoint 两个实现都能装 → `scripts/06_sasrec.py` |
| padding 行必须留一个可见位（它自己的对角线） | 整行被 mask 时 softmax 是 0/0；旧写法在 eval 的 fused kernel 下返回 NaN，再顺着第二层的 K/V 污染有效行（实测 25.2% 用户中招）。把这条不变式写进代码注释，不依赖 PyTorch 内部行为 |
| 相对时间偏置留成开关 `USE_TIME_BIAS`（默认关） | 与 Δ 时间特征是同一结论的两个方向（token 表示 / 注意力 logits）**都变差**；先验本身在 b 表上看得见，推断是与位置 embedding 重复 |
| 容量包（emb 128 + 4 层）两个旋钮一起改 | 用户确认的合并；代价是两个旋钮互相归因不了（已写进文档）。效果是两次时间特征失败之后最大的一处提升（val +21.5% / test +22.4%） |
| 发现评测缺陷后重读全部受影响数字，不照旧保留 | 缺陷使旧数字**系统性偏低**（约 21%，且不同配置污染比例不同）；两套口径、两个窗口、分轴数字全部重算或标注作废 |

## 本轮（2026-09-21）要点

- 给三个基线脚本加 `split` 参数：`04_popularity.py [split] [hour]`、`05_itemknn.py [split] [hour]`、`06_sasrec.py [split]`；不给参数就是原来的 val 行为。
- test 指标（第一遍切分的 test 窗口，4,599 个有目标的用户，候选池 627,648），recall@100 / 命中率@100：popularity 0.046988 / 0.388128（hour=1.0）、sasrec 0.070262 / 0.536638、itemknn 0.098117 / 0.579691（hour=0.5）。（sasrec 这两格后来被下面的 seq 512 版取代。）
- 两个窗口上**序完全一致**（itemknn > sasrec > popularity）；val → test 略降（recall@100：popularity −1.5%、itemknn −5.1%、sasrec −5.8%），test 是更靠后的窗口、模型只用 train 训练。
- 回归：改了脚本后重跑 val（04、05 带 hour）数字与原先一字未变；06 的 val 分支没有重跑（重跑会重训并覆盖 checkpoint，把 val 列和已跑的 test 列错开）。
- 口径澄清（原先被 `val_size=0` 这个字面误导）：官方协议**有 val**（`constants.py` 的 `VAL_SIZE = 1 天`、`timesplit.py` 的三段切分），`popularity` / `itemknn` / `bpr_als` 都先用它选超参；真的「没有 val」的是官方 **SASRec 脚本**（`train.py` / `eval.py` 两遍都 `val_size=0`，因为无超参、无早停）。官方报 test 的是**第二遍**——把第一遍的 val 窗口并回训练集，超参沿用第一遍在 val 上选出的。
- 定下实验流程与两套口径（用户 2026-09-21 选 (b)）：规则写进 `architecture.md`「实验流程与两套口径（A / B）」；该文档原先「所以现在的数字都在 val 上」那句与实况不符（第一遍切分的 test 列已跑出），已改掉。
- 版本 B 口径跑通（`02/04/05/06` 各加一个口径参数，默认 `a` 保持原行为）：`02_build_splits.py b` 产出的 id 空间（用户 9,208 / 物品 629,298）与官方 `random_rec` 打印的逐个相同；test 仍是 4,599 个有目标的用户 / 157,695 行，但 `-1` 行从 2,332 降到 1,956。B 的 sasrec 训练 309.4 秒（与 A 的 329.8 秒同量级），checkpoint 另存。
- 三基线 B 口径 test（recall@100 / 命中率@100）：popularity 0.047866 / 0.390302（**与官方六位小数全中，含 dcg 与 coverage**）、itemknn 0.108529 / 0.613394（官方 0.108476 / 0.613394，差 ≤ 5.3e-5）、sasrec 0.077438 / 0.559469（官方 0.082849 / 0.571646，低 6.5%）。sasrec 的缺口归因：单 seed 训练随机性 + 两处已记差异（词表 629,299 vs 631,004、位置编号）。（sasrec 那一行是 seq 200 时的；512 重训后是 0.076916 / 0.564253，低 7.2%，见下文。）
- A → B 同一模型 recall@100：popularity +1.9%、**sasrec +3.4%**（两遍都已改成 512）、itemknn +10.6%；三个模型的序不变。（原先记的 sasrec +10.2% 是两边都还是 seq 200 时的数，已被 512 版取代。）
- 新发现（已量化）：官方 coverage 在模型之间不是同一个定义（popularity / itemknn 用全部 9,208 个用户，sasrec 只用有目标的用户）；我们跟的是 sasrec 那种，所以 itemknn 的 coverage 才差 1.7 倍。换成全用户口径后 0.064991 vs 官方 0.064990。
- 回归：接口径参数时先跑了一次不带 `b` 的 A 口径，popularity / itemknn 的 test 数字与 `docs/baselines.md` 一字未变。
- coverage 按用户的决定（选 (c)）改成**两条都报**：`rec_eval.evaluate` 加可选参数 `all_tops`、`show()` 打两行；`04/05/06` 主表都传全用户 tops（`04` 的“只看可排名目标”对照也传，它的 tops 与主表相同；“过滤已交互”对照只有有目标那份）。
- coverage 实测（@100）：**口径 A** popularity 全用户 0.000159 / 有目标 0.000159（val、test 同）、sasrec 全用户 0.021624 / 有目标 val 0.016782、test 0.016758、itemknn 全用户 0.064442 / 有目标 val 0.037583、test 0.037859；**口径 B** popularity 0.000159、sasrec 全用户 0.021136 / 有目标 0.016674（vs 官方 0.015887）、itemknn 全用户 0.064991 / 有目标 0.038521（vs 官方 0.064990）。**两种口径各对上一张官方数字。**
- 顺带确认：`coverage(全用户)` 与评测窗口无关（top-k 只由训练集和模型决定），同一 (模型, 口径) 只有一个数，不必 val / test 各跑一遍。
- **路径二第一项：sasrec 序列长度 200 → 512 跑通**（其余超参不变，val 上没有任何选择）。口径 A：val recall@100 0.074572 → 0.082480（+10.6%）、val dcg@100 +14.2%、val coverage(有目标) +45.9%；test recall@100 0.070262 → 0.074404（+5.9%）、test dcg@100 +14.4%、coverage +47.1%。**但口径 B（512 重训）test recall@100 反而 −0.7%（0.077438 → 0.076916）**，而 dcg +8.0%、coverage +45.7%。→ **dcg 与 coverage 在两口径两窗口都一致变好，recall@100 不稳**（±6% 落在单 seed 噪声带内：同一个模型 val→test 就差 9.8%）；要定量得多 seed。
- 动机来自 EDA：历史长度（去重物品）中位 666 / p90 2,239，而截断上限一直在被顶满（两个口径下打印的序列长度中位都正好 = 200）；改成 512 后中位变成 512。
- **仍然打不过 itemknn**：test recall@100 0.074404 vs 0.098117（1.32×，改前是 1.40×）；val 0.082480 vs 0.103341。
- 代价与两个坑：① 每 epoch 12.7–15.4 秒（原 4.9–7.0），**墙钟 30.8 分钟 vs 自报 11.4 分钟——差额已查明是机器 Idle Sleep**（`pmset`：16:33:59→16:50:23 一次睡 16.4 分钟，16–17 点共 8 次 ≈ 19.6 分钟；电池供电，`perf_counter` 不计休眠）→ **估成本用 epoch 计时**；② **eval 必须按用户分批**——一次把 9,207 个用户喂进 transformer 要 19 GB 的 `(B, heads, L, L)` 注意力矩阵，实测换页到 swap 18 GB、4 分半没跑完，只能停掉（`INFER_BATCH = 512` 后每窗口 32 秒；换 2048 复算指标逐项相同）。
- 换序列长度会让旧 checkpoint 失效（位置 embedding 尺寸不匹配）：跑之前把 200 那份备份成 `state_l200.pt`；给 `06` 的 checkpoint 读取加了尺寸守卫（错配时给可读提示，不再是一个难懂的 traceback）；`state_b.pt` 后来也在 512 下重训（793.8 秒 ≈ 13.2 分钟，这次没碰到休眠，自报 = 墙钟）。
- **诊断：把 test 目标拆成「回访 / 新歌」两桶**（口径 A test，4,599 用户 / 157,695 行；三个模型各算一份 top-100，一次性核对、未固化成脚本）。目标行里**回访 105,069 行 = 66.6%**、新歌 52,626 行 = 33.4%。汇总 recall@100（命中行数 / 桶行数）：popularity 全部 0.0392 / 回访 0.0434 / 新歌 0.0307；itemknn 0.0843 / 0.1172 / 0.0186；sasrec-512 0.0713 / 0.0962 / 0.0218。top-100 落在用户历史内的比例：popularity 17.45%、sasrec 39.55%、itemknn 45.12%。
- 上条的结论：**itemknn 的优势几乎全在回访桶**（0.1172 vs 0.0962），**新歌桶上反而是 sasrec 更高**（0.0218 vs 0.0186），而新歌桶里 **popularity 最高**（0.0307）。（这是“汇总口径”的第一次测量；后来按用户决定把两桶固定进评测、改成**逐用户**口径，数字见下面两条。）
- **(c) 三列口径落地**（用户选“两条都报、都推进”）：`rec_eval.evaluate` 加可选参数 `history`（每用户训练期物品集合），多报「回访 / 新歌」两桶的 recall 与命中率（**逐用户**口径：分母 `min(桶内行数, k)`，只在桶内有目标的用户上平均）；`show()` 打两行 + 一行分桶规模；`04/05/06` 主表都传它（`06` 用**完整 train**，不是截断后的序列）。口径写进 `architecture.md`。
- 两桶实测（口径 A，recall@100，逐用户）：**val** popularity 全部 0.047727 / 回访 0.050447 / 新歌 0.046177，itemknn 0.103341 / 0.134022 / 0.024921，sasrec-512 0.082480 / 0.103669 / 0.030482；**test** popularity 0.046988 / 0.048431 / 0.044826，itemknn 0.098117 / 0.127228 / 0.024597，sasrec-512 0.074404 / 0.094903 / 0.027782；口径 B 的 sasrec-512 test 是 0.096368 / 0.030742。桶行数自洽（test 105,069 + 52,626 = 157,695）。**两条轴排序完全相反**：回访 itemknn > sasrec > popularity，新歌 popularity > sasrec > itemknn（两窗口一致）。
- **排名融合对照**（RRF，各模型 top-500 → top-100，口径 A test）：itemknn + sasrec = **0.1014**（超过 itemknn 单独的 0.0981，+3.4%），回访 0.1327 / 新歌 0.0272；sasrec ×2 权重反而差（0.0906）；三路（加 popularity）全部 0.0963 但**新歌 0.0429**（≈ popularity 单独的 0.0448）。→ 融合确有互补，但幅度小，因为两者强项压在同一条轴。一次性核对、未固化成脚本；同一进程里重算的单模型数字与主表逐项一致（0.0981 / 0.0744 / 0.0470），所以增益不是口径差异造成的。
- **融合用 d256 重跑后增益变大**（口径 A test）：itemknn + sasrec = **0.1081**（vs itemknn 单独 0.0981，**+10.2%**；d64 时是 0.1014 / +3.4%），回访 0.1429（比 itemknn 高 12%）/ 新歌 0.0285（**低于** sasrec 单独的 0.0308）；sasrec ×2 权重 0.0978；三路 0.1018 / 新歌 0.0444。→ 单模型变强时融合也跟着变强，但两条轴的互补仍主要在回访轴。
- **时间特征 Δ 试了一次：变差、已关**（口径 A val，d64/512）：recall@100 0.082480 → 0.077826（**−5.6%**）、dcg@100 −5.5%、命中率@100 −2.7%、回访 −6.5%、新歌 −9.7%、coverage(有目标) −10.6%；训练 loss 也更差（epoch 50：0.0486 vs 0.0471）。做法：`Δ = 距窗口内最后一次事件多久` 对数分桶（13 桶）当 embedding，训练/推理的查询位置都是 Δ=0（不泄未来）；成本与不加时同量级（15.0 秒/epoch，总 804 秒 ≈ 13.4 分钟）。**推断（未验证）**：位置 embedding 已隐含“多久以前”，逐位置的绝对 Δ 不是模型要的“成对 recency”（下一个候选：相对时间偏置）。脚本里留开关 `USE_TIME_FEATURE`（默认 False）。归档：`docs/baselines.md`「时间特征 Δ」一节。
- **工具流程改了：长时训练改用 `bg-train` skill**（`~/.kun/skills/bg-train`）——之前用 `sleep + tail` 轮询正是它的反模式（每轮烧一次完整 LLM 往返）。接入踩的坑：它的 sniffer 只认 `loss=` 这类 KV / tqdm / 花括号 dict，而我们的脚本原本是 `loss 0.161484`（空格）→ 状态文件一直空；已把 print 改成 `loss=`。启动：`bgtrain.py run <名字> --cwd <项目> -- uv run python -u scripts/06_sasrec.py val`；面板 `~/.bgtrain/<名字>/dashboard.html`，日志 `train.log`。
- **路径二第二项：sasrec 扩容 emb 64 → 256 跑通**（只改这一处；heads / layers / dropout 不变，口径 A）。**两个窗口全部指标都涨**：val recall@100 0.082480 → 0.087210（+5.7%）、test 0.074404 → 0.079925（+7.4%）；dcg@100 +9.7% / +8.4%、coverage(有目标) +39.5% / +39.2%、命中率 +2.1% / +3.0%；recall@10 val +11.2% / test +7.7%。
- 扩容的增益**偏向新歌轴**（val +12.9% / test +11.0%）而不是回访轴（+4.7% / +7.6%）——推断（未验证）：回访是低复杂度映射，64 维就够；新歌要刻画偏好本身，吃容量。
- **过拟合迹象**：训练 loss 0.0471 → **0.0257**（−45%），而 val recall@100 只涨 5.7% → 多出的容量有一部分花在背训练序列上。现在没有 dropout、没有早停（官方实现就是这样），这是“还能不能继续加容量”的前提。
- 成本：59.4 秒/epoch（d64 是 12.7–15.4），50 epoch 总 2828.6 秒 ≈ 47 分钟（含 val 推理，机器没休眠）；带宽只涨约 4 倍而不是按参数量平方涨——d64 时的瓶颈不在矩阵计算。
- 仍打不过 itemknn（test recall@100 0.079925 vs 0.098117，1.23×；改前 1.32×）。
- 权重：`state.pt` 现为 **d256**（d64 那份已备份成 `state_l512_d64.pt`）；**口径 B 的 d256 还没跑**。归档：`docs/baselines.md` 新增「扩容：emb 64 → 256」一节，主表/并排表/回访-新歌表都换成了 d256。（注：这条与它下面的 d256 数字后来都受 NaN 缺陷影响，见下文。）
- **手写注意力 block（第 1 步）**：`b ≡ 0` 时与旧实现逐位对照——这个对照原本是给重构兜底的，结果**抓出一个既有缺陷**（padding 行 NaN 污染有效行，见「已完成」13）。三条独立验证（旧实现关 fastpath、旧实现 train 模式、新实现）都给同一个数 **0.100252**，说明是旧实现在 eval 下那条路径的问题，不是重构引入的。
- 顺带理顺产物语义：Δ 那版存成 `state_l512_d64_delta.pt`、`state.pt` 放当前最好配置；守卫加到 **5 处**（序列长度 / 维度 / 层数 / Δ 开关 / 相对时间偏置开关），实测能拦下错配的旧 checkpoint（可读提示，不再是 traceback）。
- **相对时间偏置（第 2 步）**：Δt 进注意力 logits、按 head 分开、参照查询位自身；参照实现两处不能照抄（fun-rec 的 `attention_type` 默认没开这个偏置；它的桶边界是**原始秒 + max_interval=1024**，17 分钟以上全并成一档）。开工前三项检查：13 个桶都有实质样本、Δt 与旧 Δ 预分桶逐位等价（差 0）、完整 block0 复算差 4.8e-7。结果：全指标变差 −1.4~−4.9%，loss 持平；b 表是干净的单调衰减 → 先验没错、推断与位置 embedding 重复。
- **容量包（第 4 步）**：emb 128 + 4 层一起上 → 两个窗口全部指标 +19~30%，**sasrec 首次在两个窗口都超过 itemknn**，新歌轴也追平 popularity（只领先 0.4%）；loss −46.5%。成本 51 秒/epoch、50 epoch 2703 秒。
- **一处方法论更正**：d256 那次的“过拟合迹象”（loss −45%、val 只 +5.7%）是在**受污染的评测**下算的；修正后同样幅度的 loss 下降（−46.5%）对应 +21.5% 的 val 提升 → 那条论证作废，不能再用它当“该加 dropout”的理由（要判断得真跑对照）。
- 排名融合 0.1081 **已作废**（输入是污染过的 top-100），标成“待重做”；口径 B 的 sasrec 那一格从“低于官方 7.2%”改成“高于官方 14.1%”，并注明两边配置不同、**不算复现一致**。

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
- `artifacts/splits_b/`（gitignore）：版本 B（`val_size=0`）的切分，`train.parquet` 200.7 MB + `test.parquet` + 两张映射表；**没有 val**。
- `artifacts/eda/`（gitignore）：6 张 PNG。
- `scripts/05_itemknn.py`、`scripts/rec_eval.py`：只出终端数字，不落产物（itemknn 一档约 1 分 20 秒，13 档约 16 分钟）。
- `artifacts/sasrec/`（gitignore）：`state.pt` = **当前最好配置（seq 512 / emb 128 / 4 层，val recall@100 0.121823 / test 0.110816）**；备份：`state_l512_d64.pt`（d64/512/2 层，修正后 d64 数字的来源）、`state_l512_d256.pt`（扩容那版 d256/2 层，数字受缺陷污染）、`state_l512_d64_delta.pt`（d64+Δ 那版）、`state_l512_d64_timebias.pt`（d64+相对时间偏置那版）、`state_l200.pt`（seq 200）、`state_b.pt`（口径 B / d64/512/2 层）。换序列长度、维度、层数、Δ 开关或相对时间偏置开关都会让旧 checkpoint 装不回去（**五处守卫**，给可读提示）。
- `vendor/yambda-benchmarks/`（gitignore）：上游 clone，含自己那套 `.venv`。
- 仓库此时有 6 个 commit（`4063f27` 初始化 / `76bf298` val+test 双窗口 / `60084dc` 口径 B 那一批 + 修正过时的 git 状态描述 / `207e872` 自写注意力 block + NaN 修复 / `6d47d9c` 相对时间偏置 / `3fc582d` 容量包——后四个是 2026-09-21 这一轮里逐步提交的，每步一个 commit，便于回退）；文档归档另有一个 commit。

## 杂项

- pip 缓存 `~/Library/Caches/pip` 168 MB、uv 缓存 `~/.cache/uv` 189 MB（用户级共享，未清理）。
- uv 安装脚本改过 `~/.zshrc`（加入 `~/.local/bin` 到 PATH）。
