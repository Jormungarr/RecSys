"""sasrec 基线：自己实现一份，训练与评测口径对齐官方 benchmark。

运行：
    uv run python scripts/06_sasrec.py          # 在 val 上训练并评测（50 epoch；按当前常量 emb 64 / 2 层 / seq 512 约 13 分钟）
    uv run python scripts/06_sasrec.py test     # 在 test 上评测，复用 val 那次存下的 checkpoint（不重训；checkpoint 必须与当前常量同配置，否则被守卫拦下）
    uv run python scripts/06_sasrec.py train b  # 版本 B 口径：在更长的 train 上训练、在 test 上评测（无 val）

口径出处（vendor/yambda-benchmarks/benchmarks/models/sasrec/）：
- 序列：每用户一条 item 序列（时间升序），取最后 max_seq_len 条（`data.py::preprocess`）；官方默认 200，
  本机改成 512（理由见 `MAX_SEQ_LEN` 处）
- 模型（`model.py::SASRecEncoder`）：item embedding + 位置 embedding（倒序，最新为 0）+ **Δ 时间 embedding（本机加的，官方没有）**
  → LayerNorm → Dropout → LAYERS 层自写 block（因果掩码 + padding 掩码；与 `nn.TransformerEncoderLayer` 同构，
  自己拼是为了能往注意力分数上加**加性偏置**——TransformerEncoderLayer 只吃布尔 mask）→ 取每序列最后一个有效位置的向量
- **时间特征（本机加的，官方实现没有；默认关）**：`Δ = 距该用户窗口内最后一次事件多久`（秒），对数分桶后查一张
  `(桶数 × EMB)` 的表，加到 item / 位置 embedding 上。Δ 与 itemknn 的“该用户最后一次 − 该次时间”同义；
  训练与推理的查询位置都恰好是 Δ = 0，口径一致、不泄露未来。**2026-09-21 试过一版：val 全指标变差（见
  `docs/baselines.md`），所以 `USE_TIME_FEATURE` 默认 False。**
- **相对时间偏置（本机加的，官方实现没有；`USE_TIME_BIAS`，默认开）**：把可学的 `b[Δt 桶]`（按 head 分开）
  加到**注意力分数**上，`Δt = t_i − t_j`、参照点是查询位自身（训练位置 i 预测 i+1，只用 i 时刻已知的时间）。
  与上一条的区别：Δ 进的是注意力 logits（成对量），不是 token 表示（那是绝对新鲜度、已被位置 embedding 覆盖）。
  桶边界用本机 `TIME_BOUNDS`（对数 13 桶，不是参考实现那个 `max_interval=1024` 秒线性截断，见 `docs/baselines.md`）。
- **位置通道（`POS_MODE`，2026-09-22 加）**：`index`（默认）= 位置索引查表；`time` = **用时间替换位置** ——
  同一张位置表、不新增任何参数，下标 = Δ 在**训练集等量分位边界**（511 个，见 `build_time_edges`）里的档号，
  Δ = 窗口内最后一次已知事件 − t_i。依据：同一用户窗口内 Δ 与位置索引单调一一对应，替换是**换刻度**而不是加信息。
  （第一版用「线性 Δ / 间隔中位数」：92.7% 的位置挤在末档、每用户中位只用 14 档，已弃；训练前 main() 有硬判据拦这种情况。）
- **位置通道的四种形态 + 衰减核（2026-09-22 晚加）**：`time`（分位档，整数）/ `time_interp`（同表 + 小数档插值，不新增参数）/
  `time_mlp`（x = log1p(Δ/200) 过两层 MLP）/ `time_t2v`（可学频率的周期基 [cos(ωx), sin(ωx)]，Bochner/Time2Vec 那一支，
  只留周期项、不带线性项与相位——线性项会把 item 通道压掉，见 2026-09-22 的失败与修复）；
  `USE_DECAY` 与位置通道**正交**：把 −λ_head·log1p(Δt) 加到注意力 logits（λ 初值 0 → 起点等于基线）。
  四种形态都是“用时间替换位置”，开工前会打印**每用户可区分的位置比例**（index 1.00 / 连续 0.96 / 分位档 0.11）。
- 训练样本（`data.py::TrainDataset`）：item = seq[:-1]、positive = seq[1:]（每个位置都预测下一首），
  负样本**每个位置随机 1 个**；损失 = BCEWithLogits([正分; 负分], [1; 0])
- 超参（`train.py` 的默认值）：heads 2 / dropout 0.0 / lr 1e-3 / Adam / batch 256 / seed 42；
  epoch 数官方默认 100，本机按 `docs/benchmark_repro.md` 记录的 50。常量是**滚动**的：当前（emb 64 + 2 层）是
  2026-09-22 第二轮五臂对照用的快迭代配置；最强配置是 emb 128 + 4 层 + 序列 512（权重 `state_l512_d128_l4.pt`）。对表能力由口径 B + 文档记录保留。
- 评测（`eval.py`）：用户向量 × 全部 item embedding 内积 → top-100，指标走 `scripts/rec_eval.py`
- 设备：训练用 MPS，**推理必须用 CPU**（官方在 MPS 上跑 eval 会崩，见 `docs/benchmark_repro.md`）；
  推理按用户分批（`INFER_BATCH`）—— 模型对每个用户独立，分批不改变结果，但注意力矩阵是 `(B, heads, L, L)`，
  L=512 时全量喂 9,207 个用户要约 19 GB（实测会疯狂换页，必须分批）

与官方实现的三处已知差异：
1. 词表与 padding：官方词表 = 切分前整个文件里出现过的物品、编号从 1 开始（0 留给 padding），
   排序时连 padding 行一起当候选；我们用现有 id 空间 0..627,647，padding id = 627,648，
   **排序时排除 padding** → 候选池 627,648，与 popularity / itemknn 完全一致（官方多一个候选）。
2. 位置编号：官方是 arange(S-1, -1, -1) 再按长度掩码，S 是**该 batch 内的最大长度**，而 eval 的
   DataLoader 还 shuffle=True → 同一个用户在不同 batch 组成下位置编号会变；我们固定左填充到 MAX_SEQ_LEN、
   最新行为是位置 0，可复现（序列满 MAX_SEQ_LEN 时两者完全一致，只有短序列会被平移）。
   把 `MAX_SEQ_LEN` 设回 200 时，那份“导官方 checkpoint 前向逐位一致（差 0）”的对照成立。
3. 切分：官方 train.py / eval.py 都用 val_size=0（第二遍训练集）并在 test 上报告；我们默认（口径 A）
   在第一遍切分的 train 上训练，val 用来选超参、test 用来报告，与 popularity / itemknn 同一口径；
   `train b` 复刻官方口径（口径 B：train 更长、没有 val，候选池 629,298）。

三个模式：`val` 训练并评测（同时存 checkpoint）、`test` 不重训直接加载那份 checkpoint、
`train` 训练后在 test 上评测（口径 B 用它，因为那套切分没有 val）；B 的 checkpoint 另存 `state_b.pt`。
用户向量与 top-100 只由训练集和模型决定、与评测 split 无关，复用同一份权重才能保证 val / test 两列出自同一个模型。

注意：训练带随机性（初始化、负采样、数据顺序），不追求与官方逐位一致，只对量级。
"""

import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rec_eval import evaluate, history_flags, show, split_by_user, target_axes

ROOT = Path(__file__).resolve().parents[1]
SPLITS_ID = sys.argv[2] if len(sys.argv) > 2 else "a"  # 口径：a = 第一遍（有 val），b = 第二遍（val_size = 0）
assert SPLITS_ID in ("a", "b"), f"口径只能是 a 或 b，收到 {SPLITS_ID!r}"
SPLITS = ROOT / "artifacts" / ("splits" if SPLITS_ID == "a" else "splits_b")
CHECKPOINT = ROOT / "artifacts" / "sasrec" / ("state.pt" if SPLITS_ID == "a" else "state_b.pt")

MAX_SEQ_LEN = 512  # 官方默认 200；本机改 512 —— 训练集历史长度（去重物品）中位 666 / p90 2,239（docs/eda.md），
                   # 截到 200 时一半以上用户的历史被砍掉（实测：两个口径下序列长度中位都正好 = 200，即顶到上限）
EMB = 64  # 官方默认 64。2026-09-22（第二轮）为了「位置通道四种形态 + 衰减核」的五臂对照临时回到 64/2 层：
          # 每臂 ~13 分钟、五臂背靠背；代价是结论只在 d64/2 上成立（是否上 d128/4 等这批结果再定）。
          # 当前最强配置（emb 128 + 4 层）的权重在 state_l512_d128_l4.pt
HEADS = 2
LAYERS = 2  # 官方默认 2；与 EMB=64 一起构成本轮五臂对照的配置，理由见 EMB 处
DROPOUT = 0.0
USE_BF16 = False  # 训练步用 bf16 自混精。微基准：d128/4 层上单步 1110 → 967 ms（同进程交替测 4 轮，−13%）；
                  # 但 2026-09-22 在 d64/2 层上真跑：epoch 耗时与 fp32 相同（无收益），val recall@100 0.091862
                  # 落在三次同配置 fp32 重复（0.091091/0.091490/0.091868）的带内 → 无可辨认影响。
                  # 所以默认关；它在 d128/4 层上是否真能提速从未在真跑里验证过（要用得先单独验）
LR = 1e-3
BATCH = 256
SEED = 42
EPOCHS = 50  # 官方默认 100；本机按 docs/benchmark_repro.md 记录的 50
INIT_RANGE = 0.02  # model.py::_init_weights
SELECT_K = 100
INFER_BATCH = 512  # 推理时一次喂多少用户；注意力矩阵是 (B, heads, L, L)，全量喂 9,207 × L=512 要约 19 GB

# 时间特征：Δ = 「距该用户窗口内最后一次事件多久」（秒；最后一次 = 0），对数分桶后当一张 embedding 用。
# 口径与 itemknn 的 Δ（“该用户最后一次 − 该次时间”）同义；训练与推理的查询位置都恰好是 Δ = 0，两边一致。
TIME_BOUNDS = (5, 15, 60, 300, 1800, 7200, 21600, 86400, 259200, 604800, 1814400, 7776000)
N_TIME_BUCKETS = len(TIME_BOUNDS) + 1
USE_TIME_FEATURE = False  # Δ 时间特征开关：2026-09-21 在 d64/seq512 上试过，val 全指标变差（见 docs/baselines.md），默认关
                        # 要复现那版改成 True 重跑（checkpoint 里有无 time_embedding 必须与开关一致，下有守卫）
USE_TIME_BIAS = False  # 相对时间偏置开关：b[Δt 桶]（可学，按 head 分开）加到注意力分数上，Δt = t_i − t_j、参照查询位自身
                      # 与上面那版的区别：Δ 进的是**注意力 logits**（成对量），不是 token 表示（绝对新鲜度）。
                      # 出处：HSTU 的 rab_{p,t}、GenRank 的 ALiBi 式 bias（reference/fun-rec chapter_6_scaling）。
                      # **2026-09-21 在 d64/seq512 上试过一版：val 全指标小幅变差（recall@100 0.100252 → 0.095807，
                      # 回访 −4.9%、新歌 −2.8%），但学出的 b 表本身是干净的单调衰减（1min–30min 最高、6 桶后单调降到 −1.4）
                      # → 先验没错，是它和位置 embedding 重复且更弱。见 docs/baselines.md。默认 False。**
                      # checkpoint 里有无 time_bias 必须与开关一致，下有守卫。
POS_MODE = os.environ.get("POS_MODE", "index")  # 位置通道。可用环境变量覆盖，方便一条命令里连跑多臂
                    # index       = 位置索引查表（基线）：下标 = 「距查询点第几首」(0..511)
                    # time        = 用时间替换位置：下标 = Δ 的**等量分位档号**（整数，512 档；实测每用户只有 ~56 档）
                    # time_interp = 同上，但下标是**小数**（桶内按 log1p 插值到相邻两行），不新增参数
                    # time_mlp    = 用时间替换位置：x = log1p(Δ/200) → 两层 MLP → EMB（连续、无桶；新增参数）
                    # time_t2v    = 同上，但用可学频率的周期基 [ω·x+φ, sin(ω·x+φ)] → Linear(2·EMB, EMB)（Time2Vec/Bochner 那一支）
                    # Δ = 窗口内最后一次已知事件 − t_i（秒；与 delta_buckets_tensor 同一参照点，查询位 Δ = 0）
                    # 依据：同一用户窗口内 Δ 与位置索引单调一一对应 —— 替换不是加信息，而是**换刻度**（均匀序数 → 真实时间）
                    # 与 USE_TIME_FEATURE / USE_TIME_BIAS 的区别：那两个是**在位置之外加**时间，都实测变差；这一族是**替换**
                    # 参数集：index / time / time_interp 完全相同（严格配对）；time_mlp / time_t2v 有新参数，配对性变弱
                    # checkpoint 里记 `_pos_mode`（形状相同，装错了不会报错、只会静默算出错的分数）
USE_DECAY = os.environ.get("USE_DECAY", "0") == "1"  # 衰减核：把 −λ_head · log1p(Δt) 加到注意力 logits（λ 每 head 一个、可学、**初值 0** → 起点与基线逐位相同）
                    # 与位置通道**正交**：这一支保留位置，只把时间放进注意力权重 —— 正面检验 itemknn 的 τ^Δ 那种强先验衰减
TIME_UNIT = 200.0  # 连续编码的时间单位（秒）= 训练集相邻间隔中位数；只用于 time_mlp / time_t2v 的 x = log1p(Δ / TIME_UNIT)
TIME_EDGES: torch.Tensor | None = None  # time / time_interp 的 511 个分档边界，由 build_time_edges 装好 / 从 checkpoint 取回
assert not (POS_MODE != "index" and USE_TIME_FEATURE), "POS_MODE 已经拿 Δ 当位置通道，再叠 USE_TIME_FEATURE 会把归因搅在一起"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"  # 模式：val（训练+评 val）/ test（复用 checkpoint 评 test）/ train（口径 B：训练+评 test）
assert not (SPLITS_ID == "b" and SPLIT == "val"), "版本 B 没有 val（val_size = 0）；用 train 训练、用 test 复用 B 的 checkpoint"
EVAL_SPLIT = "test" if SPLIT == "train" else SPLIT  # 本次评测的窗口
TRAIN_FIRST = SPLIT != "test"  # val / train：从头训练；test：复用 checkpoint

TRAIN_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
INFER_DEVICE = "cpu"  # 官方 eval 在 MPS 上会崩


def load(name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(SPLITS / f"{name}.parquet", columns=columns)


def by_user(train: pd.DataFrame, column: str, n_users: int) -> list[np.ndarray]:
    """每用户一条时间升序的序列（截到最后 MAX_SEQ_LEN 条）；column 取 item_id 或 timestamp。"""
    users, chunks = split_by_user(train.uid.to_numpy(), train[column].to_numpy())
    out: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_users
    for uid, chunk in zip(users, chunks):
        out[uid] = chunk[-MAX_SEQ_LEN:].astype(np.int64)
    return out


def train_sequences(train: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """每用户一条 item 序列（时间升序，截到最后 MAX_SEQ_LEN 条）。"""
    return by_user(train, "item_id", n_users)


def train_timestamps(train: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """每用户一条 timestamp 序列（与 train_sequences 同样的截断），供 Δ 与成对 Δt 两种时间特征用。"""
    return by_user(train, "timestamp", n_users)


def pad_sequences(sequences: list[np.ndarray], users: np.ndarray, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """左填充到 MAX_SEQ_LEN（最新行为落在最后一格 = 位置 0）；返回 (item, mask)。"""
    item = np.full((len(users), MAX_SEQ_LEN), pad_id, dtype=np.int64)
    mask = np.zeros((len(users), MAX_SEQ_LEN), dtype=bool)
    for row, uid in enumerate(users):
        sequence = sequences[uid][-MAX_SEQ_LEN:]
        item[row, MAX_SEQ_LEN - len(sequence) :] = sequence
        mask[row, MAX_SEQ_LEN - len(sequence) :] = True
    return torch.from_numpy(item), torch.from_numpy(mask)


def delta_buckets(times: np.ndarray) -> np.ndarray:
    """把一条时间戳序列换成「距该序列最后一次事件多久」的桶编号（最后一次 = 0）。

    每个位置的 Δ 只用它自己和它之前的事件（不泄未来）；训练与推理用同一套口径。
    """
    if times.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.digitize(times[-1] - times, TIME_BOUNDS).astype(np.int64)


def pad_times(times: list[np.ndarray], users: np.ndarray) -> torch.Tensor:
    """左填充到 MAX_SEQ_LEN 的**原始时间戳**（与 pad_sequences 同样的对齐）。

    int32 装得下（时间戳 ~2.6e7），比 int64 省一半内存。两种时间特征都在模型里用这份原始时间戳现算，
    所以它替掉了原来那路“已分桶的 Δ”（dataset 里不再预先分桶）。
    """
    stamps = np.zeros((len(users), MAX_SEQ_LEN), dtype=np.int32)
    for row, uid in enumerate(users):
        stamp = np.asarray(times[uid])[-MAX_SEQ_LEN:].astype(np.int32)
        stamps[row, MAX_SEQ_LEN - len(stamp) :] = stamp
    return torch.from_numpy(stamps)


def delta_buckets_tensor(times: torch.Tensor) -> torch.Tensor:
    """(B, 512) 时间戳 → 「距该行最后一次事件多久」的桶编号，等价于原来在 dataset 里算的那版 Δ。

    行内最后一次事件就是参照点：训练时该行是输入序列（窗口去掉最后一条），推理时该行是整条窗口——两种情况下
    参照点都恰好是「最新已知的那次事件」，与旧实现（逐样本 `delta_buckets(times[:-1])`）一致。
    """
    bounds = torch.tensor(TIME_BOUNDS, dtype=torch.int32, device=times.device)
    # right=True 是为了与 numpy 的 `np.digitize`（区间左闭右开）对齐，否则恰好落在边界上的 Δ 会差一档
    return torch.bucketize(times[:, -1:] - times, bounds, right=True, out_int32=True)


def build_time_edges(times: list[np.ndarray]) -> torch.Tensor:
    """训练窗口内 Δ = (该用户最后一次已知事件 − 每次事件) 的 511 个**等量分位**边界 → (511,) int64。

    分位点取 i/512（i = 1..511），所以每个档位的样本量大致相等，且与 Δ 的量纲、重尾程度无关。
    2026-09-22 的第一版用「线性 Δ / 用户间隔中位数」：位置级 Δ/中位数 p50 已经是 8634（跨 7.4 个数量级），
    结果 92.7% 的位置挤进最后一档、每用户中位只用 14 档 —— 所以改成等量分位。
    只用训练集、只算一次（确定性）；它同时写进 checkpoint，评测时从 checkpoint 取，避免两边各算一套。
    """
    deltas = np.concatenate([(t[-1] - t).astype(np.int64) for t in times if len(t)])
    quantiles = np.arange(1, MAX_SEQ_LEN) / MAX_SEQ_LEN
    return torch.from_numpy(np.round(np.quantile(deltas, quantiles)).astype(np.int64))


def time_position_index(times: torch.Tensor) -> torch.Tensor:
    """(B, L) 时间戳 → (B, L) 的位置表下标，供 POS_MODE="time"（用时间替换位置）用。

    Δ = 该行最后一次已知事件 − t_i（秒；与 `delta_buckets_tensor` 同一个参照点：训练与推理的查询位都恰好 Δ = 0），
    再按 `TIME_EDGES` 分档。边界是 511 个等量分位点 → 档号恰好落在 [0, MAX_SEQ_LEN-1]，与位置表的行数一一对应。
    区间约定与 np.searchsorted(side="left") 一致（即 boundaries[i-1] < Δ ≤ boundaries[i]）。
    padding 位会拿到很大的 Δ（t = 0）→ 落进最后一档，无所谓：它们在注意力里被 mask，输出也不会被读。
    """
    assert TIME_EDGES is not None, "POS_MODE=time 需要先用 build_time_edges 装好分位边界（训练路径在 main() 里做）"
    delta = (times[:, -1:] - times).to(torch.int64)
    return torch.bucketize(delta, TIME_EDGES.to(delta.device))


def time_position_frac(times: torch.Tensor) -> torch.Tensor:
    """(B, L) → (B, L) 的**小数**位置表下标，供 POS_MODE="time_interp" 用（桶内按 log1p 线性插值）。

    整数部分与 `time_position_index` 共用同一套分位边界，另外把「在该桶里靠上还是靠下」也算进去：
    桶 (edges[b-1], edges[b]] 内部按 log1p(Δ) 线性给出小数档号 b-1+w。
    两个不变式：① 档号随 Δ 单调不减；② 在桶边界上**连续**（Δ 恰好等于 edges[b] 时 w=1、r=b，与下一桶的起点一致），
    所以它只把 time 版的「取整撞档」换成“同一张表的连续混合”，参数一个不增。
    """
    assert TIME_EDGES is not None, "POS_MODE=time_interp 需要先用 build_time_edges 装好分位边界"
    edges = TIME_EDGES.to(times.device)
    delta = (times[:, -1:] - times).to(torch.int64)
    bucket = torch.bucketize(delta, edges)  # 0..MAX_SEQ_LEN-1
    lower = torch.cat([torch.zeros(1, dtype=edges.dtype, device=edges.device), edges])[bucket]
    upper = torch.cat([edges, torch.full((1,), 1 << 62, dtype=edges.dtype, device=edges.device)])[bucket]
    x, xl, xh = torch.log1p(delta.float()), torch.log1p(lower.float()), torch.log1p(upper.float())
    w = ((x - xl) / (xh - xl).clamp(min=1e-6)).clamp(0.0, 1.0)
    return (bucket - 1 + w).clamp(0, MAX_SEQ_LEN - 1)


def time_x(times: torch.Tensor) -> torch.Tensor:
    """(B, L) → (B, L, 1)：连续时间特征 x = log1p(Δ / TIME_UNIT)，供 time_mlp / time_t2v 用（无桶）。

    TIME_UNIT = 训练集相邻间隔中位数（实测 200 s、p10 142 s，跨用户几乎是常数），作用是把 x 的尺度调到
    「相邻两首≈ 0.3~1」——数据里 75% 的相邻间隔落在 1~30 分钟，即 x ≈ 0.3~2.3，周期基/MLP 的分辨力落在这里。
    """
    delta = (times[:, -1:] - times).float()
    return torch.log1p(delta / TIME_UNIT).unsqueeze(-1)


def pairwise_delta(times: torch.Tensor) -> torch.Tensor:
    """(B, L, L)：Δt = t_i − t_j（秒，i = 查询位，j = key 位）；供衰减核用。

    与 `pairwise_buckets` 同一前提：因果 + 左填充下**有效位之间** Δt 恒 ≥ 0（对角线为 0）。
    但 (padding 位, 有效位) 这种组合会给出负数（padding 的时间戳是 0），所以调用方必须自己在 0 处截断：
    `log1p(负的大数)` 是 NaN，而 NaN 加到 −inf 掩码上会把整块掩码污染成 NaN（自检里抓到过）。
    """
    return (times[:, :, None] - times[:, None, :]).float()


def pairwise_buckets(times: torch.Tensor) -> torch.Tensor:
    """(B, 512) 时间戳 → (B, 512, 512) 的 Δt 桶编号，Δt = t_i − t_j（i = 查询位，j = key 位）；供 `USE_TIME_BIAS` 用。

    参照点是**查询位自身**，不是窗口末尾：位置 i 预测 i+1，用到的全是 i 时刻已知的时间，不泄未来。
    因果 + 左填充下 Δt 恒 ≥ 0，所以不需要参考实现那个 abs。
    """
    bounds = torch.tensor(TIME_BOUNDS, dtype=torch.int32, device=times.device)
    # right=True：与 np.digitize 同一套区间约定（见 delta_buckets_tensor）
    return torch.bucketize(times[:, :, None] - times[:, None, :], bounds, right=True, out_int32=True)


class AttentionBlock(nn.Module):
    """一个 post-norm 的 attention block，数学上与 nn.TransformerEncoderLayer 同构。

    自己拼的唯一原因是那头只接受布尔 mask，而我们要往注意力分数上加**加性浮点偏置**（相对时间偏置）。
    子模块命名与 nn.TransformerEncoderLayer 保持一致（self_attn / linear1 / linear2 / norm1 / norm2），
    所以它那套 state_dict 在这儿能直接装——第 1 步就是靠这一点与旧实现逐位对照的。
    """

    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(EMB, HEADS, dropout=DROPOUT, batch_first=True)
        self.linear1 = nn.Linear(EMB, 4 * EMB)
        self.linear2 = nn.Linear(4 * EMB, EMB)
        self.norm1 = nn.LayerNorm(EMB, eps=1e-9)
        self.norm2 = nn.LayerNorm(EMB, eps=1e-9)
        self.dropout = nn.Dropout(DROPOUT)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        attended = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        x = self.norm1(x + self.dropout1(attended))
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x))))))


class Encoder(nn.Module):
    """LAYERS 个 AttentionBlock；属性名取 `layers` 是为了与 nn.TransformerEncoder 的 state_dict 键一致。"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(AttentionBlock() for _ in range(LAYERS))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


class SASRec(nn.Module):
    """官方 SASRecEncoder + 本机加的 Δ 时间 embedding：item + 位置（POS_MODE 决定来源）+ 时间 → LayerNorm → Dropout → 因果 Transformer → 最后一格。"""

    def __init__(self, n_items: int, pad_id: int) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, EMB, padding_idx=pad_id)
        if POS_MODE in ("index", "time", "time_interp"):
            # index 用「第几首」当行号、time 用整数档号、time_interp 用小数档号 —— 三者共用同一张表（参数集完全相同）
            self.position_embedding = nn.Embedding(MAX_SEQ_LEN, EMB)
        elif POS_MODE == "time_mlp":
            self.position_time = nn.Sequential(nn.Linear(1, EMB), nn.GELU(), nn.Linear(EMB, EMB))
        elif POS_MODE == "time_t2v":
            # Bochner / Time2Vec 那一支：**只留周期项** [cos(ωx), sin(ωx)]，不带线性项、不带相位。
            # 线性项 ω·x+φ 的量级就是 x 本身（x 最大 ~11.7）→ 实测初始化时该通道 RMS 2.034，
            # 而 item 通道只有 0.018（**113 倍**），LayerNorm 一归一化就把 item 信号压掉：
            # 这就是 2026-09-22 第一版 T2V 训不起来的原因（loss 停在 0.1587，基线 0.0471）。
            # 相位去掉：cos/sin 同时存在时，相位只是两者的线性组合，投影层自己能表示。
            # 缩放见 forward 里的 sqrt(1/(2·EMB))。
            self.time_omega = nn.Parameter(torch.empty(EMB))
            self.time_proj = nn.Linear(2 * EMB, EMB)
        else:
            raise ValueError(f"POS_MODE 只能是 index / time / time_interp / time_mlp / time_t2v，收到 {POS_MODE!r}")
        if USE_DECAY:
            self.decay_lambda = nn.Parameter(torch.zeros(HEADS))  # 每 head 一个衰减强度
        if USE_TIME_FEATURE:
            self.time_embedding = nn.Embedding(N_TIME_BUCKETS, EMB)  # Δ 时间分桶（本机加的，官方没有）
        if USE_TIME_BIAS:
            self.time_bias = nn.Embedding(N_TIME_BUCKETS, HEADS)  # 相对时间偏置，按 head 分开（本机加的）
        self.layernorm = nn.LayerNorm(EMB, eps=1e-9)
        self.dropout = nn.Dropout(DROPOUT)
        self.encoder = Encoder()
        # 因果掩码注册成 buffer，.to(device) 时才会跟着走
        self.register_buffer(
            "causal", torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool), diagonal=1)
        )
        for name, value in self.named_parameters():  # 官方 _init_weights
            if "weight" in name:
                if "norm" in name:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(value.data, std=INIT_RANGE, a=-2 * INIT_RANGE, b=2 * INIT_RANGE)
            else:
                nn.init.zeros_(value.data)
        if USE_TIME_BIAS:
            nn.init.zeros_(self.time_bias.weight)  # 从「没有偏置」起步：第一轮的行为就是 b ≡ 0 那版，便于对照
        if POS_MODE == "time_t2v":
            # 上面那个通用初始化循环会把 ω 当 bias 清零（名字里没有 "weight"），所以在这里显式给初值。
            # ω log-uniform 铺在数据的 x 质量区间上：x = log1p(Δ/200)，75% 的相邻对落在 x ≈ 0.3~2.3，
            # 取周期 2π/ω ∈ [0.25, 4]（ω ≈ [1.57, 25.1]）→ 既有粗粒度平滑、也有细粒度振荡。
            # 频率初始化**没调过**；要调就是这一处。
            with torch.no_grad():
                low, high = math.log(2 * math.pi / 4), math.log(2 * math.pi / 0.25)
                self.time_omega.copy_(torch.exp(torch.empty(EMB).uniform_(low, high)))
        if USE_DECAY:
            nn.init.zeros_(self.decay_lambda)  # 核 ≡ 0 起步，与基线逐位相同

    def forward(self, item: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """item / times (B, 512) 已左填充 → (B, 512, EMB)。位置编号倒序：最新行为是 0。"""
        if POS_MODE == "index":
            positions = torch.arange(MAX_SEQ_LEN - 1, -1, -1, device=item.device)
            position_part = self.position_embedding(positions).unsqueeze(0)  # (L, EMB)：所有行共用同一份
        elif POS_MODE == "time":
            position_part = self.position_embedding(time_position_index(times))  # (B, L, EMB)：整数分位档
        elif POS_MODE == "time_interp":
            r = time_position_frac(times)  # 小数档号：同一张表按相邻两行线性混合
            i0 = r.floor().clamp(0, MAX_SEQ_LEN - 2).long()
            w = (r - i0).unsqueeze(-1)
            position_part = (1 - w) * self.position_embedding(i0) + w * self.position_embedding(i0 + 1)
        elif POS_MODE == "time_mlp":
            position_part = self.position_time(time_x(times))
        else:  # time_t2v
            angle = self.time_omega * time_x(times)  # (B, L, EMB)
            feat = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
            # sqrt(1/(2·EMB))：Bochner/RFF 的标准归一（features 的 RMS 压到 ~0.09），
            # 顺带把通道幅度对齐到 item 通道（实测 RMS 0.141 → 0.02，item 是 0.018）
            position_part = self.time_proj(feat * math.sqrt(1.0 / (2 * EMB)))
        embeddings = self.item_embedding(item) + position_part
        if USE_TIME_FEATURE:
            embeddings = embeddings + self.time_embedding(delta_buckets_tensor(times))
        embeddings = self.dropout(self.layernorm(embeddings))
        # 加性浮点 mask：未来位与 padding 位填 -inf，等价于原先的（布尔因果掩码 + src_key_padding_mask）。
        # 建在 (B, heads, 512, 512) 上是为了后面能把相对时间偏置 in-place 加到同一块内存里，不必再要一份同尺寸张量。
        attn_mask = torch.zeros(
            (item.shape[0], HEADS, MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=embeddings.dtype, device=item.device
        )
        attn_mask.masked_fill_(self.causal, float("-inf"))
        attn_mask.masked_fill_((~mask)[:, None, None, :], float("-inf"))
        # 给 padding 行留一个可见位（它自己的对角线）：整行都被 mask 掉时 softmax 是 0/0。
        # 这不是洁癖——2026-09-21 査出：旧写法（布尔 mask + src_key_padding_mask）在 eval 的 MHA fused kernel 下
        # 给整行被 mask 的 padding 行返回 NaN（train 模式走另一条路径，所以训练看起来正常），那些 NaN 又在第二层
        # 当 K/V 把有效行的输出也污染成 NaN——9,207 个用户里 2,321 个（历史 < 512 的）中招，评测数字被压低 18% 左右。
        # padding 行的输出反正不会被读（loss 只取 mask 位、评测只取最后一格），留个可见位只是为了让它保持有限值。
        attn_mask.diagonal(dim1=-2, dim2=-1).masked_fill_(
            (~mask)[:, None, :].expand(-1, HEADS, -1), 0.0
        )
        if USE_TIME_BIAS:
            # b[Δt 桶] 加到注意力分数上（不是取代点积）。Δt 的桶由原始时间戳按查询位参照现算，各层共用这一份：
            # 每层重算等于把参考书里讲的 O(N²) 访存瓶颈乘上层数。
            attn_mask += self.time_bias(pairwise_buckets(times)).permute(0, 3, 1, 2)
        if USE_DECAY:
            # −λ_head · log1p(Δt)：λ 初值 0 → 起步与基线逐位相同；被 mask 的 key 是 −inf，加完仍是 −inf。
            # Δt 必须先在 0 处截断：padding 行的时间戳是 0，与有效位相减得到负数，而 log1p(负数大) 是 NaN，
            # NaN 加到 −inf 上会把整块掩码变成 NaN（自检里就是这么抓到的）。
            attn_mask += -self.decay_lambda.view(1, -1, 1, 1) * torch.log1p(pairwise_delta(times).clamp(min=0.0)).unsqueeze(1)
        return self.encoder(embeddings * mask.unsqueeze(-1), attn_mask.view(-1, MAX_SEQ_LEN, MAX_SEQ_LEN))


class TrainDataset(Dataset):
    """每个样本 = 一条序列的 (item, positive, negative, times)，四者在同一批位置上一一对应。"""

    def __init__(self, sequences: list[np.ndarray], times: list[np.ndarray], n_items: int) -> None:
        keep = [(seq, stamp) for seq, stamp in zip(sequences, times) if len(seq) >= 2]  # 至少两个物品才有“下一首”
        self.sequences = [seq for seq, _ in keep]
        self.times = [stamp for _, stamp in keep]
        self.n_items = n_items

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sequence = self.sequences[index]
        item = sequence[:-1]
        positive = sequence[1:]
        negative = np.random.randint(0, self.n_items, size=item.shape)  # 官方 randint(1, num_items+1)
        # 原始时间戳（与 item 同一个位置对齐）：Δ 与成对 Δt 都在模型里现算
        stamps = self.times[index][:-1].astype(np.int32)
        return item, positive, negative, stamps


def collate(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], pad_id: int):
    item = np.full((len(batch), MAX_SEQ_LEN), pad_id, dtype=np.int64)
    positive = np.full_like(item, pad_id)
    negative = np.full_like(item, pad_id)
    times = np.zeros((len(batch), MAX_SEQ_LEN), dtype=np.int32)
    mask = np.zeros((len(batch), MAX_SEQ_LEN), dtype=bool)
    for row, (sample_item, sample_positive, sample_negative, sample_times) in enumerate(batch):
        n = len(sample_item)
        item[row, MAX_SEQ_LEN - n :] = sample_item
        positive[row, MAX_SEQ_LEN - n :] = sample_positive
        negative[row, MAX_SEQ_LEN - n :] = sample_negative
        times[row, MAX_SEQ_LEN - n :] = sample_times
        mask[row, MAX_SEQ_LEN - n :] = True
    return (
        torch.from_numpy(item),
        torch.from_numpy(positive),
        torch.from_numpy(negative),
        torch.from_numpy(times),
        torch.from_numpy(mask),
    )


def train(sequences: list[np.ndarray], times: list[np.ndarray], n_items: int, pad_id: int) -> SASRec:
    dataset = TrainDataset(sequences, times, n_items)
    print(f"训练样本（训练集里 ≥ 2 个物品的用户）: {len(dataset):,} / {len(sequences):,}")
    loader = DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda batch: collate(batch, pad_id),
    )
    model = SASRec(n_items, pad_id).to(TRAIN_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        total = 0.0
        for item, positive, negative, times, mask in loader:
            item, positive, negative, times, mask = (
                tensor.to(TRAIN_DEVICE) for tensor in (item, positive, negative, times, mask)
            )
            # bf16 自混精只包住训练步；enabled 里带设备判断，免得在没有 MPS 的机器上误开 CPU 的 bf16
            with torch.autocast(device_type=TRAIN_DEVICE, dtype=torch.bfloat16, enabled=USE_BF16 and TRAIN_DEVICE == "mps"):
                query = model(item, times, mask)[mask]
                positive_scores = (query * model.item_embedding(positive[mask])).sum(dim=-1)
                negative_scores = (query * model.item_embedding(negative[mask])).sum(dim=-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    torch.cat([positive_scores, negative_scores]),
                    torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]),
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        # loss 后面这个等号是给 bg-train 的 sniffer 认的（它的 KV 模式是 `loss=`），不是装饰
        print(f"  epoch {epoch:>2}/{EPOCHS}  loss={total / len(loader):.6f}  ({time.perf_counter() - started:.1f}s)")
    return model


def top_k(user_vectors: np.ndarray, item_vectors: np.ndarray, batch: int = 128) -> np.ndarray:
    """用户向量 × 物品向量，取每个用户的 top-100（同分按 item id 升序）。"""
    tops = np.empty((user_vectors.shape[0], SELECT_K), dtype=np.int32)
    for start in range(0, user_vectors.shape[0], batch):
        stop = min(start + batch, user_vectors.shape[0])
        scores = user_vectors[start:stop] @ item_vectors.T
        rows = np.arange(stop - start)[:, None]
        part = np.argpartition(-scores, SELECT_K, axis=1)[:, :SELECT_K]
        tops[start:stop] = part[rows, np.lexsort((part, -scores[rows, part]), axis=1)]
    return tops


def main() -> None:
    global TIME_EDGES  # POS_MODE="time" 时在这里把训练集的等量分位边界装好（见 build_time_edges）
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")

    train_frame = load("train", ["uid", "item_id", "timestamp"])
    target = load(EVAL_SPLIT, ["uid", "item_id", "is_organic", "played_ratio_pct"])
    n_users = int(train_frame.uid.max()) + 1
    n_items = int(train_frame.item_id.max()) + 1
    pad_id = n_items  # 官方用 0（它的词表是 1-based）；我们的 0 是真实物品，只能另拿一个 id 当 padding
    print(
        f"口径 {SPLITS_ID.upper()}；train {len(train_frame):,} 行 / {n_users:,} 用户 / {n_items:,} 候选物品；"
        f"设备：训练 {TRAIN_DEVICE}、推理 {INFER_DEVICE}"
    )

    sequences = train_sequences(train_frame, n_users)
    times = train_timestamps(train_frame, n_users)
    assert all(len(sequence) for sequence in sequences), "每个 uid 都应有训练序列（否则 padding 掩码会全 True）"
    lengths = np.array([len(sequence) for sequence in sequences])
    delta_demo = delta_buckets(times[int(np.argmax(lengths))])
    print(f"序列长度: 中位 {int(np.median(lengths))} / 最大 {lengths.max()} / 截断到 {MAX_SEQ_LEN}")
    print(f"时间特征: {N_TIME_BUCKETS} 个对数桶，边界 {TIME_BOUNDS}；最长序列的 Δ 桶分布前 10 = {delta_demo[:10].tolist()}")
    if TRAIN_FIRST and POS_MODE != "index":
        # 开工前只看两件事（0 成本）：① 位置通道的档位占用（防「92.7% 挤一档」那种退化）；
        # ② **每用户可区分的位置比例**（分位档 0.11 / 连续 0.96 —— 这条是 2026-09-22 才学会要量的）。
        all_users = np.arange(n_users)
        _, mask_all = pad_sequences(sequences, all_users, pad_id)
        all_times = pad_times(times, all_users)
        if POS_MODE in ("time", "time_interp"):
            TIME_EDGES = build_time_edges(times)
            deltas = np.concatenate([(t[-1] - t).astype(np.int64) for t in times if len(t)])
            buckets = np.searchsorted(TIME_EDGES.numpy(), deltas, side="left")  # 与 torch.bucketize 同一区间约定
            share = np.bincount(buckets, minlength=MAX_SEQ_LEN) / deltas.size
            picked = [int(v) for v in TIME_EDGES[[0, 1, 127, 255, 383, 510]]]
            print(f"位置档占用（{deltas.size:,} 个训练位置 / {MAX_SEQ_LEN} 档）：最大档 {int(share.argmax())} = {share.max():.2%}，"
                  f"用到的档 {int((share > 0).sum())}，头两档 {share[0]:.2%}/{share[1]:.2%}，末档 {share[-1]:.2%}")
            print(f"分位边界（秒，取 6 个代表）：{picked}；其余 {MAX_SEQ_LEN - 7} 个随 checkpoint 存档")
            assert share.max() < 0.30, (
                f"最大档占比 {share.max():.1%} ≥ 30%：位置档退化（线性刻度那版是 92.7% 挤一档）。先修刻度再训"
            )
        with torch.no_grad():
            if POS_MODE == "time":
                code = time_position_index(all_times)
            elif POS_MODE == "time_interp":
                code = time_position_frac(all_times)
            else:
                code = time_x(all_times)[..., 0]
            code, m = code.numpy(), mask_all.numpy()
        distinct = np.array([len(np.unique(code[i][m[i]])) / max(int(m[i].sum()), 1) for i in range(n_users)])
        print(f"位置通道（POS_MODE={POS_MODE}）每用户可区分的位置比例：中位 {np.median(distinct):.2f} / "
              f"p10 {np.percentile(distinct, 10):.2f}（index 基线 = 1.00）")
    if USE_TIME_BIAS:
        # 开工前先看 Δt 桶的占用：确认 13 个桶都有样本、长桶不是只由 padding 撑起来的
        sample = np.arange(min(64, n_users))
        sample_mask = pad_sequences(sequences, sample, pad_id)[1].numpy()
        pairs = (
            sample_mask[:, :, None]
            & sample_mask[:, None, :]
            & np.tril(np.ones((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=bool))  # 因果：只算 j ≤ i
        )
        counts = np.bincount(pairwise_buckets(pad_times(times, sample)).numpy()[pairs], minlength=N_TIME_BUCKETS)
        share = counts / counts.sum()
        print(f"Δt 桶占用（前 {len(sample)} 个用户、两端都有效的配对共 {counts.sum():,} 对）:")
        print("  " + "  ".join(f"{index}:{value:.1%}" for index, value in enumerate(share)))

    if TRAIN_FIRST:
        model = train(sequences, times, n_items, pad_id)
        model.to(INFER_DEVICE)  # 先搬回 CPU 再存，免得 checkpoint 绑定 MPS
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        # POS_MODE 与分位边界都写进 checkpoint：两种模式参数形状相同，形状守卫拦不住「装错了位置通道的权重」
        payload = {**model.state_dict(), "_pos_mode": POS_MODE}
        if POS_MODE in ("time", "time_interp"):
            payload["_time_edges"] = TIME_EDGES  # 让评测/复评用同一套分档边界，而不是另算一套
        torch.save(payload, CHECKPOINT)
        print(f"checkpoint: {CHECKPOINT.relative_to(ROOT)}")
    else:
        model = SASRec(n_items, pad_id).to(INFER_DEVICE)
        state = torch.load(CHECKPOINT, map_location=INFER_DEVICE)
        # 老 checkpoint 没有这个键（那时只有 index 一种模式），默认值就是它真实的值
        saved_mode = state.pop("_pos_mode", "index")
        assert saved_mode == POS_MODE, (
            f"{CHECKPOINT.name} 是 POS_MODE={saved_mode!r} 训出来的，而当前 POS_MODE={POS_MODE!r}："
            "两种模式参数形状相同、装错了不会报错，只会静默算出错的分数 → 换模式后要按当前模式重训"
        )
        saved_edges = state.pop("_time_edges", None)
        if POS_MODE in ("time", "time_interp"):
            assert saved_edges is not None, (
                f"{CHECKPOINT.name} 里没有分档边界（_time_edges）：这一版要按当前代码重训一次，别用历史 checkpoint 复评"
            )
            TIME_EDGES = saved_edges
        assert USE_DECAY == ("decay_lambda" in state), (
            f"{CHECKPOINT.name} 的衰减核与当前 USE_DECAY={USE_DECAY} 不一致：先重训或把开关改回来"
        )
        saved_emb = state["item_embedding.weight"].shape[1]
        if POS_MODE in ("index", "time", "time_interp"):
            saved_len = state["position_embedding.weight"].shape[0]
            assert saved_len == MAX_SEQ_LEN, (
                f"{CHECKPOINT.name} 的位置 embedding 是 {saved_len} 行，而 MAX_SEQ_LEN = {MAX_SEQ_LEN}："
                "换过序列长度后要先按当前配置重训一次（对应模式会覆盖该 checkpoint），再回来评测"
            )
        assert saved_emb == EMB, (
            f"{CHECKPOINT.name} 的 embedding 维度是 {saved_emb}，而 EMB = {EMB}："
            "换过维度后要先按当前配置重训一次（对应模式会覆盖该 checkpoint），再回来评测"
        )
        assert USE_TIME_FEATURE == ("time_embedding.weight" in state), (
            f"{CHECKPOINT.name} 的时间特征与当前 USE_TIME_FEATURE={USE_TIME_FEATURE} 不一致：先重训或把开关改回来"
        )
        assert USE_TIME_BIAS == ("time_bias.weight" in state), (
            f"{CHECKPOINT.name} 的相对时间偏置与当前 USE_TIME_BIAS={USE_TIME_BIAS} 不一致：先重训或把开关改回来"
        )
        saved_layers = len({key.split(".")[2] for key in state if key.startswith("encoder.layers.")})
        assert saved_layers == LAYERS, (
            f"{CHECKPOINT.name} 是 {saved_layers} 层，而 LAYERS = {LAYERS}：换过层数后要先按当前配置重训一次"
        )
        model.load_state_dict(state)
        print(f"复用 checkpoint: {CHECKPOINT.relative_to(ROOT)}（换 split 不重训）")

    if USE_TIME_BIAS:
        # 只有 13 × heads 个数，直接打出来看它有没有学出「越久远越不重要」的单调衰减
        print(f"b[Δt 桶]（桶 0 最近，{N_TIME_BUCKETS} 桶 × {HEADS} head）:")
        for index, row in enumerate(model.time_bias.weight.detach().cpu().numpy()):
            print(f"  {index:>2}  " + "  ".join(f"{value:+.5f}" for value in row))

    target_users, target_chunks = split_by_user(target.uid.to_numpy(), target.item_id.to_numpy())
    print(f"{EVAL_SPLIT:<5} {len(target):,} 行 / {len(target_users):,} 个有目标的用户")

    # 每用户训练期交互过的物品：用**完整 train**，不是截断后的序列——否则超过 MAX_SEQ_LEN 的那些历史
    # 会被当成“没交过”，回访目标被误判成新歌；uid 空间稠密，所以列表下标就是 uid
    train_history = [
        np.unique(chunk) for chunk in split_by_user(train_frame.uid.to_numpy(), train_frame.item_id.to_numpy())[1]
    ]
    target_history = [train_history[u] for u in target_users]

    # 分层轴（is_organic / 参与度 / 回访×来源）：口径见 rec_eval.py 文件头，列来自 2026-09-22 补进 splits 的三个字段
    back = history_flags(
        target.uid.to_numpy(), target.item_id.to_numpy(), train_frame.uid.to_numpy(), train_frame.item_id.to_numpy()
    )
    axes = target_axes(
        target.uid.to_numpy(), target.is_organic.to_numpy(), target.played_ratio_pct.to_numpy(), back
    )

    model.eval()
    item_tensor, mask = pad_sequences(sequences, np.arange(n_users), pad_id)
    times_tensor = pad_times(times, np.arange(n_users))
    user_vectors = np.empty((n_users, EMB), dtype=np.float32)
    with torch.no_grad():
        # 按用户分批：模型对每个用户独立，分批不改变结果，只是把 (B, heads, L, L) 的注意力矩阵压到能放下
        for start in range(0, n_users, INFER_BATCH):
            stop = min(start + INFER_BATCH, n_users)
            embeddings = model(
                item_tensor[start:stop].to(INFER_DEVICE),
                times_tensor[start:stop].to(INFER_DEVICE),
                mask[start:stop].to(INFER_DEVICE),
            )
            # 左填充 → 最后一格就是最新行为
            user_vectors[start:stop] = embeddings[:, -1, :].float().cpu().numpy()
    item_vectors = model.item_embedding.weight[:n_items].detach().float().cpu().numpy()  # 排除 padding 行
    tops = top_k(user_vectors, item_vectors)

    # tops 按 uid 行序给，chunks 按目标 split 的分组序 —— 必须对齐（见 architecture.md）
    metrics, n_targets = evaluate(
        list(tops[target_users]), target_chunks, n_items, all_tops=tops, history=target_history, axes=axes
    )
    show(f"[{EVAL_SPLIT}] 官方口径（不过滤已交互物品、目标保留不可排名行）", metrics, n_targets, n_items)
    print(f"\n总耗时 {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
