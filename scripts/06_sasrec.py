"""sasrec 基线：自己实现一份，训练与评测口径对齐官方 benchmark。

运行：
    uv run python scripts/06_sasrec.py          # 在 val 上训练并评测（50 epoch 约 5.5 分钟）
    uv run python scripts/06_sasrec.py test     # 在 test 上评测，复用 val 那次存下的 checkpoint（约 1 分钟）
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
- 训练样本（`data.py::TrainDataset`）：item = seq[:-1]、positive = seq[1:]（每个位置都预测下一首），
  负样本**每个位置随机 1 个**；损失 = BCEWithLogits([正分; 负分], [1; 0])
- 超参（`train.py` 的默认值）：heads 2 / layers 2 / dropout 0.0 / lr 1e-3 / Adam / batch 256 / seed 42；
  epoch 数官方默认 100，本机按 `docs/benchmark_repro.md` 记录的 50。**偏离官方默认的有两处**：
  序列长度 512（官方 200，动机与效果见 `docs/baselines.md`）、embedding 维度 256（官方 64，扩容实验）——
  也就是说本机现在这份 sasrec 已经不是“官方复现”配置，对表能力由口径 B + 文档记录保留。
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

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rec_eval import evaluate, show, split_by_user

ROOT = Path(__file__).resolve().parents[1]
SPLITS_ID = sys.argv[2] if len(sys.argv) > 2 else "a"  # 口径：a = 第一遍（有 val），b = 第二遍（val_size = 0）
assert SPLITS_ID in ("a", "b"), f"口径只能是 a 或 b，收到 {SPLITS_ID!r}"
SPLITS = ROOT / "artifacts" / ("splits" if SPLITS_ID == "a" else "splits_b")
CHECKPOINT = ROOT / "artifacts" / "sasrec" / ("state.pt" if SPLITS_ID == "a" else "state_b.pt")

MAX_SEQ_LEN = 512  # 官方默认 200；本机改 512 —— 训练集历史长度（去重物品）中位 666 / p90 2,239（docs/eda.md），
                   # 截到 200 时一半以上用户的历史被砍掉（实测：两个口径下序列长度中位都正好 = 200，即顶到上限）
EMB = 64  # 官方默认 64（d256 扩容实验的结论见 docs/baselines.md；当前做时间特征，回 64 以便快迭代）
HEADS = 2
LAYERS = 2
DROPOUT = 0.0
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
    """每用户一条 timestamp 序列（与 train_sequences 同样的截断），供 Δ 时间特征用。"""
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


def pad_deltas(times: list[np.ndarray], users: np.ndarray) -> torch.Tensor:
    """左填充到 MAX_SEQ_LEN 的 Δ 桶编号（与 pad_sequences 同样的对齐；padding 位填 0，反正会被 mask 掉）。"""
    delta = np.zeros((len(users), MAX_SEQ_LEN), dtype=np.int64)
    for row, uid in enumerate(users):
        buckets = delta_buckets(np.asarray(times[uid]))
        delta[row, MAX_SEQ_LEN - len(buckets) :] = buckets
    return torch.from_numpy(delta)


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
    """官方 SASRecEncoder + 本机加的 Δ 时间 embedding：item + 位置 + 时间 → LayerNorm → Dropout → 因果 Transformer → 最后一格。"""

    def __init__(self, n_items: int, pad_id: int) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, EMB, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, EMB)
        if USE_TIME_FEATURE:
            self.time_embedding = nn.Embedding(N_TIME_BUCKETS, EMB)  # Δ 时间分桶（本机加的，官方没有）
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

    def forward(self, item: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """item / delta (B, 200) 已左填充 → (B, 200, EMB)。位置倒序编号：最新行为是 0。"""
        positions = torch.arange(MAX_SEQ_LEN - 1, -1, -1, device=item.device)
        embeddings = self.item_embedding(item) + self.position_embedding(positions)
        if USE_TIME_FEATURE:
            embeddings = embeddings + self.time_embedding(delta)
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
        return self.encoder(embeddings * mask.unsqueeze(-1), attn_mask.view(-1, MAX_SEQ_LEN, MAX_SEQ_LEN))


class TrainDataset(Dataset):
    """每个样本 = 一条序列的 (item, positive, negative, delta)，四者在同一批位置上一一对应。"""

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
        # Δ 以**这条输入序列（= 窗口去掉最后一条）的最后一次事件**为参照，不是整条窗口：
        # 这样训练与推理的查询位置都恰好是 Δ = 0，口径一致
        delta = delta_buckets(self.times[index][:-1])
        return item, positive, negative, delta


def collate(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], pad_id: int):
    item = np.full((len(batch), MAX_SEQ_LEN), pad_id, dtype=np.int64)
    positive = np.full_like(item, pad_id)
    negative = np.full_like(item, pad_id)
    delta = np.zeros_like(item)
    mask = np.zeros((len(batch), MAX_SEQ_LEN), dtype=bool)
    for row, (sample_item, sample_positive, sample_negative, sample_delta) in enumerate(batch):
        n = len(sample_item)
        item[row, MAX_SEQ_LEN - n :] = sample_item
        positive[row, MAX_SEQ_LEN - n :] = sample_positive
        negative[row, MAX_SEQ_LEN - n :] = sample_negative
        delta[row, MAX_SEQ_LEN - n :] = sample_delta
        mask[row, MAX_SEQ_LEN - n :] = True
    return (
        torch.from_numpy(item),
        torch.from_numpy(positive),
        torch.from_numpy(negative),
        torch.from_numpy(delta),
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
        for item, positive, negative, delta, mask in loader:
            item, positive, negative, delta, mask = (
                tensor.to(TRAIN_DEVICE) for tensor in (item, positive, negative, delta, mask)
            )
            query = model(item, delta, mask)[mask]
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
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")

    train_frame = load("train", ["uid", "item_id", "timestamp"])
    target = load(EVAL_SPLIT, ["uid", "item_id"])
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

    if TRAIN_FIRST:
        model = train(sequences, times, n_items, pad_id)
        model.to(INFER_DEVICE)  # 先搬回 CPU 再存，免得 checkpoint 绑定 MPS
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"checkpoint: {CHECKPOINT.relative_to(ROOT)}")
    else:
        model = SASRec(n_items, pad_id).to(INFER_DEVICE)
        state = torch.load(CHECKPOINT, map_location=INFER_DEVICE)
        saved_len = state["position_embedding.weight"].shape[0]
        saved_emb = state["item_embedding.weight"].shape[1]
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
        model.load_state_dict(state)
        print(f"复用 checkpoint: {CHECKPOINT.relative_to(ROOT)}（换 split 不重训）")

    target_users, target_chunks = split_by_user(target.uid.to_numpy(), target.item_id.to_numpy())
    print(f"{EVAL_SPLIT:<5} {len(target):,} 行 / {len(target_users):,} 个有目标的用户")

    # 每用户训练期交互过的物品：用**完整 train**，不是截断后的序列——否则超过 MAX_SEQ_LEN 的那些历史
    # 会被当成“没交过”，回访目标被误判成新歌；uid 空间稠密，所以列表下标就是 uid
    train_history = [
        np.unique(chunk) for chunk in split_by_user(train_frame.uid.to_numpy(), train_frame.item_id.to_numpy())[1]
    ]
    target_history = [train_history[u] for u in target_users]

    model.eval()
    item_tensor, mask = pad_sequences(sequences, np.arange(n_users), pad_id)
    delta_tensor = pad_deltas(times, np.arange(n_users))
    user_vectors = np.empty((n_users, EMB), dtype=np.float32)
    with torch.no_grad():
        # 按用户分批：模型对每个用户独立，分批不改变结果，只是把 (B, heads, L, L) 的注意力矩阵压到能放下
        for start in range(0, n_users, INFER_BATCH):
            stop = min(start + INFER_BATCH, n_users)
            embeddings = model(
                item_tensor[start:stop].to(INFER_DEVICE),
                delta_tensor[start:stop].to(INFER_DEVICE),
                mask[start:stop].to(INFER_DEVICE),
            )
            # 左填充 → 最后一格就是最新行为
            user_vectors[start:stop] = embeddings[:, -1, :].float().cpu().numpy()
    item_vectors = model.item_embedding.weight[:n_items].detach().float().cpu().numpy()  # 排除 padding 行
    tops = top_k(user_vectors, item_vectors)

    # tops 按 uid 行序给，chunks 按目标 split 的分组序 —— 必须对齐（见 architecture.md）
    metrics, n_targets = evaluate(
        list(tops[target_users]), target_chunks, n_items, all_tops=tops, history=target_history
    )
    show(f"[{EVAL_SPLIT}] 官方口径（不过滤已交互物品、目标保留不可排名行）", metrics, n_targets, n_items)
    print(f"\n总耗时 {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
