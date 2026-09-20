"""sasrec 基线：自己实现一份，训练与评测口径对齐官方 benchmark。

运行：
    uv run python scripts/06_sasrec.py

口径出处（vendor/yambda-benchmarks/benchmarks/models/sasrec/）：
- 序列：每用户一条 item 序列（时间升序），取最后 max_seq_len=200 条（`data.py::preprocess`）
- 模型（`model.py::SASRecEncoder`）：item embedding + 位置 embedding（倒序，最新为 0）→ LayerNorm →
  Dropout → 2 层 TransformerEncoder（因果掩码 + padding 掩码）→ 取每序列最后一个有效位置的向量
- 训练样本（`data.py::TrainDataset`）：item = seq[:-1]、positive = seq[1:]（每个位置都预测下一首），
  负样本**每个位置随机 1 个**；损失 = BCEWithLogits([正分; 负分], [1; 0])
- 超参（`train.py` 的默认值）：emb 64 / heads 2 / layers 2 / dropout 0.0 / lr 1e-3 / Adam /
  batch 256 / seed 42 / seq 200；epoch 数官方默认 100，本机按 `docs/benchmark_repro.md` 记录的 50
- 评测（`eval.py`）：用户向量 × 全部 item embedding 内积 → top-100，指标走 `scripts/rec_eval.py`
- 设备：训练用 MPS，**推理必须用 CPU**（官方在 MPS 上跑 eval 会崩，见 `docs/benchmark_repro.md`）

与官方实现的三处已知差异：
1. 词表与 padding：官方词表 = 切分前整个文件里出现过的物品、编号从 1 开始（0 留给 padding），
   排序时连 padding 行一起当候选；我们用现有 id 空间 0..627,647，padding id = 627,648，
   **排序时排除 padding** → 候选池 627,648，与 popularity / itemknn 完全一致（官方多一个候选）。
2. 位置编号：官方是 arange(S-1, -1, -1) 再按长度掩码，S 是**该 batch 内的最大长度**，而 eval 的
   DataLoader 还 shuffle=True → 同一个用户在不同 batch 组成下位置编号会变；我们固定左填充到 200、
   最新行为是位置 0，可复现（序列满 200 时两者完全一致，只有短序列会被平移）。
3. 切分：官方 train.py / eval.py 都用 val_size=0（第二遍训练集）并在 test 上报告；我们在第一遍切分的
   train 上训练、val 上评测，与 popularity / itemknn 同一口径。

注意：训练带随机性（初始化、负采样、数据顺序），不追求与官方逐位一致，只对量级。
"""

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from rec_eval import evaluate, show, split_by_user

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "artifacts" / "splits"
CHECKPOINT = ROOT / "artifacts" / "sasrec" / "state.pt"

MAX_SEQ_LEN = 200
EMB = 64
HEADS = 2
LAYERS = 2
DROPOUT = 0.0
LR = 1e-3
BATCH = 256
SEED = 42
EPOCHS = 50  # 官方默认 100；本机按 docs/benchmark_repro.md 记录的 50
INIT_RANGE = 0.02  # model.py::_init_weights
SELECT_K = 100

TRAIN_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
INFER_DEVICE = "cpu"  # 官方 eval 在 MPS 上会崩


def load(name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_parquet(SPLITS / f"{name}.parquet", columns=columns)


def train_sequences(train: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """每用户一条时间升序的 item 序列（截到最后 MAX_SEQ_LEN 条）。"""
    users, chunks = split_by_user(train.uid.to_numpy(), train.item_id.to_numpy())
    sequences: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_users
    for uid, chunk in zip(users, chunks):
        sequences[uid] = chunk[-MAX_SEQ_LEN:].astype(np.int64)
    return sequences


def pad_sequences(sequences: list[np.ndarray], users: np.ndarray, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """左填充到 MAX_SEQ_LEN（最新行为落在最后一格 = 位置 0）；返回 (item, mask)。"""
    item = np.full((len(users), MAX_SEQ_LEN), pad_id, dtype=np.int64)
    mask = np.zeros((len(users), MAX_SEQ_LEN), dtype=bool)
    for row, uid in enumerate(users):
        sequence = sequences[uid][-MAX_SEQ_LEN:]
        item[row, MAX_SEQ_LEN - len(sequence) :] = sequence
        mask[row, MAX_SEQ_LEN - len(sequence) :] = True
    return torch.from_numpy(item), torch.from_numpy(mask)


class SASRec(nn.Module):
    """官方 SASRecEncoder：item + 位置 embedding → LayerNorm → Dropout → 因果 Transformer → 最后一格。"""

    def __init__(self, n_items: int, pad_id: int) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(n_items + 1, EMB, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(MAX_SEQ_LEN, EMB)
        self.layernorm = nn.LayerNorm(EMB, eps=1e-9)
        self.dropout = nn.Dropout(DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=EMB,
            nhead=HEADS,
            dim_feedforward=4 * EMB,
            dropout=DROPOUT,
            activation=nn.GELU(),
            layer_norm_eps=1e-9,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, LAYERS)
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

    def forward(self, item: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """item (B, 200) 已左填充 → (B, 200, EMB)。位置倒序编号：最新行为是 0。"""
        positions = torch.arange(MAX_SEQ_LEN - 1, -1, -1, device=item.device)
        embeddings = self.item_embedding(item) + self.position_embedding(positions)
        embeddings = self.dropout(self.layernorm(embeddings))
        return self.encoder(
            embeddings * mask.unsqueeze(-1), mask=self.causal, src_key_padding_mask=~mask
        )


class TrainDataset(Dataset):
    """每个样本 = 一条序列的 (item, positive, negative)，三者在同一批位置上一一对应。"""

    def __init__(self, sequences: list[np.ndarray], n_items: int) -> None:
        self.samples = [sequence for sequence in sequences if len(sequence) >= 2]  # 至少要两个物品才有"下一首"
        self.n_items = n_items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence = self.samples[index]
        item = sequence[:-1]
        positive = sequence[1:]
        negative = np.random.randint(0, self.n_items, size=item.shape)  # 官方 randint(1, num_items+1)
        return item, positive, negative


def collate(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray]], pad_id: int):
    item = np.full((len(batch), MAX_SEQ_LEN), pad_id, dtype=np.int64)
    positive = np.full_like(item, pad_id)
    negative = np.full_like(item, pad_id)
    mask = np.zeros((len(batch), MAX_SEQ_LEN), dtype=bool)
    for row, (sample_item, sample_positive, sample_negative) in enumerate(batch):
        n = len(sample_item)
        item[row, MAX_SEQ_LEN - n :] = sample_item
        positive[row, MAX_SEQ_LEN - n :] = sample_positive
        negative[row, MAX_SEQ_LEN - n :] = sample_negative
        mask[row, MAX_SEQ_LEN - n :] = True
    return (torch.from_numpy(item), torch.from_numpy(positive), torch.from_numpy(negative), torch.from_numpy(mask))


def train(sequences: list[np.ndarray], n_items: int, pad_id: int) -> SASRec:
    dataset = TrainDataset(sequences, n_items)
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
        for item, positive, negative, mask in loader:
            item, positive, negative, mask = (
                tensor.to(TRAIN_DEVICE) for tensor in (item, positive, negative, mask)
            )
            query = model(item, mask)[mask]
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
        print(f"  epoch {epoch:>2}/{EPOCHS}  loss {total / len(loader):.6f}  ({time.perf_counter() - started:.1f}s)")
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

    train_frame = load("train", ["uid", "item_id"])
    val = load("val", ["uid", "item_id"])
    n_users = int(train_frame.uid.max()) + 1
    n_items = int(train_frame.item_id.max()) + 1
    pad_id = n_items  # 官方用 0（它的词表是 1-based）；我们的 0 是真实物品，只能另拿一个 id 当 padding
    print(
        f"train {len(train_frame):,} 行 / {n_users:,} 用户 / {n_items:,} 候选物品；"
        f"设备：训练 {TRAIN_DEVICE}、推理 {INFER_DEVICE}"
    )

    sequences = train_sequences(train_frame, n_users)
    assert all(len(sequence) for sequence in sequences), "每个 uid 都应有训练序列（否则 padding 掩码会全 True）"
    lengths = np.array([len(sequence) for sequence in sequences])
    print(f"序列长度: 中位 {int(np.median(lengths))} / 最大 {lengths.max()} / 截断到 {MAX_SEQ_LEN}")

    model = train(sequences, n_items, pad_id)
    model.to(INFER_DEVICE)  # 先搬回 CPU 再存，免得 checkpoint 绑定 MPS
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT)
    print(f"checkpoint: {CHECKPOINT.relative_to(ROOT)}")

    val_users, val_chunks = split_by_user(val.uid.to_numpy(), val.item_id.to_numpy())
    print(f"val   {len(val):,} 行 / {len(val_users):,} 个有目标的用户")

    model.eval()
    item_tensor, mask = pad_sequences(sequences, np.arange(n_users), pad_id)
    with torch.no_grad():
        embeddings = model(item_tensor.to(INFER_DEVICE), mask.to(INFER_DEVICE))
    user_vectors = embeddings[:, -1, :].float().cpu().numpy()  # 左填充 → 最后一格就是最新行为
    item_vectors = model.item_embedding.weight[:n_items].detach().float().cpu().numpy()  # 排除 padding 行
    tops = top_k(user_vectors, item_vectors)

    # tops 按 uid 行序给，chunks 按 val 分组序 —— 必须对齐（见 architecture.md）
    metrics, n_targets = evaluate(list(tops[val_users]), val_chunks, n_items)
    show("[val] 官方口径（不过滤已交互物品、目标保留不可排名行）", metrics, n_targets, n_items)
    print(f"\n总耗时 {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
