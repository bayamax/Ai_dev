#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pair_classifier_stream.py
投稿ベクトル ⇔ アカウントベクトルのペアを
正例(同一UID)=1／負例(異UID)=0 で学習するストリーミング版。
・UID ハッシュで 90 % / 10 % の train / val split
・進捗を tqdm で表示
・--resume でチェックポイント継続学習
・MLP に Dropout を追加（DROPOUT_RATE で制御）
"""

import os, csv, random, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# ─────────── ファイルパス ───────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CHECKPOINT_PATH = os.path.join(VAST_DIR, "pair_classifier_rw_stream.ckpt")

# ─────────── ハイパーパラメータ ───────────
POST_DIM      = 3072
BATCH_SIZE    = 128
EPOCHS        = 500
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
NEG_RATIO     = 5      # 正例 1 本につき負例 1 本
VAL_RATIO     = 0.10   # UID の 10 % をバリデーションに
DROPOUT_RATE  = 0.3    # ★ 追加：MLP 内ドロップアウト
PATIENCE      = 15
MIN_DELTA     = 1e-4

# ────────────────────────────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l + 1 : r]
    arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None


def uid_to_val(uid: str, ratio: float) -> bool:
    """UID をハッシュして一貫した train/val 割り当てを返す"""
    h = int(hashlib.md5(uid.encode('utf-8')).hexdigest(), 16)
    return (h % 10_000) / 10_000.0 < ratio


# ─────────── データセット ───────────
class PairStreamDataset(IterableDataset):
    def __init__(self, posts_csv, account_npy, split="train"):
        assert split in ("train", "val")
        self.posts_csv = posts_csv
        self.rw_dict   = np.load(account_npy, allow_pickle=True).item()
        self.uids      = list(self.rw_dict.keys())
        self.split     = split
        self.rw_dim    = next(iter(self.rw_dict.values())).shape[0]

    def __iter__(self):
        with open(self.posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f)
            next(rdr)  # ヘッダスキップ
            for uid, _, vec_str in rdr:
                if uid not in self.rw_dict:
                    continue
                if uid_to_val(uid, VAL_RATIO) ^ (self.split == "val"):
                    continue

                vec = parse_vec(vec_str, POST_DIM)
                if vec is None:
                    continue

                post_t = torch.from_numpy(vec)

                # 正例
                acc_pos = torch.from_numpy(self.rw_dict[uid].astype(np.float32))
                yield post_t, acc_pos, torch.tensor(1.0)

                # 負例
                for _ in range(NEG_RATIO):
                    neg_uid = uid
                    while neg_uid == uid:
                        neg_uid = random.choice(self.uids)
                    acc_neg = torch.from_numpy(self.rw_dict[neg_uid].astype(np.float32))
                    yield post_t, acc_neg, torch.tensor(0.0)


# ─────────── モデル ───────────
class PairClassifier(nn.Module):
    def __init__(self, post_dim, rw_dim, hidden_dim=512, dropout=DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim + rw_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, post_vec, acc_vec):
        x = torch.cat([post_vec, acc_vec], dim=1)
        return self.net(x).squeeze(1)


# ─────────── 学習ループ ───────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    train_ds = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, split="train")
    val_ds   = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = PairClassifier(POST_DIM, train_ds.rw_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 1
    best_val    = float("inf")
    patience    = 0

    # resume
    if args.resume and os.path.isfile(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        patience    = ckpt["patience"]
        print(f"[Resume] epoch={ckpt['epoch']}  best_val={best_val:.4f}  patience={patience}")

    for epoch in range(start_epoch, EPOCHS + 1):
        # ── Train ─────────────────────────
        model.train()
        total_train, n_train = 0.0, 0
        for post, acc, label in tqdm(
            train_loader,
            desc=f"Epoch {epoch} [train]",
            unit="batch",
            dynamic_ncols=True,
        ):
            post, acc, label = post.to(device), acc.to(device), label.to(device)
            logits = model(post, acc)
            loss   = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = post.size(0)
            total_train += loss.item() * bs
            n_train     += bs
        train_loss = total_train / n_train

        # ── Validation ─────────────────────
        model.eval()
        total_val, n_val = 0.0, 0
        with torch.no_grad():
            for post, acc, label in tqdm(
                val_loader,
                desc=f"Epoch {epoch} [val]  ",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            ):
                post, acc, label = post.to(device), acc.to(device), label.to(device)
                logits = model(post, acc)
                total_val += criterion(logits, label).item() * post.size(0)
                n_val     += post.size(0)
        val_loss = total_val / n_val

        print(f"Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # ── Checkpoint & Early-Stopping ────
        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_val":    best_val,
                    "patience":    patience,
                },
                CHECKPOINT_PATH,
            )
            print(f"  ✔ saved checkpoint → {CHECKPOINT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"[Done] best_val = {best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="load checkpoint and continue training",
    )
    args = parser.parse_args()
    train(args)