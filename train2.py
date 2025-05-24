#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pair_cosine_stream.py
投稿ベクトル ⇔ アカウントベクトルのペアを
正例(+1)／負例(–1)として CosineEmbeddingLoss で学習するストリーミング版。

・UID ハッシュで 90 % / 10 % の train / val split
・進捗バーなし（行単位のログのみ）
・--resume でチェックポイント継続学習
・負例 2 種
    ├─ランダム他人アカウントベクトル
    └─「正解アカウントベクトル＋微小ノイズ」を L2 正規化した擬似アカウント
"""

import os, csv, random, argparse, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

# ─────────── ファイルパス ───────────
VAST_DIR        = "/workspace/edit_agent/vast"
POSTS_CSV       = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY     = os.path.join(VAST_DIR, "account_vectors.npy")
CHECKPOINT_PATH = os.path.join(VAST_DIR, "pair_cosine_rw_stream.ckpt")

# ─────────── ハイパーパラメータ ───────────
POST_DIM      = 3072
BATCH_SIZE    = 128
EPOCHS        = 500
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
NEG_RATIO     = 5           # 正例1本につき負例5本（うち1本はノイズ負例）
VAL_RATIO     = 0.1
DROPOUT_RATE  = 0.1
PATIENCE      = 15
MIN_DELTA     = 1e-4
NOISE_STD     = 0.05        # ノイズ負例で加えるガウスノイズの標準偏差

# ─────────── ヘルパ関数 ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):            # 空
        return None
    if s.startswith('"[') and s.endswith(']"'): # 先頭と末尾の " を除く
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    vec = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return vec if vec.size == dim else None

def uid_to_val(uid: str, ratio: float) -> bool:
    h = int(hashlib.md5(uid.encode('utf-8')).hexdigest(), 16)
    return (h % 10_000) / 10_000.0 < ratio

def l2_normalize(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr)
    return arr / n if n != 0 else arr

# ─────────── データセット ───────────
class PairStreamDataset(IterableDataset):
    """
    (post_vec, acc_vec, target) をストリームで返す
      target = +1 … 同一UID
      target = -1 … ランダム他人 or ノイズ負例
    """
    def __init__(self, posts_csv, account_npy, split="train"):
        assert split in ("train", "val")
        self.posts_csv = posts_csv
        self.rw_dict   = np.load(account_npy, allow_pickle=True).item()
        self.uids      = list(self.rw_dict.keys())
        self.split     = split
        self.rw_dim    = next(iter(self.rw_dict.values())).shape[0]

    def __iter__(self):
        with open(self.posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_str in rdr:
                if uid not in self.rw_dict:
                    continue
                is_val = uid_to_val(uid, VAL_RATIO)
                if (self.split == "train" and is_val) or (self.split == "val" and not is_val):
                    continue

                post_vec = parse_vec(vec_str, POST_DIM)
                if post_vec is None:
                    continue
                post_t = torch.from_numpy(post_vec)

                # 正例 (+1)
                acc_pos = torch.from_numpy(self.rw_dict[uid])
                yield post_t, acc_pos, torch.tensor(1, dtype=torch.float32)

                # 負例 (-1)
                for i in range(NEG_RATIO):
                    if i == 0:
                        # ----- ノイズ負例 -----
                        base = self.rw_dict[uid]
                        noise = np.random.normal(0, NOISE_STD, size=base.shape).astype(np.float32)
                        acc_neg_np = l2_normalize(base + noise)
                    else:
                        # ----- ランダム他人 -----
                        neg_uid = uid
                        while neg_uid == uid:
                            neg_uid = random.choice(self.uids)
                        acc_neg_np = self.rw_dict[neg_uid]
                    acc_neg = torch.from_numpy(acc_neg_np)
                    yield post_t, acc_neg, torch.tensor(-1, dtype=torch.float32)

# ─────────── モデル ───────────
class CosineMapper(nn.Module):
    """
    post_vec (3072D) → 共有埋め込み空間 (rw_dim) へ線形射影
    アカウントベクトルはそのまま使う
    """
    def __init__(self, post_dim, rw_dim, hidden_dim=512, dropout=DROPOUT_RATE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rw_dim),
        )

    def forward(self, post_vec):
        return self.net(post_vec)

# ─────────── 学習 ───────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    train_ds = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, split="train")
    val_ds   = PairStreamDataset(POSTS_CSV, ACCOUNT_NPY, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = CosineMapper(POST_DIM, train_ds.rw_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CosineEmbeddingLoss()

    start_epoch = 1
    best_val    = float("inf")
    patience    = 0

    if args.resume and os.path.isfile(CHECKPOINT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
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
        for post, acc, target in train_loader:
            post, acc, target = post.to(device), acc.to(device), target.to(device)
            post_emb = model(post)
            loss     = criterion(
                nn.functional.normalize(post_emb),
                nn.functional.normalize(acc),
                target
            )
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
            for post, acc, target in val_loader:
                post, acc, target = post.to(device), acc.to(device), target.to(device)
                post_emb = model(post)
                loss     = criterion(
                    nn.functional.normalize(post_emb),
                    nn.functional.normalize(acc),
                    target
                )
                total_val += loss.item() * post.size(0)
                n_val     += post.size(0)
        val_loss = total_val / n_val

        print(f"Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # ── Checkpoint & Early-Stopping ────
        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val":    best_val,
                "patience":    patience,
            }, CHECKPOINT_PATH)
            print(f"  ✔ saved checkpoint → {CHECKPOINT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"[Done] best_val = {best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="load checkpoint and continue training")
    args = parser.parse_args()
    train(args)