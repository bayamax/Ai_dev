#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_post_to_rw.py ― 投稿ベクトル１件ずつで学習するスクリプト
  • データ: /workspace/edit_agent/vast/aggregated_posting_vectors.csv
           /workspace/edit_agent/vast/account_vectors.npy
  • モデル: 投稿ベクトル (3072) → encoder_dim → RW ベクトル (元の次元)
  • 読み込み: IterableDataset でチャンク読み込み → DataLoader が自動でバッチ化／GPU転送
  • 損失: MSE
  • 早期停止＋モデル保存機能
"""
import os
import csv
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

# ── ハードコード済みパス ──────────────────────
VAST_DIR         = "/workspace/edit_agent/vast"
POSTS_CSV        = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY      = os.path.join(VAST_DIR, "account_vectors.npy")
DEFAULT_SAVE_PTH = os.path.join(VAST_DIR, "post_to_rw.pt")

# ── ユーティリティ関数 ──────────────────────
def parse_vec(s: str, dim: int) -> Optional[np.ndarray]:
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    # カンマもスペースも sep=' ' で一度に
    return np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ') \
           if np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ').size == dim \
           else None

# ── Dataset ──────────────────────────────────
class PostRWDataset(IterableDataset):
    def __init__(self,
                 posts_csv: str,
                 account_npy: str,
                 post_dim: int,
                 chunk_size: int = 1000):
        super().__init__()
        self.posts_csv  = posts_csv
        self.post_dim   = post_dim
        self.chunk_size = chunk_size
        # account_vectors.npy は {uid: np.ndarray(rw_dim)} の dict
        self.rw_dict: dict = np.load(account_npy, allow_pickle=True).item()
        # 予測先の次元
        any_vec = next(iter(self.rw_dict.values()))
        self.rw_dim = any_vec.shape[0]

    def __iter__(self):
        f = open(self.posts_csv, encoding='utf-8')
        rdr = csv.reader(f)
        next(rdr)  # ヘッダ行
        count = 0
        batch = []
        for uid, _, vec_str in rdr:
            if uid not in self.rw_dict:
                continue
            vec = parse_vec(vec_str, self.post_dim)
            if vec is None:
                continue
            post_t = torch.from_numpy(vec)
            rw_t   = torch.from_numpy(self.rw_dict[uid])
            yield post_t, rw_t
        f.close()

# ── モデル定義 ───────────────────────────────
class PostToRW(nn.Module):
    def __init__(self,
                 post_dim: int,
                 enc_dim: int,
                 rw_dim: int,
                 dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(post_dim, enc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, rw_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, post_dim)
        return self.net(x)

# ── 学習ループ ───────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # データセット & DataLoader
    ds = PostRWDataset(
        posts_csv=args.posts_csv,
        account_npy=args.account_npy,
        post_dim=args.post_dim,
        chunk_size=args.chunk_size
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # モデル／オプティマイザ／損失
    sample = next(iter(loader))
    rw_dim = sample[1].shape[1]
    model = PostToRW(
        post_dim=args.post_dim,
        enc_dim=args.enc_dim,
        rw_dim=rw_dim,
        dropout=args.dropout
    ).to(device)
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience  = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        n_samples  = 0

        for posts, targets in tqdm(loader, desc=f"Epoch {epoch} [train]"):
            posts, targets = posts.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(posts)
            loss  = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * posts.size(0)
            n_samples  += posts.size(0)

        train_loss = total_loss / n_samples
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}")

        # 早期停止は簡易的に train_loss で判断（val データないため）
        if train_loss < best_val - args.min_delta:
            best_val, patience = train_loss, 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✔ saved best → {args.save_path}")
        else:
            patience += 1
            print(f"  (no improve {patience}/{args.patience})")
            if patience >= args.patience:
                print("Early stopping.")
                break

    print(f"[Done] best_train_loss = {best_val:.4f}")

# ── CLI ───────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Per-post → RW 変換モデル学習")
    p.add_argument("--posts_csv",    default=POSTS_CSV)
    p.add_argument("--account_npy",  default=ACCOUNT_NPY)
    p.add_argument("--save_path",    default=DEFAULT_SAVE_PTH)
    p.add_argument("--post_dim",     type=int,   default=3072)
    p.add_argument("--enc_dim",      type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--batch_size",   type=int,   default=256)
    p.add_argument("--chunk_size",   type=int,   default=1000)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--min_delta",    type=float, default=1e-4)
    args = p.parse_args()
    train(args)