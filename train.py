#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― 投稿セットからランダムウォーク埋め込みを予測する Set‐Transformer 学習スクリプト
  • CSV はチャンク読み込み (chunksize=1000 デフォルト)
  • 投稿ベクトルのパースとテンソル化は GPU 上で実行
  • 早期停止・モデル保存機能付き
"""

import os
import sys
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ───────── 既定ディレクトリ ─────────
VAST_DIR            = "/workspace/edit_agent/vast"
DEFAULT_POSTS_CSV   = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
DEFAULT_ACCOUNT_NPY = os.path.join(VAST_DIR, "account_vectors.npy")
DEFAULT_SAVE_PATH   = os.path.join(VAST_DIR, "set_transformer_rw.pt")

# ───────── ハイパラ ─────────
POST_DIM    = 3072
MAX_POSTS   = 50
CHUNK_SIZE  = 1000    # CSV 読み込みチャンクサイズ（デフォルト）
BATCH_SIZE  = 64
EPOCHS      = 100
LR          = 1e-4
WEIGHT_DECAY= 1e-5
VAL_SPLIT   = 0.1
PATIENCE    = 10
MIN_DELTA   = 1e-4

ENC_DIM     = 512
N_HEADS     = 4
N_LAYERS    = 2
DROPOUT     = 0.1

# ───────── Dataset ─────────
class RWDataset(Dataset):
    def __init__(self, posts_csv, acc_npy, max_posts, chunk_size, device):
        # ランダムウォーク埋め込み読み込み
        try:
            rw_dict = np.load(acc_npy, allow_pickle=True).item()
        except Exception as e:
            sys.exit(f"ERROR: account_vectors.npy が読み込めません: {e}")
        self.rw_dim = next(iter(rw_dict.values())).shape[0]
        self.device = device

        # 投稿ベクトルをチャンクで読み込み
        user_posts = defaultdict(list)
        for chunk in pd.read_csv(posts_csv, header=0, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                uid = row[0]
                if uid not in rw_dict:
                    continue
                s = row[2].strip()
                if s.startswith('"[') and s.endswith(']"'):
                    s = s[1:-1]
                l, r = s.find('['), s.rfind(']')
                if 0 <= l < r:
                    s = s[l+1:r]
                arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
                if arr.size != POST_DIM:
                    continue
                # GPU 上に直接載せる
                tensor = torch.tensor(arr, dtype=torch.float32, device=self.device)
                user_posts[uid].append(tensor)

        # サンプル組み立て
        self.samples = []
        for uid, vecs in user_posts.items():
            if not vecs:
                continue
            if max_posts > 0 and len(vecs) > max_posts:
                vecs = vecs[-max_posts:]
            posts_tensor = torch.stack(vecs, dim=0)  # (L, POST_DIM)
            target = torch.tensor(rw_dict[uid], dtype=torch.float32, device=self.device)
            self.samples.append((posts_tensor, target))

        if not self.samples:
            sys.exit("ERROR: RWDataset に読み込まれたデータがありません")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ───────── collate_fn ─────────
def collate_fn(batch):
    posts_list, targets = zip(*batch)
    # 各バッチ内シーケンス長
    lengths = torch.tensor([p.size(0) for p in posts_list], device=posts_list[0].device)
    max_len = int(lengths.max().item())
    # pad
    padded = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True, padding_value=0.0)
    # mask: True=padding
    padding_mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets, dim=0)
    return padded, padding_mask, targets

# ───────── モデル ─────────
class SetToRW(nn.Module):
    def __init__(self, post_dim, enc_dim, rw_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(post_dim, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=n_heads,
            dim_feedforward=enc_dim * 4,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = nn.Linear(enc_dim, rw_dim)

    def forward(self, posts, pad_mask):
        # posts: (B, S, POST_DIM), pad_mask: (B, S)
        x = self.proj(posts)                     # (B, S, ENC_DIM)
        x = x.permute(1, 0, 2)                   # (S, B, ENC_DIM)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (S, B, ENC_DIM)
        x = x.permute(1, 0, 2)                   # (B, S, ENC_DIM)
        valid = (~pad_mask).unsqueeze(-1).float()# (B, S, 1)
        summed = (x * valid).sum(dim=1)          # (B, ENC_DIM)
        lengths = valid.sum(dim=1).clamp(min=1.0)# (B, 1)
        pooled = summed / lengths                # (B, ENC_DIM)
        return self.decoder(pooled)              # (B, rw_dim)

# ───────── 学習ルーチン ─────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # データセット
    ds = RWDataset(
        posts_csv=args.posts_csv,
        acc_npy=args.account_npy,
        max_posts=args.max_posts,
        chunk_size=args.chunk_size,
        device=device
    )
    n_val = int(len(ds) * args.val_split)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # モデル
    model = SetToRW(
        post_dim=POST_DIM,
        enc_dim=args.enc_dim,
        rw_dim=ds.rw_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        # — train —
        model.train()
        sum_loss = 0.0
        for posts, mask, targets in tqdm(tr_loader, desc=f"Epoch {epoch} [train]"):
            posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(posts, mask)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += loss.item() * posts.size(0)
        train_loss = sum_loss / len(tr_loader.dataset)

        # — val —
        model.eval()
        sum_val = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_loader, desc=f"Epoch {epoch} [val]"):
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
                sum_val += criterion(model(posts, mask), targets).item() * posts.size(0)
        val_loss = sum_val / len(va_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        # early stopping
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            patience = 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✔ saved best → {args.save_path}")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping.")
                break

    print(f"[Done] best_val_loss = {best_val:.4f}")

# ───────── CLI ─────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SetTransformer → RW embedding prediction")
    p.add_argument("--posts_csv",    default=DEFAULT_POSTS_CSV)
    p.add_argument("--account_npy",  default=DEFAULT_ACCOUNT_NPY)
    p.add_argument("--save_path",    default=DEFAULT_SAVE_PATH)
    p.add_argument("--chunk_size",   type=int,   default=CHUNK_SIZE,
                   help="CSV読み込みのチャンクサイズ (default: 1000)")
    p.add_argument("--max_posts",    type=int,   default=MAX_POSTS)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--val_split",    type=float, default=VAL_SPLIT)
    p.add_argument("--patience",     type=int,   default=PATIENCE)
    p.add_argument("--min_delta",    type=float, default=MIN_DELTA)

    p.add_argument("--enc_dim",      type=int,   default=ENC_DIM)
    p.add_argument("--n_heads",      type=int,   default=N_HEADS)
    p.add_argument("--n_layers",     type=int,   default=N_LAYERS)
    p.add_argument("--dropout",      type=float, default=DROPOUT)

    args = p.parse_args()
    train(args)