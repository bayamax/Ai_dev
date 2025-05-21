#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― 投稿セットからランダムウォーク埋め込みを予測する Set-Transformer 学習スクリプト
  • CSV をチャンク単位で読み込み（--chunk_size）
  • 読み込んだテンソルは即座に GPU へ転送
  • 早期停止・モデル保存機能あり
"""

import os
import sys
import argparse
import re
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

# ───────── データセット ─────────
class RWDataset(Dataset):
    def __init__(self, posts_csv, acc_npy, max_posts, chunk_size, device):
        # ランダムウォーク埋め込み読み込み
        rw_dict = np.load(acc_npy, allow_pickle=True).item()
        self.rw_dim = next(iter(rw_dict.values())).shape[0]
        self.device = device

        # 投稿ベクトルをチャンク読み込み
        user_posts = defaultdict(list)
        for chunk in pd.read_csv(posts_csv, usecols=[0,2],
                                 header=0, chunksize=chunk_size):
            for uid, vec_str in zip(chunk.iloc[:,0], chunk.iloc[:,1]):
                if uid not in rw_dict: 
                    continue
                s = str(vec_str).strip()
                if s.startswith('"[') and s.endswith(']"'):
                    s = s[1:-1]
                l, r = s.find('['), s.rfind(']')
                if 0 <= l < r:
                    s = s[l+1:r]
                arr = np.fromstring(s.replace(',', ' '),
                                   dtype=np.float32, sep=' ')
                if arr.size == 0:
                    continue
                user_posts[uid].append(arr)

        # 各ユーザの最新 max_posts 件をテンソル化して GPU へ
        self.samples = []
        for uid, vecs in user_posts.items():
            if not vecs:
                continue
            vecs = vecs[-max_posts:]
            posts_tensor = torch.tensor(
                np.stack(vecs, axis=0),
                dtype=torch.float32, device=self.device
            )
            target = torch.tensor(
                rw_dict[uid], dtype=torch.float32, device=self.device
            )
            self.samples.append((posts_tensor, target))

        if not self.samples:
            sys.exit("ERROR: No data loaded for RWDataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    posts_list, targets = zip(*batch)
    device = posts_list[0].device

    # 可変長シーケンスをパディング
    lengths = torch.tensor(
        [p.size(0) for p in posts_list],
        dtype=torch.long, device=device
    )
    max_len = lengths.max().item()
    padded = torch.nn.utils.rnn.pad_sequence(
        posts_list, batch_first=True, padding_value=0.0
    ).to(device)
    padding_mask = torch.arange(
        max_len, device=device
    ).unsqueeze(0) >= lengths.unsqueeze(1)

    targets = torch.stack(targets)
    return padded, padding_mask, targets


# ───────── モデル ─────────
class SetToRW(nn.Module):
    def __init__(self, post_dim, enc_dim, rw_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(post_dim, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim, nhead=n_heads,
            dim_feedforward=enc_dim*4, dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = nn.Linear(enc_dim, rw_dim)

    def forward(self, posts, pad_mask):
        # posts: (B, S, D_in), pad_mask: (B, S)
        x = self.proj(posts)                   # (B, S, D_enc)
        x = x.permute(1, 0, 2)                # (S, B, D_enc)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (S, B, D_enc)
        x = x.permute(1, 0, 2)                # (B, S, D_enc)

        valid = (~pad_mask).unsqueeze(-1).float()  # (B, S, 1)
        summed = (x * valid).sum(dim=1)           # (B, D_enc)
        lengths = valid.sum(dim=1).clamp(min=1.0)  # (B, 1)
        pooled = summed / lengths                 # (B, D_enc)

        return self.decoder(pooled)               # (B, rw_dim)


# ───────── 学習ルーチン ─────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # データセット読み込み
    ds = RWDataset(
        args.posts_csv,
        args.account_npy,
        args.max_posts,
        args.chunk_size,
        device
    )
    n_val = int(len(ds) * args.val_split)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    va_loader = DataLoader(
        va_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    model = SetToRW(
        args.post_dim, args.enc_dim, ds.rw_dim,
        args.n_heads, args.n_layers, args.dropout
    ).to(device)
    crit      = nn.MSELoss()
    optim_    = optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay
    )

    best_val, patience = float("inf"), 0

    for ep in range(1, args.epochs+1):
        # —— train ——
        model.train()
        sum_loss = 0.0
        for posts, mask, targets in tqdm(tr_loader, desc=f"Epoch {ep} [train]"):
            optim_.zero_grad()
            preds = model(posts, mask)
            loss  = crit(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_.step()
            sum_loss += loss.item() * posts.size(0)
        train_loss = sum_loss / len(tr_loader.dataset)

        # —— val ——
        model.eval()
        sum_val = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_loader, desc=f"Epoch {ep} [val]"):
                sum_val += crit(
                    model(posts, mask), targets
                ).item() * posts.size(0)
        val_loss = sum_val / len(va_loader.dataset)

        print(f"Epoch {ep}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - args.min_delta:
            best_val, patience = val_loss, 0
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
    p = argparse.ArgumentParser()
    p.add_argument("--posts_csv",    default=DEFAULT_POSTS_CSV)
    p.add_argument("--account_npy",  default=DEFAULT_ACCOUNT_NPY)
    p.add_argument("--save_path",    default=DEFAULT_SAVE_PATH)
    p.add_argument("--max_posts",    type=int,   default=50)
    p.add_argument("--chunk_size",   type=int,   default=100000)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--val_split",    type=float, default=0.1)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--min_delta",    type=float, default=1e-4)
    p.add_argument("--post_dim",     type=int,   default=3072)
    p.add_argument("--enc_dim",      type=int,   default=512)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--n_layers",     type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.1)
    args = p.parse_args()

    train(args)