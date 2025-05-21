#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — 投稿ベクトルセットからランダムウォーク埋め込みを予測するモデルの学習スクリプト

タスク:
  • 入力: 各ユーザーの投稿ベクトルセット (最大 max_posts 件)
  • 出力: ランダムウォーク埋め込みベクトル (次元 out_dim)
損失: CosineEmbeddingLoss（全例正例ラベル +1）
モデル: 
  1) Linear(post_dim→hid)
  2) TransformerEncoderLayer × n_layers
  3) マスク付き平均プーリング
  4) MLP(hid→hid/2→out_dim)
"""

import os
import sys
import csv
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class RWDataset(Dataset):
    """投稿ベクトルCSVと account_vectors.npy からデータセットを作成"""
    def __init__(self, posts_csv: str, acc_npy: str, max_posts: int = 50):
        # ランダムウォークベクトル読み込み
        self.rw_dict = np.load(acc_npy, allow_pickle=True).item()  # {user_id: np.ndarray(out_dim)}
        self.max_posts = max_posts

        # 投稿ベクトル群を読み込んでユーザーごとに集約
        tmp = defaultdict(list)
        with open(posts_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                user_id = row[0]
                if user_id not in self.rw_dict:
                    continue
                emb_str = row[2]  # "[v1, v2, ...]"
                # 内側の数値だけ抜き出す
                values = np.fromstring(emb_str.strip('[]'), sep=' ')
                if values.size == 0:
                    continue
                tmp[user_id].append(values)
        
        # 最終的なリストとインデックスを作成
        self.uids = []
        self.posts = []
        for uid, vecs in tmp.items():
            if len(vecs) == 0:
                continue
            truncated = vecs[-self.max_posts:]
            self.uids.append(uid)
            self.posts.append(torch.tensor(np.stack(truncated), dtype=torch.float32))
        if not self.uids:
            raise RuntimeError("No users with valid posts found")

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        posts = self.posts[idx]               # Tensor (S, post_dim)
        uid = self.uids[idx]
        rw = torch.tensor(self.rw_dict[uid], dtype=torch.float32)  # Tensor (out_dim,)
        # パディング不要位置を False、パディング位置を True にするマスク
        mask = torch.zeros(posts.size(0), dtype=torch.bool)
        return posts, mask, rw


def collate_fn(batch):
    """可変長シーケンスをバッチ処理用にパディング"""
    posts_list, masks, rws = zip(*batch)
    lengths = [p.size(0) for p in posts_list]
    max_len = max(lengths)
    # pad_sequence は長さが違うリストを (B, S, D) にパディング
    padded = nn.utils.rnn.pad_sequence(posts_list, batch_first=True, padding_value=0.0)
    # padding_mask: True=padding, False=data
    pad_masks = torch.arange(max_len).unsqueeze(0) >= torch.tensor(lengths).unsqueeze(1)
    return padded, pad_masks, torch.stack(rws)


class Post2RW(nn.Module):
    """投稿セット→ランダムウォーク埋め込みへの変換モデル"""
    def __init__(self,
                 post_dim: int = 3072,
                 hid: int = 512,
                 out_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(post_dim, hid)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid, nhead=n_heads,
            dim_feedforward=hid*4, dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hid, hid//2),
            nn.ReLU(inplace=True),
            nn.Linear(hid//2, out_dim)
        )

    def forward(self, posts: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        posts: (B, S, post_dim)
        pad_mask: (B, S)  True=padding
        return: (B, out_dim)
        """
        h = self.proj(posts)  # (B,S,hid)
        h = self.encoder(h, src_key_padding_mask=pad_mask)  # (B,S,hid)
        # マスク付き平均プーリング
        valid = (~pad_mask).unsqueeze(-1).float()            # (B,S,1)
        pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1e-6)  # (B,hid)
        return self.head(pooled)  # (B,out_dim)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # データセット準備
    ds = RWDataset(args.posts_csv, args.acc_npy, args.max_posts)
    n_val = int(len(ds) * args.val_split)
    n_tr = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=2)

    # モデル／損失関数／最適化
    model = Post2RW(args.post_dim, args.hid, args.out_dim,
                    args.n_heads, args.n_layers, args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CosineEmbeddingLoss(margin=0.0)

    best_val = float('inf')
    for ep in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        total_loss = 0.0
        for posts, mask, rw in tqdm(tr_loader, desc=f"Ep{ep}/train"):
            posts, mask, rw = posts.to(device), mask.to(device), rw.to(device)
            optimizer.zero_grad()
            pred = model(posts, mask)          # (B,out_dim)
            labels = torch.ones(pred.size(0), device=device)
            loss = criterion(pred, rw, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * posts.size(0)
        train_loss = total_loss / len(tr_loader.dataset)

        # --- validation ---
        model.eval()
        total_vloss = 0.0
        with torch.no_grad():
            for posts, mask, rw in va_loader:
                posts, mask, rw = posts.to(device), mask.to(device), rw.to(device)
                pred = model(posts, mask)
                labels = torch.ones(pred.size(0), device=device)
                total_vloss += criterion(pred, rw, labels).item() * posts.size(0)
        val_loss = total_vloss / len(va_loader.dataset)

        print(f"Ep {ep}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        scheduler.step()

        # --- early stopping & save ---
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save(model.state_dict(), args.ckpt)
            print(f"  ✔ saved best → {args.ckpt}")
        elif ep - args.patience >= 0:
            print("Early stopping.")
            break

    print(f"[Done] best_val = {best_val:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Post→RW embedding model")
    p.add_argument("--posts_csv",    required=True,
                   help="aggregated_posting_vectors.csv のパス")
    p.add_argument("--acc_npy",      required=True,
                   help="account_vectors.npy のパス")
    p.add_argument("--ckpt",         default="best_rw.pt",
                   help="モデルチェックポイント保存先")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--wd",           type=float, default=1e-5)
    p.add_argument("--val_split",    type=float, default=0.1)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--min_delta",    type=float, default=1e-4)
    p.add_argument("--max_posts",    type=int,   default=50)
    p.add_argument("--post_dim",     type=int,   default=3072)
    p.add_argument("--hid",          type=int,   default=512)
    p.add_argument("--out_dim",      type=int,   default=128)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--n_layers",     type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.1)
    args = p.parse_args()
    train(args)