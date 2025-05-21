#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― 投稿セットからランダムウォーク埋め込みを予測する Set-Transformer 学習スクリプト

• CSV は streaming 読み込み
• 損失：コサイン類似度ベース（1 − cosine_similarity）
• 早期停止・モデル保存
"""

import os
import sys
import re
import csv
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ───────── 既定ディレクトリ ─────────
VAST_DIR            = "/workspace/edit_agent/vast"
POSTS_CSV           = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY         = os.path.join(VAST_DIR, "account_vectors.npy")
DEFAULT_SAVE_PATH   = os.path.join(VAST_DIR, "set_transformer_rw.pt")

# ───────── 定数 ─────────
MAX_POSTS     = 50
BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.1
PATIENCE      = 10
MIN_DELTA     = 1e-4

# ───────── Dataset ─────────
class RWDataset(Dataset):
    def __init__(self, posts_csv, acc_npy, max_posts=MAX_POSTS):
        # 1) ランダムウォーク埋め込み読み込み
        rw_dict = np.load(acc_npy, allow_pickle=True).item()
        self.rw_dim = next(iter(rw_dict.values())).shape[0]

        # 2) 投稿ベクトルを csv.reader で streaming
        user_posts = defaultdict(list)
        with open(posts_csv, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in tqdm(reader, desc="Loading posts"):
                if len(row) < 3:
                    continue
                uid, vec_str = row[0], row[2]
                if uid not in rw_dict:
                    continue
                s = vec_str.strip()
                if s.startswith('"[') and s.endswith(']"'):
                    s = s[1:-1]
                l, r = s.find('['), s.rfind(']')
                if 0 <= l < r:
                    s = s[l+1:r]
                arr = np.fromstring(re.sub(r'[\s,]+',' ', s), dtype=np.float32, sep=' ')
                if arr.size > 0:
                    user_posts[uid].append(arr)
        # 3) Tensor化 & 投稿数トランケート
        samples = []
        for uid, vecs in user_posts.items():
            if not vecs: continue
            vecs = vecs[-max_posts:]
            posts_tensor = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)
            target = torch.tensor(rw_dict[uid], dtype=torch.float32)
            samples.append((posts_tensor, target))
        if not samples:
            sys.exit("ERROR: No samples loaded for RWDataset")
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    posts_list, targets = zip(*batch)
    lengths = torch.tensor([p.size(0) for p in posts_list], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True, padding_value=0.0)
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets)
    return padded, mask, targets

# ───────── モデル ─────────
class SetToRW(nn.Module):
    def __init__(self, post_dim, enc_dim, rw_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(post_dim, enc_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=n_heads,
            dim_feedforward=enc_dim*4,
            dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.decoder = nn.Linear(enc_dim, rw_dim)

    def forward(self, posts, pad_mask):
        x = self.proj(posts)                           # (B, S, D_enc)
        x = x.permute(1, 0, 2)                         # (S, B, D_enc)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x.permute(1, 0, 2)                         # (B, S, D_enc)
        valid = (~pad_mask).unsqueeze(-1).float()
        summed = (x * valid).sum(dim=1)
        lengths = valid.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        return self.decoder(pooled)                    # (B, rw_dim)

# ───────── 学習ルーチン ─────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    ds = RWDataset(args.posts_csv, args.acc_npy, args.max_posts)
    n_val = int(len(ds) * args.val_split)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SetToRW(
        post_dim=ds.samples[0][0].size(1),
        enc_dim=args.enc_dim,
        rw_dim=ds.rw_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    patience = 0

    for epoch in range(1, args.epochs+1):
        # train
        model.train()
        sum_loss = 0.0
        for posts, mask, targets in tqdm(tr_loader, desc=f"Epoch {epoch} [train]"):
            posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(posts, mask)
            cos = F.cosine_similarity(preds, targets, dim=1)       # (B,)
            loss = (1 - cos).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += loss.item() * posts.size(0)
        train_loss = sum_loss / len(tr_loader.dataset)

        # val
        model.eval()
        sum_val = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_loader, desc=f"Epoch {epoch} [val]"):
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
                preds = model(posts, mask)
                cos = F.cosine_similarity(preds, targets, dim=1)
                sum_val += ((1 - cos).sum().item())
        val_loss = sum_val / len(va_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - args.min_delta:
            best_val = val_loss; patience = 0
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
if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--posts_csv",    default=POSTS_CSV)
    p.add_argument("--acc_npy",      default=ACCOUNT_NPY)
    p.add_argument("--save_path",    default=DEFAULT_SAVE_PATH)
    p.add_argument("--max_posts",    type=int,   default=MAX_POSTS)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",       type=int,   default=EPOCHS)
    p.add_argument("--lr",           type=float, default=LR)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--val_split",    type=float, default=VAL_SPLIT)
    p.add_argument("--patience",     type=int,   default=PATIENCE)
    p.add_argument("--min_delta",    type=float, default=MIN_DELTA)
    p.add_argument("--enc_dim",      type=int,   default=512)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--n_layers",     type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.1)
    args = p.parse_args()
    train(args)