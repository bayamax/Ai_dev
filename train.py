#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― Set-Transformer による投稿セット→ランダムウォーク埋め込み回帰

概要:
  ・入力データ（ハードコード、CLIで上書き可）：
      - /workspace/edit_agent/vast/aggregated_posting_vectors.csv
      - /workspace/edit_agent/vast/account_vectors.npy
  ・出力モデル：
      - /workspace/edit_agent/vast/set_transformer_rw_predictor.pt
  ・モデル：
      - 投稿埋め込みを TransformerEncoder で集約し平均プーリング
      - プーリング出力を線形デコーダでランダムウォーク埋め込み次元に射影
  ・損失：
      - コサイン損失 1 - cos(pred, target) のバッチ平均
  ・早期停止・学習率スケジューラ対応
"""

import os
import sys
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

# ─────────── ハードコード済みパス ───────────
VAST_DIR              = "/workspace/edit_agent/vast"
DEFAULT_POSTS_CSV     = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
DEFAULT_ACCOUNT_NPY   = os.path.join(VAST_DIR, "account_vectors.npy")
DEFAULT_MODEL_SAVE    = os.path.join(VAST_DIR, "set_transformer_rw_predictor.pt")

# ─────────── ハイパラ ───────────
POST_DIM       = 3072
ENC_DIM        = 512
N_HEADS        = 4
N_LAYERS       = 2
DROPOUT        = 0.1

BATCH_SIZE     = 64
EPOCHS         = 100
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
VAL_SPLIT      = 0.1
PATIENCE       = 10
MIN_DELTA      = 1e-4
MAX_POSTS      = 50

# ─────────── ユーティリティ: 文字列→ベクトル ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

# ─────────── Dataset ───────────
class RWDataset(Dataset):
    def __init__(self, posts_csv, account_npy, max_posts):
        # ランダムウォーク埋め込み読み込み
        rw_dict = np.load(account_npy, allow_pickle=True).item()
        # 投稿ベクトル読み込み
        posts_map = defaultdict(list)
        with open(posts_csv, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for uid, _, vec_str in tqdm(reader, desc="Loading posts"):
                if uid not in rw_dict:
                    continue
                v = parse_vec(vec_str, POST_DIM)
                if v is not None:
                    posts_map[uid].append(v)
        # サンプル作成：投稿ゼロは除外
        self.samples = []
        for uid, vecs in posts_map.items():
            if not vecs:
                continue
            vecs = vecs[-max_posts:]
            post_tensor = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)
            target = torch.tensor(rw_dict[uid], dtype=torch.float32)
            self.samples.append((post_tensor, target))
        if not self.samples:
            sys.exit("ERROR: No valid samples (all accounts have zero posts?)")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    posts_list, targets = zip(*batch)
    lengths = torch.tensor([p.size(0) for p in posts_list])
    max_len = lengths.max().item()
    padded = torch.nn.utils.rnn.pad_sequence(posts_list, batch_first=True)
    pad_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets)
    return padded, pad_mask, targets

# ─────────── Model ───────────
class SetToRW(nn.Module):
    def __init__(self, post_dim, enc_dim, rw_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(post_dim, enc_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim, nhead=n_heads,
            dim_feedforward=enc_dim*4, dropout=dropout,
            batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.Linear(enc_dim, rw_dim)
    def forward(self, posts, pad_mask):
        # posts: (B, S, D_in), pad_mask: (B, S)
        x = self.proj(posts)               # (B, S, D_enc)
        x = x.permute(1,0,2)               # (S, B, D_enc)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # (S,B,D_enc)
        x = x.permute(1,0,2)               # (B, S, D_enc)
        valid = (~pad_mask).unsqueeze(-1).float() # (B,S,1)
        summed = (x * valid).sum(dim=1)    # (B, D_enc)
        lengths = valid.sum(dim=1).clamp(min=1.0)  # (B,1)
        pooled = summed / lengths          # (B, D_enc)
        return self.decoder(pooled)        # (B, rw_dim)

# ─────────── Loss: cosine ───────────
def cosine_loss(pred, target):
    cos = F.cosine_similarity(pred, target, dim=1)
    return (1.0 - cos).mean()

# ─────────── Training ───────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = RWDataset(args.posts_csv, args.account_npy, args.max_posts)
    n_val = int(len(ds) * args.val_split)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SetToRW(
        post_dim=args.post_dim,
        enc_dim=args.enc_dim,
        rw_dim=ds.samples[0][1].size(0),
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val, patience = float("inf"), 0

    for ep in range(1, args.epochs+1):
        # train
        model.train()
        total_train = 0.0
        for posts, mask, targets in tqdm(tr_loader, desc=f"Epoch {ep} [train]"):
            posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(posts, mask)
            loss  = cosine_loss(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train += loss.item() * posts.size(0)
        train_loss = total_train / len(tr_loader.dataset)

        # val
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_loader, desc=f"Epoch {ep} [val]"):
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
                preds = model(posts, mask)
                total_val += cosine_loss(preds, targets).item() * posts.size(0)
        val_loss = total_val / len(va_loader.dataset)

        print(f"Epoch {ep}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - args.min_delta:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"  ✔ saved best → {args.save_path}")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping."); break

        scheduler.step()

    print(f"[Done] best_val_loss = {best_val:.4f}")

# ─────────── CLI ───────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--posts_csv",     default=DEFAULT_POSTS_CSV)
    p.add_argument("--account_npy",   default=DEFAULT_ACCOUNT_NPY)
    p.add_argument("--save_path",     default=DEFAULT_MODEL_SAVE)
    p.add_argument("--post_dim",      type=int,   default=POST_DIM)
    p.add_argument("--enc_dim",       type=int,   default=ENC_DIM)
    p.add_argument("--n_heads",       type=int,   default=N_HEADS)
    p.add_argument("--n_layers",      type=int,   default=N_LAYERS)
    p.add_argument("--dropout",       type=float, default=DROPOUT)
    p.add_argument("--batch_size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--epochs",        type=int,   default=EPOCHS)
    p.add_argument("--lr",            type=float, default=LR)
    p.add_argument("--weight_decay",  type=float, default=WEIGHT_DECAY)
    p.add_argument("--val_split",     type=float, default=VAL_SPLIT)
    p.add_argument("--patience",      type=int,   default=PATIENCE)
    p.add_argument("--min_delta",     type=float, default=MIN_DELTA)
    p.add_argument("--max_posts",     type=int,   default=MAX_POSTS)
    args = p.parse_args()
    train(args)