#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― 投稿ベクトルからランダムウォーク埋め込みを
「3072次元→チャンク化→Transformer → プーリング→予測」する学習スクリプト

★ 特徴次元を chunk_size ごとにチャンク化し、トークンとして扱うことで
  Attention の O(L^2) コストを抑制します。
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ─────────── ハードコード済みパス ───────────
VAST_DIR          = "/workspace/edit_agent/vast"
POSTS_CSV         = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY       = os.path.join(VAST_DIR, "account_vectors.npy")
DEFAULT_SAVE_PATH = os.path.join(VAST_DIR, "set_transformer_rw.pt")

# ─────────── ハイパラ ───────────
POST_DIM      = 3072
CHUNK_SIZE    = 32            # 次元をこれだけずつ切ってトークン化
D_MODEL       = 256           # 各チャンクを射影するモデル次元
ENC_LAYERS    = 4
N_HEADS       = 4
DROPOUT       = 0.1

MAX_POSTS     = 50
BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.1
PATIENCE      = 10
MIN_DELTA     = 1e-4

# ─────────── ユーティリティ ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'):
        return None
    if s.startswith('"[') and s.endswith(']"'):
        s = s[1:-1]
    l, r = s.find('['), s.rfind(']')
    if 0 <= l < r:
        s = s[l+1:r]
    s = s.replace(',', ' ')
    arr = np.fromstring(s, dtype=np.float32, sep=' ')
    return arr if arr.size == dim else None

class RWDataset(Dataset):
    """投稿シーケンス → ランダムウォーク埋め込み予測用データセット"""
    def __init__(self, posts_csv, acc_npy, max_posts=MAX_POSTS):
        # 1) ランダムウォーク埋め込み読み込み
        rw_dict = np.load(acc_npy, allow_pickle=True).item()
        # 2) 投稿ベクトル読み込み
        user_posts = defaultdict(list)
        with open(posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_str in tqdm(rdr, desc="Loading posts"):
                if uid not in rw_dict:
                    continue
                vec = parse_vec(vec_str, POST_DIM)
                if vec is not None:
                    user_posts[uid].append(vec)
        # 3) 投稿ゼロアカウントは除外
        self.samples = []
        for uid, vecs in user_posts.items():
            if not vecs:
                continue
            vecs = vecs[-max_posts:]
            posts_t = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)  # (S, D)
            target  = torch.tensor(rw_dict[uid], dtype=torch.float32)            # (rw_dim,)
            self.samples.append((posts_t, target))
        if not self.samples:
            sys.exit("ERROR: No users with posts found.")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    posts, targets = zip(*batch)
    lengths = torch.tensor([p.size(0) for p in posts], dtype=torch.long)
    max_len = lengths.max().item()
    padded = torch.nn.utils.rnn.pad_sequence(posts, batch_first=True, padding_value=0.0)  # (B,S,D)
    # True=padding
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)                    # (B,S)
    targets = torch.stack(targets)                                                       # (B,rw_dim)
    return padded, mask, targets

# ─────────── モデル ───────────
class ChunkedTransformerRW(nn.Module):
    def __init__(self, post_dim, chunk_size, d_model,
                 n_heads, enc_layers, dropout, rw_dim):
        super().__init__()
        assert post_dim % chunk_size == 0, "post_dim must be divisible by chunk_size"
        self.chunk_size   = chunk_size
        self.num_chunks   = post_dim // chunk_size
        self.patch_proj   = nn.Linear(chunk_size, d_model)
        encoder_layer      = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=False
        )
        self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=enc_layers)
        self.decoder      = nn.Linear(d_model, rw_dim)

    def forward(self, posts, pad_mask):
        """
        posts: (B, S, D)
        pad_mask: (B, S)  True = padding
        """
        B, S, D = posts.size()
        # 1) 各投稿ベクトルをチャンク化 → トークン列 (B, S*num_chunks, chunk_size)
        x = posts.view(B, S, self.num_chunks, self.chunk_size)
        x = x.flatten(1, 2)           # (B, S*num_chunks, chunk_size)
        # 2) 各トークンを d_model に射影
        x = self.patch_proj(x)        # (B, T, d_model), T=S*num_chunks
        # 3) トークン用マスク生成
        pm = pad_mask.unsqueeze(-1).expand(B, S, self.num_chunks)
        token_mask = pm.flatten(1, 2) # (B, T)
        # 4) Transformer エンコード
        x = x.permute(1, 0, 2)        # (T, B, d_model)
        x = self.transformer(x, src_key_padding_mask=token_mask)  # (T, B, d_model)
        x = x.permute(1, 0, 2)        # (B, T, d_model)
        # 5) マスク平均プーリング
        valid = (~token_mask).unsqueeze(-1).float()  # (B, T, 1)
        sum_vec = (x * valid).sum(dim=1)             # (B, d_model)
        lengths = valid.sum(dim=1).clamp(min=1.0)    # (B, 1)
        pooled  = sum_vec / lengths                  # (B, d_model)
        # 6) デコーダー→ランダムウォーク埋め込み予測
        return self.decoder(pooled)                  # (B, rw_dim)

# ─────────── 学習ループ ───────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = RWDataset(POSTS_CSV, ACCOUNT_NPY, max_posts=MAX_POSTS)
    n_val = int(len(ds) * VAL_SPLIT)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # モデル初期化
    sample_rw = ds[0][1]
    rw_dim = sample_rw.size(0)
    model = ChunkedTransformerRW(
        post_dim=POST_DIM,
        chunk_size=CHUNK_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        enc_layers=ENC_LAYERS,
        dropout=DROPOUT,
        rw_dim=rw_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, patience = float("inf"), 0
    for epoch in range(1, EPOCHS+1):
        # — train —
        model.train()
        total_loss = 0.0
        for posts, mask, targets in tqdm(tr_loader, desc=f"Epoch {epoch} [train]"):
            posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(posts, mask)
            loss  = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * posts.size(0)
        train_loss = total_loss / len(tr_loader.dataset)

        # — validation —
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_loader, desc=f"Epoch {epoch} [val]"):
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
                preds = model(posts, mask)
                total_val += criterion(preds, targets).item() * posts.size(0)
        val_loss = total_val / len(va_loader.dataset)

        print(f"Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # early stopping
        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(DEFAULT_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), DEFAULT_SAVE_PATH)
            print(f"  ✔ best model saved → {DEFAULT_SAVE_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val_loss = {best_val:.4f}")

if __name__ == "__main__":
    train()