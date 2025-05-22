#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py ― 投稿ベクトルからランダムウォーク埋め込みを
「3072次元→チャンク化→Transformer → プーリング→予測」する学習スクリプト

★ 特徴次元を chunk_size ごとにチャンク化し、トークンとして扱うことで
  Attention の O(L^2) コストを抑制します。
  • --resume を指定すると、前回のチェックポイントからモデル重み、
    オプティマイザ状態、早期停止情報をまるごと復元して続きから学習します。
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

# ─────────── ハードコード済みパス ───────────
VAST_DIR          = "/workspace/edit_agent/vast"
POSTS_CSV         = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY       = os.path.join(VAST_DIR, "account_vectors.npy")
CHECKPOINT_PATH   = os.path.join(VAST_DIR, "set_transformer_rw.ckpt")

# ─────────── ハイパラ ───────────
POST_DIM      = 3072
CHUNK_SIZE    = 128
D_MODEL       = 256
ENC_LAYERS    = 16
N_HEADS       = 4
DROPOUT       = 0.3

MAX_POSTS     = 50
BATCH_SIZE    = 64
EPOCHS        = 500
LR            = 1e-4
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.1
PATIENCE      = 20
MIN_DELTA     = 1e-4

# ─────────── ユーティリティ ───────────
def parse_vec(s: str, dim: int):
    s = s.strip()
    if not s or s in ("[]", '"[]"'): return None
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
        rw_dict = np.load(acc_npy, allow_pickle=True).item()
        user_posts = defaultdict(list)
        with open(posts_csv, encoding='utf-8') as f:
            rdr = csv.reader(f); next(rdr)
            for uid, _, vec_str in tqdm(rdr, desc="Loading posts"):
                if uid not in rw_dict: continue
                vec = parse_vec(vec_str, POST_DIM)
                if vec is not None:
                    user_posts[uid].append(vec)
        self.samples = []
        for uid, vecs in user_posts.items():
            if not vecs: continue
            vecs = vecs[-max_posts:]
            posts_t = torch.tensor(np.stack(vecs, axis=0), dtype=torch.float32)
            target  = torch.tensor(rw_dict[uid], dtype=torch.float32)
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
    padded = torch.nn.utils.rnn.pad_sequence(posts, batch_first=True, padding_value=0.0)
    mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    targets = torch.stack(targets)
    return padded, mask, targets

class ChunkedTransformerRW(nn.Module):
    def __init__(self, post_dim, chunk_size, d_model,
                 n_heads, enc_layers, dropout, rw_dim):
        super().__init__()
        assert post_dim % chunk_size == 0
        self.chunk_size = chunk_size
        self.num_chunks = post_dim // chunk_size
        self.patch_proj = nn.Linear(chunk_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)
        self.decoder     = nn.Linear(d_model, rw_dim)

    def forward(self, posts, pad_mask):
        B, S, D = posts.size()
        x = posts.view(B, S, self.num_chunks, self.chunk_size)
        x = x.flatten(1, 2)               # (B, S*num_chunks, chunk_size)
        x = self.patch_proj(x)            # (B, T, d_model)
        pm = pad_mask.unsqueeze(-1).expand(B, S, self.num_chunks)
        token_mask = pm.flatten(1, 2)     # (B, T)
        x = x.permute(1, 0, 2)            # (T, B, d_model)
        x = self.transformer(x, src_key_padding_mask=token_mask)
        x = x.permute(1, 0, 2)            # (B, T, d_model)
        valid = (~token_mask).unsqueeze(-1).float()
        sum_vec = (x * valid).sum(dim=1)
        lengths = valid.sum(dim=1).clamp(min=1.0)
        pooled  = sum_vec / lengths
        return self.decoder(pooled)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = RWDataset(POSTS_CSV, ACCOUNT_NPY, max_posts=MAX_POSTS)
    n_val = int(len(ds) * VAL_SPLIT)
    n_tr  = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

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

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    start_epoch = 1
    best_val    = float("inf")
    patience    = 0

    # resume: モデル＋オプティマイザ＋ベストスコア＋patience を復元
    if args.resume and os.path.isfile(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt["best_val"]
        patience    = ckpt["patience"]
        print(f"[Resume] epoch={ckpt['epoch']}  best_val={best_val:.4f}  patience={patience}")

    for epoch in range(start_epoch, EPOCHS+1):
        # train
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

        # val
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for posts, mask, targets in tqdm(va_loader, desc=f"Epoch {epoch} [val]"):
                posts, mask, targets = posts.to(device), mask.to(device), targets.to(device)
                preds = model(posts, mask)
                total_val += criterion(preds, targets).item() * posts.size(0)
        val_loss = total_val / len(va_loader.dataset)

        print(f"Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # checkpoint & early stopping
        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val":    best_val,
                "patience":    patience
            }, CHECKPOINT_PATH)
            print(f"  ✔ saved checkpoint → {CHECKPOINT_PATH}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping."); break

    print(f"[Done] best_val_loss = {best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="load checkpoint and continue training")
    args = parser.parse_args()
    train(args)