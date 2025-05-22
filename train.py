#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_post_feature_transformer.py — 投稿ベクトル（3072次元）から
Transformer を使って重要特徴を抽出し、ランダムウォーク埋め込みを予測する
"""

import os, csv, random, argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ─────────── 固定パス ───────────
VAST_DIR       = "/workspace/edit_agent/vast"
POSTS_CSV      = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY    = os.path.join(VAST_DIR, "account_vectors.npy")
CKPT_SAVE_PATH = os.path.join(VAST_DIR, "feature_transformer_post2account.pt")

# ─────────── ハイパラ ───────────
POST_DIM    = 3072   # 投稿ベクトルの次元
D_MODEL     = 256    # Transformer の内部次元
N_HEADS     = 8
N_LAYERS    = 2
DROPOUT     = 0.1

LR          = 5e-4
WD          = 1e-5
EPOCHS      = 50
BATCH_SIZE  = 128
VAL_SPLIT   = 0.1
CLIP_NORM   = 1.0

# ─────────── データセット ───────────
class SinglePostDataset(Dataset):
    def __init__(self, posts_csv, account_npy):
        # ランダムウォーク埋め込みをロード
        rw_dict = np.load(account_npy, allow_pickle=True).item()
        self.user2rw = {u: torch.tensor(v, dtype=torch.float32) 
                         for u,v in rw_dict.items()}
        self.rw_dim = next(iter(self.user2rw.values())).shape[0]

        # CSV を逐次読み込み
        self.samples = []
        with open(posts_csv, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for uid, _, vec_str in tqdm(reader, desc="Loading posts"):
                if uid not in self.user2rw:
                    continue
                # "['0.1', '0.2', ...]" → np.array([...],dtype=float32)
                s = vec_str.strip()
                if s.startswith('"[') and s.endswith(']"'):
                    s = s[1:-1]
                l, r = s.find('['), s.rfind(']')
                if 0<=l<r:
                    s = s[l+1:r]
                arr = np.fromstring(s.replace(',', ' '), dtype=np.float32, sep=' ')
                if arr.size != POST_DIM:
                    continue
                post = torch.from_numpy(arr)  # (POST_DIM,)
                rw   = self.user2rw[uid]      # (rw_dim,)
                self.samples.append((post, rw))

        if not self.samples:
            raise RuntimeError("No valid samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # post: (POST_DIM,), rw: (rw_dim,)
        return self.samples[idx]

def collate_fn(batch):
    posts, rws = zip(*batch)
    posts = torch.stack(posts, dim=0)   # (B, POST_DIM)
    rws   = torch.stack(rws, dim=0)     # (B, rw_dim)
    return posts, rws

# ─────────── モデル ───────────
class FeatureTransformer(nn.Module):
    def __init__(self, post_dim, d_model, n_heads, n_layers, dropout, rw_dim):
        super().__init__()
        # 各特徴量 scalar→d_model に射影
        self.feature_proj = nn.Linear(1, d_model)
        # TransformerEncoder （seq_len=post_dim, d_model）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # プーリング後に rw_dim に写像
        self.output_proj = nn.Linear(d_model, rw_dim)

    def forward(self, posts):
        # posts: (B, POST_DIM)
        B, D = posts.size()
        # reshape → (POST_DIM, B, 1)
        x = posts.transpose(0,1).unsqueeze(-1)
        # project → (POST_DIM, B, d_model)
        x = self.feature_proj(x)
        # self-attn
        x = self.transformer(x)  # (POST_DIM, B, d_model)
        # (B, POST_DIM, d_model)
        x = x.transpose(0,1)
        # mean pool over seq (特徴) 次元
        rep = x.mean(dim=1)      # (B, d_model)
        # 出力 rw_dim
        return self.output_proj(rep)

# ─────────── 学習ループ ───────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = SinglePostDataset(POSTS_CSV, ACCOUNT_NPY)
    N   = len(ds)
    n_val = int(N*VAL_SPLIT)
    n_tr  = N - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val])
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                           collate_fn=collate_fn, num_workers=2)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=2)

    model = FeatureTransformer(
        post_dim=POST_DIM,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        rw_dim=ds.rw_dim
    ).to(device)

    # 出力・目標とも正規化して MSE
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val, patience = float("inf"), 0
    for ep in range(1, EPOCHS+1):
        # — train —
        model.train()
        tot, cnt = 0.0, 0
        for posts, rws in tqdm(tr_loader, desc=f"Epoch {ep} [train]"):
            posts, rws = posts.to(device), rws.to(device)
            optimizer.zero_grad()
            pred = model(posts)            # (B, rw_dim)
            # L2 正規化
            pred = F.normalize(pred, dim=1)
            tgt  = F.normalize(rws, dim=1)
            loss = criterion(pred, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            tot += loss.item() * posts.size(0)
            cnt += posts.size(0)
        train_loss = tot/cnt

        # — valid —
        model.eval()
        vt, vc = 0.0, 0
        with torch.no_grad():
            for posts, rws in va_loader:
                posts, rws = posts.to(device), rws.to(device)
                pred = model(posts)
                pred = F.normalize(pred, dim=1)
                tgt  = F.normalize(rws,   dim=1)
                l = criterion(pred, tgt)
                vt += l.item()*posts.size(0)
                vc += posts.size(0)
        val_loss = vt/vc

        print(f"Ep{ep}/{EPOCHS} train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(CKPT_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), CKPT_SAVE_PATH)
            print("  ✔ saved best →", CKPT_SAVE_PATH)
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping."); break
        scheduler.step()

    print(f"[Done] best_val={best_val:.4f}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patience", type=int, default=5)
    args = p.parse_args()
    train(args)