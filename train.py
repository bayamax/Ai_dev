#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_se_rw.py ― SE-Block＋MLP による投稿ベクトル→ランダムウォーク埋め込み予測
（ファイルパスはすべてハードコード）
"""

import os
import csv
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# ─────────── ハードコード済みパス ───────────
VAST_DIR           = "/workspace/edit_agent/vast"
POSTS_CSV          = os.path.join(VAST_DIR, "aggregated_posting_vectors.csv")
ACCOUNT_NPY        = os.path.join(VAST_DIR, "account_vectors.npy")
MODEL_SAVE_PATH    = os.path.join(VAST_DIR, "rw_se_predictor.pt")

# ─────────── ハイパラ ───────────
POST_DIM        = 3072
MAX_POSTS       = 50
REDUCTION_RATIO = 16    # SE ブロックの削減比
HIDDEN_DIM      = 512   # MLP の中間次元
LR              = 1e-4
WEIGHT_DECAY    = 1e-5
BATCH_SIZE      = 128
EPOCHS          = 100
VAL_SPLIT       = 0.1
PATIENCE        = 10
MIN_DELTA       = 1e-4

# ─────────── utils: ベクトル文字列パース ───────────
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

# ─────────── データセット ───────────
class MeanPostRWdataset(Dataset):
    def __init__(self):
        # ランダムウォーク埋め込み読み込み
        rw_dict = np.load(ACCOUNT_NPY, allow_pickle=True).item()
        # 投稿をチャンク読み込み
        posts_map = defaultdict(list)
        with open(POSTS_CSV, encoding='utf-8') as f:
            reader = csv.reader(f); next(reader)
            for uid, _, vec_str in tqdm(reader, desc="Loading posts"):
                if uid not in rw_dict:
                    continue
                v = parse_vec(vec_str, POST_DIM)
                if v is not None:
                    posts_map[uid].append(v)
        # サンプル生成
        self.samples = []
        for uid, vecs in posts_map.items():
            if len(vecs) == 0:
                continue
            arr = np.stack(vecs[-MAX_POSTS:], axis=0)
            mean_vec = arr.mean(axis=0).astype(np.float32)
            target   = rw_dict[uid].astype(np.float32)
            self.samples.append((mean_vec, target))
        if not self.samples:
            raise RuntimeError("No samples loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)

# ─────────── Squeeze-and-Excitation ブロック ───────────
class SEBlock(nn.Module):
    def __init__(self, dim, reduction):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)

    def forward(self, x):
        # x: (B, D)
        s = x.mean(dim=0, keepdim=True)       # (1, D)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))        # (1, D)
        return x * s                          # broadcast (B, D)

# ─────────── モデル ───────────
class SE_RW_Predictor(nn.Module):
    def __init__(self, post_dim, hidden_dim, rw_dim, reduction):
        super().__init__()
        self.se      = SEBlock(post_dim, reduction)
        self.predict = nn.Sequential(
            nn.Linear(post_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, rw_dim)
        )

    def forward(self, x):
        # x: (B, post_dim)
        x = self.se(x)
        return self.predict(x)                # (B, rw_dim)

# ─────────── 学習ループ ───────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    ds = MeanPostRWdataset()
    n_val = int(len(ds) * VAL_SPLIT)
    n_tr  = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_tr, n_val])

    tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    rw_dim = ds.samples[0][1].shape[0]
    model = SE_RW_Predictor(
        post_dim=POST_DIM,
        hidden_dim=HIDDEN_DIM,
        rw_dim=rw_dim,
        reduction=REDUCTION_RATIO
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val, patience = float("inf"), 0

    for ep in range(1, EPOCHS + 1):
        # — train —
        model.train()
        sum_loss = 0.0
        for x, y in tqdm(tr_loader, desc=f"Epoch {ep} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = (1 - F.cosine_similarity(pred, y, dim=1)).mean()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * x.size(0)
        train_loss = sum_loss / len(train_ds)

        # — val —
        model.eval()
        sum_val = 0.0
        with torch.no_grad():
            for x, y in tqdm(va_loader, desc=f"Epoch {ep} [val]"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                l = (1 - F.cosine_similarity(pred, y, dim=1)).mean()
                sum_val += l.item() * x.size(0)
        val_loss = sum_val / len(val_ds)

        print(f"Ep {ep}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")
        scheduler.step()

        if val_loss < best_val - MIN_DELTA:
            best_val, patience = val_loss, 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  ✔ saved best →", MODEL_SAVE_PATH)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    print(f"[Done] best_val={best_val:.4f}")

if __name__ == "__main__":
    train()